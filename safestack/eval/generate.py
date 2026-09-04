"""PASS A: batch generation over prepared eval suites — content-hash cached, with traces.

Templated on ``runner.run_experiment`` (reuses ``set_seeds``, the ``RunRecord`` provenance block,
``TraceWriter``, and the ``gateway.generate()``/``close()`` finally) but iterates whole suites and
caches every generation by content hash, so a killed run resumes from the cache (ADR-0007). The
policy gateway is closed before any judge pass loads — the "never two large models resident"
invariant becomes a process boundary (ADR-0003).
"""

from __future__ import annotations

import logging
import platform as _platform
import uuid
from pathlib import Path

from safestack import __version__
from safestack.config import Message
from safestack.datasets.schema import EvalRecord
from safestack.datasets.validate import validate_manifest
from safestack.determinism import set_seeds
from safestack.eval.cache import ContentHashStore, GenerationCacheEntry
from safestack.eval.config import EvalExperimentConfig, load_eval_config
from safestack.eval.guards import reject_api_backend
from safestack.guardrails import build_guardrail
from safestack.hashing import canonical_json, content_hash, model_fingerprint
from safestack.model_gateway import GenerationRequest, build_gateway
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec
from safestack.runner import git_commit
from safestack.tracing import RunRecord, TraceRecord, TraceWriter, hash_text, redact

log = logging.getLogger("safestack")


def _manifest_path(data_dir: Path, suite: str) -> Path:
    return data_dir / "manifests" / f"{suite}.yaml"


def _read_records(data_dir: Path, manifest) -> list[EvalRecord]:
    path = data_dir / "prepared" / manifest.split / f"{manifest.name}.jsonl"
    records: list[EvalRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(EvalRecord.model_validate_json(line))
    return records


def _pipeline_total_ms(
    gen_ms: float | None,
    input_ms: float | None,
    output_ms: float | None,
    blocked_at: str | None,
) -> float | None:
    """Served-path wall-time for one item. An INPUT block never reaches generation in a deployment,
    so it costs only the input screen (``input_ms``); every other served path runs generation and
    counts it plus whatever guardrail screens ran. Returns None when nothing was timed (C1 "none"),
    matching the pre-guardrail C1 trace. Consistent with the shipped C3 (gen + output): generation
    counts iff the served path reaches it."""
    if blocked_at == "input":
        return input_ms
    screens = [ms for ms in (input_ms, output_ms) if ms is not None]
    if gen_ms is None or not screens:
        return None
    return gen_ms + sum(screens)


def run_suite(
    config: str | Path | EvalExperimentConfig,
    *,
    backend_override: str | None = None,
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    cache_dir: str | Path | None = None,
    limit: int | None = None, batch_size: int = 32,
) -> Path:
    """Generate every prepared record of ``cfg.suites`` through the policy gateway (caching each
    output by content hash), then run the configured guardrail as a separate pass and write one
    trace per record with its blocked_at / final_response decision. Returns the run directory.

    The trace write is DEFERRED until after ``gateway.close()`` so the guardrail scores the cached
    generations while the policy model is unloaded (ADR-0003). The input pre-pass runs first and an
    input block short-circuits the output check (the C4 control flow); ``safestack/eval/metrics.py``
    is unchanged because it already reads ``blocked_at`` off the trace, input or output (ADR-0009).
    """
    cfg = config if isinstance(config, EvalExperimentConfig) else load_eval_config(str(config))
    # ADR-0017 dec.7: fail closed before ANY generation if a card resolves to backend 'api' (covers
    # the --backend override) -- harmful content must never leave the self-hosted box.
    reject_api_backend(cfg, backend_override=backend_override, models_dir=models_dir)
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "cache"
    spec = resolve_model_spec(cfg.model, backend_override=backend_override, models_dir=models_dir)
    set_seeds(cfg.decode.seed)

    # Build the guardrail up front so an unsupported placement or a rule-4 violation fails BEFORE
    # generation; any heavy model it holds loads lazily in the output pass, after the policy is
    # freed. The outer try/finally below guarantees the guardrail is closed even on a generation
    # error.
    guardrail = build_guardrail(cfg, models_dir=models_dir, backend_override=backend_override)

    gen_store = ContentHashStore(cache_dir, "generations")
    fingerprint = model_fingerprint(spec)
    decode_dump = cfg.decode.model_dump(mode="json")
    run_id = uuid.uuid4().hex
    writer = TraceWriter(Path(runs_dir) / run_id)

    n_hits = n_misses = 0
    # Only (rec, content_hash) is held for the deferred trace write; the generation text is re-read
    # from the cache in the output pass, so peak memory stays flat regardless of suite size.
    pending: list[tuple[EvalRecord, str]] = []
    try:
        gateway = build_gateway(spec)
        try:
            # Cache MISSES are buffered and generated in batches of batch_size (one batched forward
            # pass fills idle GPU; ~5-20x). Each generation is still cached per content_hash,
            # so a killed run resumes item-by-item exactly as the single-item path did.
            miss_batch: list[tuple[EvalRecord, tuple[Message, ...], str]] = []
            queued: set[str] = set()  # content_hashes cached on disk OR buffered this run (dedup)

            def _flush() -> None:
                nonlocal n_misses
                if not miss_batch:
                    return
                reqs = [GenerationRequest(messages=m, params=cfg.decode) for _, m, _ in miss_batch]
                for (m_rec, m_msgs, m_ch), result in zip(
                    miss_batch, gateway.generate_batch(reqs), strict=True
                ):
                    gen_store.put(
                        m_ch,
                        GenerationCacheEntry(
                            content_hash=m_ch,
                            eval_id=m_rec.eval_id,
                            suite=m_rec.suite,
                            split=m_rec.split,
                            model_fingerprint=fingerprint,
                            decode=decode_dump,
                            messages=[{"role": mm.role, "content": mm.content} for mm in m_msgs],
                            text=result.text,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            finish_reason=result.finish_reason,
                            generation_ms=result.generation_ms,
                        ),
                    )
                    n_misses += 1
                miss_batch.clear()

            for suite in cfg.suites:
                manifest = validate_manifest(_manifest_path(data_dir, suite), data_dir=data_dir)
                records = _read_records(data_dir, manifest)
                if limit is not None:
                    records = records[:limit]
                for rec in records:
                    messages = (Message(role="user", content=rec.prompt),)
                    ch = content_hash(fingerprint, messages, cfg.decode)
                    if gen_store.get(ch) is not None or ch in queued:
                        n_hits += 1  # cached on disk, or a duplicate already queued this run
                    else:
                        queued.add(ch)
                        miss_batch.append((rec, messages, ch))
                        if len(miss_batch) >= batch_size:
                            _flush()
                    pending.append((rec, ch))
            _flush()  # generate the final partial batch before the guardrail pass reads the cache
        finally:
            gateway.close()  # free the policy model before the guardrail pass loads (ADR-0003)

        # Guardrail pass: re-read each cached generation now the policy model is unloaded and write
        # its trace with the guardrail decision. The input pre-pass runs first; an input block
        # SHORT-CIRCUITS the output check (a refused prompt is never output-screened -- the C4
        # flow). Each guardrail self-gates on placement, so C1 "none" passes both stages through.
        for rec, ch in pending:
            gen = gen_store.get(ch)
            text = gen["text"]
            gen_ms = gen.get("generation_ms")
            input_ms = output_ms = None
            blocked_at = None
            final_response = text  # a passing item returns the cached generation unchanged
            in_dec = guardrail.check_input(rec.prompt)
            input_ms = in_dec.guardrail_ms
            if in_dec.blocked_at is not None:
                blocked_at = in_dec.blocked_at
                final_response = in_dec.final_response
            else:
                out_dec = guardrail.check_output(rec.prompt, text)
                output_ms = out_dec.guardrail_ms
                blocked_at = out_dec.blocked_at
                final_response = out_dec.final_response
            total_ms = _pipeline_total_ms(gen_ms, input_ms, output_ms, blocked_at)
            writer.append_trace(
                TraceRecord(
                    trace_id=uuid.uuid4().hex,
                    run_id=run_id,
                    model_id=spec.model_id,
                    backend=spec.backend,
                    adapter_id=spec.adapter,
                    content_hash=ch,
                    prompt=redact(rec.prompt, public_log=False),
                    output=redact(text, public_log=False),
                    decode=decode_dump,
                    seed=cfg.decode.seed,
                    eval_id=rec.eval_id,
                    suite=rec.suite,
                    split=rec.split,
                    condition_id=cfg.condition_id,
                    guardrail_config=cfg.guardrail_config,
                    blocked_at=blocked_at,
                    final_response=redact(final_response, public_log=False),
                    input_guardrail_ms=input_ms,
                    output_guardrail_ms=output_ms,
                    generation_ms=gen_ms,
                    total_ms=total_ms,
                )
            )
    finally:
        guardrail.close()  # always freed, even if generation raised before the output pass

    config_dump = cfg.model_dump(mode="json")
    accelerator, library_versions = _runtime_provenance()
    writer.write_run(
        RunRecord(
            run_id=run_id,
            experiment_id=cfg.experiment_id,
            config=config_dump,
            model_spec=spec.model_dump(mode="json"),
            seed=cfg.decode.seed,
            config_hash=hash_text(canonical_json(config_dump)),
            package_version=__version__,
            python_version=_platform.python_version(),
            platform=_platform.platform(),
            git_commit=git_commit(),
            n_generations=n_hits + n_misses,
            n_cache_hits=n_hits,
            n_cache_misses=n_misses,
            accelerator=accelerator,
            library_versions=library_versions,
        )
    )
    log.info(
        "eval run %s | experiment=%s | model=%s | suites=%d | hits=%d misses=%d",
        run_id,
        cfg.experiment_id,
        spec.model_id,
        len(cfg.suites),
        n_hits,
        n_misses,
    )
    return writer.run_dir


def _runtime_provenance() -> tuple[str | None, dict]:
    """The GPU name + torch/transformers/bitsandbytes versions for run.json (ADR-0007 decision 6).

    Returns (None, {}) on the torch-free base install so the mock / CPU path stays torch-free.
    """
    try:
        import torch
    except ImportError:
        return (None, {})
    versions = {"torch": torch.__version__}
    for name in ("transformers", "bitsandbytes"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            pass
    accelerator = None
    try:
        if torch.cuda.is_available():
            accelerator = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return (accelerator, versions)
