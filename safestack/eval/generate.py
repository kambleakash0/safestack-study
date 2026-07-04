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
from safestack.hashing import canonical_json, content_hash, model_fingerprint
from safestack.model_gateway import GenerationRequest, build_gateway
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec
from safestack.runner import _git_commit
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


def run_suite(
    config: str | Path | EvalExperimentConfig,
    *,
    backend_override: str | None = None,
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> Path:
    """Generate every prepared record of ``cfg.suites`` through the policy gateway, caching each
    output by content hash and writing one trace per record. Returns the run directory."""
    cfg = config if isinstance(config, EvalExperimentConfig) else load_eval_config(str(config))
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "cache"
    spec = resolve_model_spec(cfg.model, backend_override=backend_override, models_dir=models_dir)
    set_seeds(cfg.decode.seed)

    gen_store = ContentHashStore(cache_dir, "generations")
    fingerprint = model_fingerprint(spec)
    decode_dump = cfg.decode.model_dump(mode="json")
    run_id = uuid.uuid4().hex
    writer = TraceWriter(Path(runs_dir) / run_id)

    n_hits = n_misses = 0
    gateway = build_gateway(spec)
    try:
        for suite in cfg.suites:
            manifest = validate_manifest(_manifest_path(data_dir, suite), data_dir=data_dir)
            records = _read_records(data_dir, manifest)
            if limit is not None:
                records = records[:limit]
            for rec in records:
                messages = (Message(role="user", content=rec.prompt),)
                ch = content_hash(fingerprint, messages, cfg.decode)
                cached = gen_store.get(ch)
                if cached is None:
                    n_misses += 1
                    result = gateway.generate(
                        GenerationRequest(messages=messages, params=cfg.decode)
                    )
                    gen_store.put(
                        ch,
                        GenerationCacheEntry(
                            content_hash=ch,
                            eval_id=rec.eval_id,
                            suite=rec.suite,
                            split=rec.split,
                            model_fingerprint=fingerprint,
                            decode=decode_dump,
                            messages=[{"role": m.role, "content": m.content} for m in messages],
                            text=result.text,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            finish_reason=result.finish_reason,
                            generation_ms=result.generation_ms,
                        ),
                    )
                    text = result.text
                else:
                    n_hits += 1
                    text = cached["text"]
                writer.append_trace(
                    TraceRecord(
                        trace_id=uuid.uuid4().hex,
                        run_id=run_id,
                        model_id=spec.model_id,
                        backend=spec.backend,
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
                        blocked_at=None,  # C1: no guardrail, nothing is ever blocked
                        final_response=redact(text, public_log=False),  # identity transform in C1
                    )
                )
    finally:
        gateway.close()  # free the policy model before any judge pass loads (ADR-0003)

    config_dump = cfg.model_dump(mode="json")
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
            git_commit=_git_commit(),
            n_generations=n_hits + n_misses,
            n_cache_hits=n_hits,
            n_cache_misses=n_misses,
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
