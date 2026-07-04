"""Config-driven orchestration. Importable and testable without the CLI; satisfies both
Phase 0 exit criteria (generate one prompt; load and log one config)."""

from __future__ import annotations

import logging
import platform as _platform
import subprocess
import uuid
from pathlib import Path

from safestack import __version__
from safestack.determinism import set_seeds
from safestack.hashing import canonical_json
from safestack.model_gateway import GenerationRequest, GenerationResult, build_gateway
from safestack.registry import DEFAULT_MODELS_DIR, load_experiment_config, resolve_model_ref
from safestack.tracing import RunRecord, TraceRecord, TraceWriter, hash_text, redact

log = logging.getLogger("safestack")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _build_request(cfg) -> GenerationRequest:
    if cfg.messages is not None:
        return GenerationRequest(messages=tuple(cfg.messages), params=cfg.decode)
    return GenerationRequest.from_prompt(cfg.prompt, cfg.decode)


def run_experiment(
    config_path: str | Path,
    *,
    backend_override: str | None = None,
    runs_dir: str | Path = "runs",
    dry_run: bool = False,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> GenerationResult | None:
    cfg = load_experiment_config(config_path)  # exit criterion 2a: config loaded + validated
    spec = resolve_model_ref(cfg, backend_override=backend_override, models_dir=models_dir)
    set_seeds(cfg.decode.seed)

    run_id = uuid.uuid4().hex
    writer = TraceWriter(Path(runs_dir) / run_id)
    config_dump = cfg.model_dump(mode="json")
    run_record = RunRecord(
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
        n_generations=0 if dry_run else 1,
    )
    writer.write_run(run_record)  # exit criterion 2b: config logged
    log.info(
        "run %s | experiment=%s | model=%s | backend=%s | dry_run=%s",
        run_id,
        cfg.experiment_id,
        spec.model_id,
        spec.backend,
        dry_run,
    )
    if dry_run:
        return None

    request = _build_request(cfg)
    gateway = build_gateway(spec)
    try:
        result = gateway.generate(request)  # exit criterion 1: one prompt through the gateway
    finally:
        gateway.close()

    rendered = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    writer.append_trace(
        TraceRecord(
            trace_id=uuid.uuid4().hex,
            run_id=run_id,
            model_id=result.model_id,
            backend=result.backend,
            content_hash=result.content_hash,
            prompt=redact(rendered, public_log=False),
            output=redact(result.text, public_log=False),
            decode=cfg.decode.model_dump(mode="json"),
            seed=cfg.decode.seed,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_ms=result.generation_ms,
            finish_reason=result.finish_reason,
        )
    )
    return result
