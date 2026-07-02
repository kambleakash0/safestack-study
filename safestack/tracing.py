"""Run/generation records + JSONL writer + redaction hooks.

Records are a forward-compatible SUBSET of the master plan section 13.4 serving trace and
use its field names, so later phases only ADD fields. Reserved fields are declared but not
populated in Phase 0. Full text stays under gitignored `runs/` (RESPONSIBLE_USE section 6.2).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def hash_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def redact(text: str, public_log: bool) -> str:
    """Full text when private (default); a content hash when marked for public export."""
    return hash_text(text) if public_log else text


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RunRecord(BaseModel):
    run_id: str
    experiment_id: str
    created_at: str = Field(default_factory=_utc_now)
    config: dict
    model_spec: dict
    seed: int
    config_hash: str
    package_version: str
    python_version: str
    platform: str
    git_commit: str | None = None
    n_generations: int = 0
    schema_version: int = SCHEMA_VERSION


class TraceRecord(BaseModel):
    trace_id: str
    run_id: str
    model_id: str
    backend: str
    content_hash: str
    prompt: str
    output: str
    decode: dict
    seed: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_ms: float | None = None
    finish_reason: str | None = None
    created_at: str = Field(default_factory=_utc_now)
    public_log: bool = False
    schema_version: int = SCHEMA_VERSION
    # Reserved for later phases (section 13.4); declared, not populated in Phase 0.
    condition_id: str | None = None
    adapter_id: str | None = None
    guardrail_config: str | None = None
    input_guardrail_ms: float | None = None
    output_guardrail_ms: float | None = None
    total_ms: float | None = None
    blocked_at: str | None = None
    judge_label: str | None = None


class TraceWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_run(self, record: RunRecord) -> Path:
        path = self.run_dir / "run.json"
        path.write_text(record.model_dump_json(indent=2))
        return path

    def append_trace(self, record: TraceRecord) -> None:
        with (self.run_dir / "traces.jsonl").open("a") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()
