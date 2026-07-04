"""Aggregate-only metrics artifacts (ADR-0007 decision 7).

These are the committed, publishable numbers: counts, rates, seeded CIs, per-segment breakdowns,
judge fingerprints, and a provenance hash tying every number to a specific cache state. They carry
NO raw prompt/response text — the classified text stays in the gitignored generation cache. The
writer is byte-stable (sorted keys + floats pre-rounded in metrics) so re-running the report on the
same cache reproduces an identical file.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class MetricResult(_Frozen):
    name: str
    point: float
    ci_low: float
    ci_high: float
    n: int
    extra: dict = Field(default_factory=dict)


class SegmentResult(_Frozen):
    metric: str
    segment: str
    point: float
    ci_low: float
    ci_high: float
    n: int


class MetricsArtifact(_Frozen):
    experiment_id: str
    condition_id: str
    suite: str
    split: str
    policy_model_id: str
    n: int
    metrics: list[MetricResult]
    segments: list[SegmentResult] = Field(default_factory=list)
    judge_fingerprints: dict = Field(default_factory=dict)
    bootstrap: dict = Field(default_factory=dict)  # {seed, n, percentile_method}
    manifest_hash: str = ""
    config_hash: str = ""
    provenance_hash: str = ""
    schema_version: int = 1

    def to_json(self) -> str:
        """Byte-stable serialisation: sorted keys, floats already rounded by the metrics pass."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def write_artifact(artifact: MetricsArtifact, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.to_json(), encoding="utf-8")
    return path
