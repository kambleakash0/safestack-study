"""Step 7: responsible-use guard for the eval harness.

Metrics artifacts are aggregate-only and never carry raw prompt/response text, and any committed
report guards against leaking harmful text. Mirrors the dataset-layer leakage tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.report import write_report

FIX = Path("tests/fixtures/eval_suites")
REPORTS = Path("reports")
# Keys that would carry raw text; an aggregate-only artifact must contain none of them.
_FORBIDDEN_KEYS = {"prompt", "output", "messages", "text", "response", "raw_first_line"}


def _all_keys(obj: object) -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def _cfg() -> EvalExperimentConfig:
    return EvalExperimentConfig(
        experiment_id="leak_check",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"],
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=500,
    )


def test_metrics_artifact_is_aggregate_only(tmp_path: Path) -> None:
    cfg = _cfg()
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    out = write_report(
        run_dir,
        cfg,
        "harmful_fixture",
        reports_dir=tmp_path / "reports",
        data_dir=FIX,
        cache_dir=cache,
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert _FORBIDDEN_KEYS.isdisjoint(set(_all_keys(doc)))
    assert "PLACEHOLDER harmful request" not in out.read_text(encoding="utf-8")


def test_harmful_fixture_samples_are_hashed() -> None:
    lines = (
        (FIX / "public_sanitized_examples" / "harmful_fixture.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    for ln in lines:
        if ln.strip():
            rec = json.loads(ln)
            assert rec["prompt"].startswith("sha256:")  # public_release=false -> prompt hashed
            assert "PLACEHOLDER harmful request" not in rec["prompt"]


def test_committed_reports_contain_no_harmful_text() -> None:
    """Guards future committed C1 artifacts: aggregate-only, no raw prompt/response keys."""
    metrics_dir = REPORTS / "metrics"
    if not metrics_dir.exists():
        return  # nothing committed yet; the invariant still holds vacuously
    for f in metrics_dir.glob("*.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        assert _FORBIDDEN_KEYS.isdisjoint(set(_all_keys(doc))), f"raw-text key in {f}"
