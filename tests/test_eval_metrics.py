"""Step 5: metrics + bootstrap CIs + aggregate-only artifact. Base install (no model, no numpy)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safestack.config import ModelSpec
from safestack.eval.artifacts import MetricsArtifact
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.judges.mock import MockSafetyJudge
from safestack.eval.metrics import _percentile, bootstrap_ci, suite_metrics

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg(**over: object) -> EvalExperimentConfig:
    base = dict(
        experiment_id="metrics_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=2000,
    )
    base.update(over)
    return EvalExperimentConfig(**base)


def _run_and_judge(tmp_path: Path, cfg: EvalExperimentConfig) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    return run_dir, cache


def _metric(art: MetricsArtifact, name: str):
    return next(m for m in art.metrics if m.name == name)


# ---- pure bootstrap unit tests (golden values pin the RNG + percentile method) ----


def test_bootstrap_golden_values() -> None:
    assert bootstrap_ci([1.0, 0, 1, 1, 0, 0, 1, 0, 1, 1], n=2000, seed=0) == (0.6, 0.3, 0.9)
    assert bootstrap_ci([5.0, 3, 4, 1, 2, 5, 3, 4], n=2000, seed=0) == (3.375, 2.5, 4.25)
    assert bootstrap_ci([1.0], n=2000, seed=0) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([], n=2000, seed=0) == (0.0, 0.0, 0.0)
    assert _percentile([0, 1, 2, 3, 4], 50) == 2.0


def test_ci_brackets_point() -> None:
    point, lo, hi = bootstrap_ci([1.0, 0, 0, 1, 0, 1, 0, 0], n=1500, seed=3)
    assert lo <= point <= hi


# ---- exact metrics off deterministic mock labels ----


def test_exact_asr_and_fnr(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    art = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)

    # Independently recompute expected ASR from the mock judge rule over the recorded outputs.
    judge = MockSafetyJudge(fingerprint={}, judge_prompt_version="v1")
    outputs = [
        json.loads(ln)["output"]
        for ln in (run_dir / "traces.jsonl").read_text().splitlines()
        if ln.strip() and json.loads(ln)["suite"] == "harmful_fixture"
    ]
    expected = sum(judge.score("", o).label == "unsafe" for o in outputs) / len(outputs)
    assert _metric(art, "asr").point == round(expected, 6)  # artifact rounds for byte-stability
    assert _metric(art, "asr").extra["n_unparsed"] == 0
    assert _metric(art, "guardrail_fnr").point == 1.0  # C1: guardrail lets everything through
    assert art.n == 6


def test_overrefusal_and_helpfulness(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    orr = suite_metrics(run_dir, cfg, "overrefusal_fixture", data_dir=FIX, cache_dir=cache)
    assert _metric(orr, "over_refusal").point == 0.0  # mock outputs contain no refusal markers
    assert _metric(orr, "guardrail_fpr").point == 0.0  # C1: nothing blocked

    helped = suite_metrics(run_dir, cfg, "helpfulness_fixture", data_dir=FIX, cache_dir=cache)
    h = _metric(helped, "benign_helpfulness")
    assert 1.0 <= h.point <= 5.0
    assert h.extra["answer_rate"] == 1.0  # no refusals among mock outputs


def test_paired_reporting_enforced(tmp_path: Path) -> None:
    lonely = _cfg(suites=["harmful_fixture"])  # ASR with no benign suites
    run_dir, cache = _run_and_judge(tmp_path, lonely)
    with pytest.raises(ValueError, match="rule 5"):
        suite_metrics(run_dir, lonely, "harmful_fixture", data_dir=FIX, cache_dir=cache)


def test_artifact_is_byte_stable_and_leak_free(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    a = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    b = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    assert a.to_json() == b.to_json()  # deterministic, byte-identical
    # Aggregate-only: no raw prompt text leaks into the artifact.
    assert "PLACEHOLDER harmful request" not in a.to_json()
    assert a.provenance_hash.startswith("sha256:")
