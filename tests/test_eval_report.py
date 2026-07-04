"""Step 6: report writing + compare table (CI-overlap significance + dynamic-range gate)."""

from __future__ import annotations

import json
from pathlib import Path

from safestack.config import ModelSpec
from safestack.eval.artifacts import MetricResult, MetricsArtifact, write_artifact
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.report import compare, gate_readout, write_report

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg() -> EvalExperimentConfig:
    return EvalExperimentConfig(
        experiment_id="report_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=1000,
    )


def _m(name: str, point: float, lo: float, hi: float) -> MetricResult:
    return MetricResult(name=name, point=point, ci_low=lo, ci_high=hi, n=10)


def _art(cond: str, metrics: list[MetricResult]) -> MetricsArtifact:
    return MetricsArtifact(
        experiment_id="e",
        condition_id=cond,
        suite="harmful_fixture",
        split="eval_harmful",
        policy_model_id="m",
        n=10,
        metrics=metrics,
    )


def test_write_report_is_aggregate_only(tmp_path: Path) -> None:
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
    assert out.name == "report_smoke__harmful_fixture__C1.json"
    text = out.read_text(encoding="utf-8")
    assert "PLACEHOLDER harmful request" not in text  # no raw prompt text
    doc = json.loads(text)
    assert {m["name"] for m in doc["metrics"]} == {"asr", "guardrail_fnr"}


def test_compare_table_and_overlap_note(tmp_path: Path) -> None:
    a = write_artifact(_art("C1", [_m("asr", 0.60, 0.45, 0.75)]), tmp_path / "a.json")
    b = write_artifact(
        _art("C2", [_m("asr", 0.55, 0.40, 0.70)]), tmp_path / "b.json"
    )  # overlaps C1
    text = compare([a, b], fmt="md")
    assert "| condition | suite | metric |" in text
    assert "no significant difference: C1 vs C2 on asr" in text

    c = write_artifact(_art("C3", [_m("asr", 0.10, 0.02, 0.18)]), tmp_path / "c.json")  # disjoint
    text2 = compare([a, c], fmt="csv")
    assert "condition,suite,metric" in text2
    assert "no significant difference" not in text2


def test_gate_readout_verdicts() -> None:
    keep = gate_readout([_art("C1", [_m("asr", 0.60, 0.45, 0.75)])])
    assert "KEEP" in keep
    switch = gate_readout([_art("Cx", [_m("asr", 0.20, 0.10, 0.30)])])
    assert "SWITCH" in switch
    borderline = gate_readout([_art("Cy", [_m("asr", 0.35, 0.25, 0.45)])])
    assert "BORDERLINE" in borderline


def test_compare_gate_flag_includes_readout(tmp_path: Path) -> None:
    a = write_artifact(_art("C1", [_m("asr", 0.60, 0.45, 0.75)]), tmp_path / "a.json")
    text = compare([a], gate=True)
    assert "Dynamic-range gate (ADR-0002)" in text
    assert "ASR = 0.6" in text
