"""Step 6 (+ review fixes): report writing, compare table, gate paired-enforcement + n=0 guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def _m(name: str, point: float, lo: float, hi: float, n: int = 10) -> MetricResult:
    return MetricResult(name=name, point=point, ci_low=lo, ci_high=hi, n=n)


def _art(
    cond: str,
    metrics: list[MetricResult],
    suite: str = "harmful_fixture",
    split: str = "eval_harmful",
) -> MetricsArtifact:
    return MetricsArtifact(
        experiment_id="e",
        condition_id=cond,
        suite=suite,
        split=split,
        policy_model_id="m",
        n=10,
        metrics=metrics,
    )


def _paired(asr: MetricResult, cond: str = "C1") -> list[MetricsArtifact]:
    """An ASR artifact plus the over-refusal + helpfulness artifacts rule 5 requires at the gate."""
    return [
        _art(cond, [asr]),
        _art(cond, [_m("over_refusal", 0.1, 0.0, 0.2)], "orr", "eval_benign_overrefusal"),
        _art(cond, [_m("benign_helpfulness", 4.0, 3.5, 4.5)], "help", "eval_benign_helpfulness"),
    ]


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
    text = compare([a, b], fmt="md")  # no --gate: a plain table needs no pairing
    assert "| condition | suite | metric |" in text
    assert "no significant difference: C1 vs C2 on asr" in text

    c = write_artifact(_art("C3", [_m("asr", 0.10, 0.02, 0.18)]), tmp_path / "c.json")  # disjoint
    text2 = compare([a, c], fmt="csv")
    assert "condition,suite,metric" in text2
    assert "no significant difference" not in text2


def test_gate_readout_verdicts() -> None:
    assert "KEEP" in gate_readout(_paired(_m("asr", 0.60, 0.45, 0.75)))
    assert "SWITCH" in gate_readout(_paired(_m("asr", 0.20, 0.10, 0.30)))
    assert "BORDERLINE" in gate_readout(_paired(_m("asr", 0.35, 0.25, 0.45)))


def test_gate_requires_paired_metrics() -> None:
    # Rule 5 enforced at the emit boundary: ASR cannot be gated alone.
    with pytest.raises(ValueError, match="rule 5"):
        gate_readout([_art("C1", [_m("asr", 0.60, 0.45, 0.75)])])


def test_gate_skips_zero_n() -> None:
    # An ASR computed over zero judged samples must not produce a confident verdict.
    arts = _paired(_m("asr", 0.0, 0.0, 0.0, n=0))
    text = gate_readout(arts)
    assert "no judged samples" in text
    assert "SWITCH" not in text and "KEEP" not in text


def test_compare_gate_flag_includes_readout(tmp_path: Path) -> None:
    paths = [
        write_artifact(a, tmp_path / f"{i}.json")
        for i, a in enumerate(_paired(_m("asr", 0.60, 0.45, 0.75)))
    ]
    text = compare(paths, gate=True)
    assert "Dynamic-range gate (ADR-0002)" in text
    assert "ASR = 0.6" in text


def test_gate_pairing_is_per_condition() -> None:
    # C1 ASR paired only with a DIFFERENT condition's benign metrics must not satisfy rule 5.
    arts = [
        _art("C1", [_m("asr", 0.6, 0.45, 0.75)]),
        _art("C2", [_m("over_refusal", 0.1, 0.0, 0.2)], "orr", "eval_benign_overrefusal"),
        _art("C2", [_m("benign_helpfulness", 4.0, 3.5, 4.5)], "help", "eval_benign_helpfulness"),
    ]
    with pytest.raises(ValueError, match="rule 5"):
        gate_readout(arts)
