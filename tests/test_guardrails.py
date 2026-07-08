"""Phase 2 / C3: guardrail subpackage + the output-guardrail seam (mock-first, base install).

Covers the Guardrail ABC + mock + build_guardrail factory (incl. the ADR-0004 rule-4 guard), the
run_suite output pass (blocked_at / final_response / timings land on the trace), and a regression
proving the metrics layer is untouched: with a real block pattern, guardrail_fpr>0, guardrail_fnr<1,
and ASR drops vs the C1 baseline -- all read straight off blocked_at (ADR-0009).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.metrics import suite_metrics
from safestack.guardrails import (
    SAFE_REFUSAL,
    NullGuardrail,
    build_guardrail,
)
from safestack.guardrails.mock import MockOutputGuardrail

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg(**over: object) -> EvalExperimentConfig:
    base = dict(
        experiment_id="guard_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=2000,
    )
    base.update(over)
    return EvalExperimentConfig(**base)


def _out_cfg(**over: object) -> EvalExperimentConfig:
    defaults: dict = dict(
        condition_id="C3",
        guardrail_config="output",
        output_guardrail=ModelSpec(model_id="mock_guard", backend="mock"),
    )
    defaults.update(over)
    return _cfg(**defaults)


# ---- base + mock units ----


def test_null_guardrail_is_passthrough() -> None:
    g = NullGuardrail()
    out = g.check_output("prompt", "a response")
    assert out.blocked_at is None
    assert out.final_response == "a response"
    assert out.guardrail_ms is None
    # The base input stage defaults to pass-through (no input guardrail wired here).
    assert g.check_input("prompt").blocked_at is None


def test_mock_output_guardrail_deterministic_and_spans() -> None:
    g = MockOutputGuardrail()
    responses = [f"response number {i}" for i in range(12)]
    decisions = [g.check_output("p", r) for r in responses]
    # Deterministic: a second pass gives identical verdicts.
    assert [d.blocked_at for d in decisions] == [
        g.check_output("p", r).blocked_at for r in responses
    ]
    blocked = [d for d in decisions if d.blocked_at is not None]
    passed = [d for d in decisions if d.blocked_at is None]
    assert blocked and passed  # the mock spans both outcomes over a normal set of responses
    for d in blocked:
        assert d.blocked_at == "output"
        assert d.final_response == SAFE_REFUSAL
    for d, r in zip(decisions, responses, strict=True):
        if d.blocked_at is None:
            assert d.final_response == r  # passed responses are returned unchanged


# ---- factory ----


def test_build_none_returns_null() -> None:
    assert isinstance(build_guardrail(_cfg()), NullGuardrail)


def test_build_output_mock() -> None:
    assert isinstance(build_guardrail(_out_cfg()), MockOutputGuardrail)


def test_build_output_requires_a_model() -> None:
    with pytest.raises(ValueError, match="requires cfg.output_guardrail"):
        build_guardrail(_cfg(guardrail_config="output"))


@pytest.mark.parametrize("placement", ["input", "input_output"])
def test_build_input_placements_not_built_yet(placement: str) -> None:
    with pytest.raises(NotImplementedError, match="not built yet"):
        build_guardrail(_cfg(guardrail_config=placement))


def test_build_real_backend_not_built_yet() -> None:
    cfg = _out_cfg(output_guardrail=ModelSpec(model_id="granite_guardian_2b", backend="hf_local"))
    with pytest.raises(NotImplementedError, match="Granite Guardian"):
        build_guardrail(cfg)


def test_rule4_guardrail_may_not_be_the_safety_judge() -> None:
    # Guardrail model == the safety judge -> circular, rejected (ADR-0004 rule 4).
    cfg = _out_cfg(
        safety_judge=ModelSpec(model_id="shared_guard", backend="mock"),
        output_guardrail=ModelSpec(model_id="shared_guard", backend="mock"),
    )
    with pytest.raises(ValueError, match="rule 4"):
        build_guardrail(cfg)


# ---- run_suite output pass: the seam ----


def _traces(run_dir: Path) -> list[dict]:
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def test_output_pass_writes_block_decisions_to_traces(tmp_path: Path) -> None:
    run_dir = run_suite(
        _out_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=tmp_path / "c"
    )
    traces = _traces(run_dir)
    blocked = [t for t in traces if t["blocked_at"] is not None]
    passed = [t for t in traces if t["blocked_at"] is None]
    assert blocked and passed  # the mock guardrail spans both across the fixtures
    for t in traces:
        assert t["condition_id"] == "C3"
        assert t["guardrail_config"] == "output"
        assert t["output_guardrail_ms"] == 0.0  # the mock stamps a zero-cost timing
        assert t["total_ms"] is not None
    for t in blocked:
        assert t["blocked_at"] == "output"
        assert t["final_response"] == SAFE_REFUSAL  # blocked -> canned refusal
        assert t["output"] != SAFE_REFUSAL  # the pre-block response is preserved (private trace)
    for t in passed:
        assert t["final_response"] == t["output"]  # passed -> response unchanged


def test_c1_none_trace_invariants_unchanged(tmp_path: Path) -> None:
    # The deferred-write refactor must not disturb the C1 pass-through (ADR-0008 anchors).
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=tmp_path / "c")
    for t in _traces(run_dir):
        assert t["guardrail_config"] == "none"
        assert t["blocked_at"] is None
        assert t["final_response"] == t["output"]
        assert t["output_guardrail_ms"] is None
        assert t["total_ms"] is None


# ---- metrics regression: NO metrics.py change ----


def _run_judge_metrics(tmp_path: Path, cfg: EvalExperimentConfig, sub: str, suite: str):
    cache = tmp_path / sub
    run_dir = run_suite(cfg, runs_dir=tmp_path / f"{sub}_runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    return suite_metrics(run_dir, cfg, suite, data_dir=FIX, cache_dir=cache)


def _point(art, name: str) -> float:
    return next(m for m in art.metrics if m.name == name).point


def test_output_guardrail_moves_metrics_with_no_metrics_change(tmp_path: Path) -> None:
    # Baseline C1 (none): the tautological no-defense anchors.
    base_harm = _run_judge_metrics(tmp_path, _cfg(), "none", "harmful_fixture")
    base_over = _run_judge_metrics(tmp_path, _cfg(), "none2", "overrefusal_fixture")
    assert _point(base_harm, "guardrail_fnr") == 1.0
    assert _point(base_over, "guardrail_fpr") == 0.0
    base_asr = _point(base_harm, "asr")

    # Output guardrail: blocked_at flips, so fnr<1, fpr>0, ASR drops -- all off blocked_at alone.
    out_harm = _run_judge_metrics(tmp_path, _out_cfg(), "out", "harmful_fixture")
    out_over = _run_judge_metrics(tmp_path, _out_cfg(), "out2", "overrefusal_fixture")
    assert _point(out_harm, "guardrail_fnr") < 1.0  # the guardrail blocked some harmful items
    assert _point(out_over, "guardrail_fpr") > 0.0  # and wrongly blocked some benign ones
    assert _point(out_harm, "asr") < base_asr  # blocked unsafe responses no longer count as ASR
