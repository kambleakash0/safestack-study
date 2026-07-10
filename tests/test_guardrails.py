"""Phase 2 / C2-C4: guardrail subpackage + the composite input/output seam (mock-first, base
install).

Covers the Guardrail ABC + composite mock (input + output stages, self-gated on placement) + the
build_guardrail factory for all four placements (incl. the ADR-0004 rule-4 guard and the ADR-0009
one-model input_output rule), the run_suite output pass (blocked_at / final_response / timings land
on the trace), and a regression proving the metrics layer is untouched: with a real block pattern,
guardrail_fpr>0, guardrail_fnr<1, and ASR drops vs the C1 baseline -- all read off blocked_at.
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
from safestack.guardrails.mock import MockGuardrail

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
    g = MockGuardrail()
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


def test_mock_input_guardrail_deterministic_and_independent() -> None:
    g = MockGuardrail(placement="input")
    prompts = [f"prompt number {i}" for i in range(12)]
    decisions = [g.check_input(p) for p in prompts]
    # Deterministic: a second pass gives identical verdicts.
    assert [d.blocked_at for d in decisions] == [g.check_input(p).blocked_at for p in prompts]
    blocked = [d for d in decisions if d.blocked_at is not None]
    passed = [d for d in decisions if d.blocked_at is None]
    assert blocked and passed  # the input mock spans both outcomes over a normal set of prompts
    for d in blocked:
        assert d.blocked_at == "input"
        assert d.final_response == SAFE_REFUSAL
    for d in passed:
        assert d.final_response == ""  # passing input check returns "" (sentinel, never surfaced)
    # Independence: the input block-set differs from the output block-set over the same items, so
    # the C4 short-circuit and the input-metrics regression are genuinely selective, not correlated.
    og = MockGuardrail(placement="input_output")
    in_blocks = [og.check_input(p).blocked_at is not None for p in prompts]
    out_blocks = [og.check_output(p, f"resp {p}").blocked_at is not None for p in prompts]
    assert in_blocks != out_blocks


def test_mock_check_input_gated_when_output_only() -> None:
    # An output-placement guardrail's input stage is a pass-through (mirrors C3: input never runs).
    d = MockGuardrail(placement="output").check_input("anything")
    assert d.blocked_at is None and d.final_response == "" and d.guardrail_ms is None


def test_mock_check_output_gated_when_input_only() -> None:
    # An input-placement guardrail's output stage is a pass-through (mirrors C2: output never runs).
    d = MockGuardrail(placement="input").check_output("p", "a response")
    assert d.blocked_at is None and d.final_response == "a response" and d.guardrail_ms is None


# ---- factory ----


def test_build_none_returns_null() -> None:
    assert isinstance(build_guardrail(_cfg()), NullGuardrail)


def test_build_output_mock() -> None:
    g = build_guardrail(_out_cfg())
    assert isinstance(g, MockGuardrail) and g.placement == "output"


def test_build_output_requires_a_model() -> None:
    with pytest.raises(ValueError, match="requires cfg.output_guardrail"):
        build_guardrail(_cfg(guardrail_config="output"))


def _in_cfg(**over: object) -> EvalExperimentConfig:
    defaults: dict = dict(
        condition_id="C2",
        guardrail_config="input",
        input_guardrail=ModelSpec(model_id="mock_guard", backend="mock"),
    )
    defaults.update(over)
    return _cfg(**defaults)


def _io_cfg(**over: object) -> EvalExperimentConfig:
    defaults: dict = dict(
        condition_id="C4",
        guardrail_config="input_output",
        input_guardrail=ModelSpec(model_id="mock_guard", backend="mock"),
        output_guardrail=ModelSpec(model_id="mock_guard", backend="mock"),
    )
    defaults.update(over)
    return _cfg(**defaults)


def test_build_input_mock() -> None:
    g = build_guardrail(_in_cfg())
    assert isinstance(g, MockGuardrail) and g.placement == "input"


def test_build_input_output_mock() -> None:
    # C4: ONE composite guardrail serves both stages over a single model (ADR-0009 decision 2).
    g = build_guardrail(_io_cfg())
    assert isinstance(g, MockGuardrail) and g.placement == "input_output"


def test_build_input_requires_a_model() -> None:
    with pytest.raises(ValueError, match="requires cfg.input_guardrail"):
        build_guardrail(_cfg(guardrail_config="input"))


def test_build_input_output_requires_both_models() -> None:
    with pytest.raises(ValueError, match="requires both"):
        build_guardrail(
            _cfg(
                guardrail_config="input_output",
                input_guardrail=ModelSpec(model_id="mock_guard", backend="mock"),
            )
        )


def test_build_input_output_requires_same_model() -> None:
    # Two different models for the two stages -> C4's single-composite invariant is violated.
    with pytest.raises(ValueError, match="SAME model"):
        build_guardrail(_io_cfg(output_guardrail=ModelSpec(model_id="other_guard", backend="mock")))


def test_rule4_input_guardrail_may_not_be_the_safety_judge() -> None:
    # Input guardrail == the safety judge -> circular, rejected (ADR-0004 rule 4).
    with pytest.raises(ValueError, match="rule 4"):
        build_guardrail(
            _in_cfg(
                safety_judge=ModelSpec(model_id="shared_guard", backend="mock"),
                input_guardrail=ModelSpec(model_id="shared_guard", backend="mock"),
            )
        )


def test_build_output_real_granite_is_lazy() -> None:
    # The real backend now returns a Granite guardrail; the heavy model must NOT load at
    # construction (ADR-0003) -- it materialises only on the first check_output.
    from safestack.guardrails.granite import GraniteGuardrail

    cfg = _out_cfg(
        output_guardrail=ModelSpec(
            model_id="granite_guardian_2b", backend="hf_local", checkpoint="x", device="cuda"
        )
    )
    g = build_guardrail(cfg)
    assert isinstance(g, GraniteGuardrail)
    assert g._gateway is None  # lazy: no weight load at build time


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
    from safestack.eval.judges.mock import MockSafetyJudge

    # Baseline C1 (none): the tautological no-defense anchors.
    base_harm = _run_judge_metrics(tmp_path, _cfg(), "none", "harmful_fixture")
    base_over = _run_judge_metrics(tmp_path, _cfg(), "none2", "overrefusal_fixture")
    assert _point(base_harm, "guardrail_fnr") == 1.0
    assert _point(base_over, "guardrail_fpr") == 0.0

    # Output guardrail: run + judge, then recompute the metrics BY HAND from the two independent
    # mock predicates; the artifact must match exactly (metrics.py reads blocked_at correctly).
    cache = tmp_path / "out"
    run_dir = run_suite(_out_cfg(), runs_dir=tmp_path / "out_runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=_out_cfg(), data_dir=FIX, cache_dir=cache)
    out_harm = suite_metrics(run_dir, _out_cfg(), "harmful_fixture", data_dir=FIX, cache_dir=cache)
    out_over = suite_metrics(
        run_dir, _out_cfg(), "overrefusal_fixture", data_dir=FIX, cache_dir=cache
    )

    judge, guard = MockSafetyJudge(fingerprint={}, judge_prompt_version="v1"), MockGuardrail()

    def unsafe(t: dict) -> bool:
        return judge.score("", t["output"]).label == "unsafe"

    def blocked(t: dict) -> bool:
        return guard.check_output(t["prompt"], t["output"]).blocked_at is not None

    harm = [t for t in _traces(run_dir) if t["suite"] == "harmful_fixture"]
    over = [t for t in _traces(run_dir) if t["suite"] == "overrefusal_fixture"]
    exp_asr = sum(unsafe(t) and not blocked(t) for t in harm) / len(harm)
    exp_fnr = sum(not blocked(t) for t in harm) / len(harm)
    exp_fpr = sum(blocked(t) for t in over) / len(over)
    base_asr = sum(unsafe(t) for t in harm) / len(harm)

    assert _point(out_harm, "asr") == round(exp_asr, 6)
    assert _point(out_harm, "guardrail_fnr") == round(exp_fnr, 6)
    assert _point(out_over, "guardrail_fpr") == round(exp_fpr, 6)
    # Non-degenerate: the guardrail blocks a PARTIAL subset of harmful (0 < fnr < 1) and some benign
    # (fpr > 0), on a predicate independent of the judge. Blocking the unsafe item drops ASR from a
    # real baseline to 0, and the exact match above proves metrics.py excluded exactly that item.
    assert 0.0 < exp_fnr < 1.0
    assert exp_fpr > 0.0
    assert base_asr > 0.0
    assert exp_asr < base_asr


def test_guardrails_import_first_has_no_circular_import() -> None:
    # The Colab pre-flight imports safestack.guardrails before safestack.eval; that order must not
    # trigger the eval.config <-> generate <-> guardrails circular import. A fresh interpreter is
    # needed because pytest has already imported these modules in a cycle-safe order.
    import subprocess
    import sys

    code = (
        "from safestack.guardrails.base import SAFE_REFUSAL; "
        "from safestack.guardrails.granite import GraniteGuardrail; "
        "from safestack.guardrails import build_guardrail"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
