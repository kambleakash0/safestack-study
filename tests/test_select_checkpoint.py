"""Rule-9 checkpoint selection (ADR-0015 amendment 1, FU5b): the tripwire, the ASR+over-refusal
objective, the within-CI-noise tiebreak, the degenerate flags, artifact extraction, and the
byte-stable committed selection. Pure/mock -- no model load, runs under `-m 'not hf'`."""

from __future__ import annotations

import json

import pytest

from safestack.eval.artifacts import MetricResult, MetricsArtifact
from safestack.train.select import (
    CheckpointDevMetrics,
    MetricPoint,
    checkpoint_dev_metrics,
    select_checkpoint,
    write_selection,
)


def _ckpt(name, step, *, asr, orr, ans, asr_ci=None, orr_ci=None):
    asr_ci = asr_ci or (asr, asr)
    orr_ci = orr_ci or (orr, orr)
    return CheckpointDevMetrics(
        checkpoint=name,
        step=step,
        asr=MetricPoint(point=asr, ci_low=asr_ci[0], ci_high=asr_ci[1]),
        over_refusal=MetricPoint(point=orr, ci_low=orr_ci[0], ci_high=orr_ci[1]),
        helpfulness_answer_rate=ans,
    )


# base: high ASR (so an SFT drop is possible), high answer-rate (the tripwire reference).
BASE = _ckpt("base", 0, asr=0.50, asr_ci=(0.40, 0.60), orr=0.05, ans=0.90)


def test_empty_candidates_raises():
    with pytest.raises(ValueError, match="at least one candidate"):
        select_checkpoint([], BASE)


def test_single_clean_selection_no_flags():
    c = _ckpt("c1", 100, asr=0.20, asr_ci=(0.10, 0.30), orr=0.08, ans=0.88)
    sel = select_checkpoint([c], BASE)
    assert sel.selected_checkpoint == "c1" and sel.selected_step == 100
    assert sel.flags == [] and sel.dev_degenerate is False
    assert sel.n_candidates == 1 and sel.n_survivors == 1 and sel.n_contenders == 1
    assert sel.objective == pytest.approx(0.28)
    assert sel.tripwire_margin == 0.10


def test_tripwire_rejects_mode_collapse_even_at_lowest_objective():
    # A mode-collapses: ASR 0 (refuses harmful) AND lowest objective, but its answer-rate
    # craters to 0.40 (< base 0.90 - 0.10 = 0.80) -> REJECTED; B survives, proving the
    # tripwire overrides the objective (the whole point of the mode-collapse gate).
    a = _ckpt("collapse", 100, asr=0.0, orr=0.05, ans=0.40)  # objective 0.05, lowest
    b = _ckpt("healthy", 200, asr=0.25, orr=0.10, ans=0.87)  # objective 0.35
    sel = select_checkpoint([a, b], BASE)
    assert sel.selected_checkpoint == "healthy"
    assert sel.n_survivors == 1 and "no_tripwire_survivor" not in sel.flags
    assert sel.dev_degenerate is False  # B beats base on ASR (0.25 < 0.50)


def test_all_fail_tripwire_falls_back_and_flags():
    a = _ckpt("c1", 100, asr=0.10, orr=0.30, ans=0.50)  # objective 0.40
    b = _ckpt("c2", 200, asr=0.05, orr=0.20, ans=0.55)  # objective 0.25, lower
    sel = select_checkpoint([a, b], BASE)
    assert "no_tripwire_survivor" in sel.flags and sel.dev_degenerate is True
    assert sel.n_survivors == 0
    assert sel.selected_checkpoint == "c2"  # best objective over ALL candidates in the fallback


def test_no_asr_improvement_flag_when_asr_not_below_base():
    # Survives the tripwire but ASR point 0.55 is not below base 0.50 -> directional pre-warning.
    c = _ckpt("c1", 100, asr=0.55, asr_ci=(0.45, 0.65), orr=0.05, ans=0.88)
    sel = select_checkpoint([c], BASE)
    assert sel.flags == ["no_asr_improvement"] and sel.dev_degenerate is True


def test_within_ci_noise_tiebreak_prefers_lower_asr():
    # Two survivors whose objective intervals OVERLAP -> tied; the higher objective-point checkpoint
    # wins because it has the lower ASR (ADR-0004 rule 6: do not select on noise).
    a = _ckpt("a", 100, asr=0.20, asr_ci=(0.10, 0.30), orr=0.10, orr_ci=(0.05, 0.15), ans=0.88)
    #   objective point 0.30, interval [0.15, 0.45]
    b = _ckpt("b", 200, asr=0.10, asr_ci=(0.02, 0.20), orr=0.25, orr_ci=(0.15, 0.35), ans=0.88)
    #   objective point 0.35 (worse), interval [0.17, 0.55] -> overlaps a
    sel = select_checkpoint([a, b], BASE)
    assert sel.n_contenders == 2  # both within CI noise of the best
    assert sel.selected_checkpoint == "b"  # lower ASR wins the tiebreak despite worse objective


def test_ci_separable_best_wins_without_tiebreak():
    # A's objective is CI-separably below B's (no overlap): A wins outright, and B's lower ASR
    # must NOT pull it in as a contender.
    a = _ckpt("a", 100, asr=0.08, asr_ci=(0.04, 0.12), orr=0.05, orr_ci=(0.03, 0.08), ans=0.88)
    #   interval [0.07, 0.20]
    b = _ckpt("b", 200, asr=0.02, asr_ci=(0.00, 0.05), orr=0.45, orr_ci=(0.40, 0.55), ans=0.88)
    #   interval [0.40, 0.60] -> no overlap with a
    sel = select_checkpoint([a, b], BASE)
    assert sel.selected_checkpoint == "a" and sel.n_contenders == 1


def test_tiebreak_earliest_step_on_asr_tie():
    a = _ckpt("late", 300, asr=0.20, asr_ci=(0.10, 0.30), orr=0.10, orr_ci=(0.05, 0.15), ans=0.88)
    b = _ckpt("early", 100, asr=0.20, asr_ci=(0.10, 0.30), orr=0.12, orr_ci=(0.06, 0.18), ans=0.88)
    sel = select_checkpoint([a, b], BASE)
    assert sel.selected_checkpoint == "early"  # equal ASR -> earliest/simplest checkpoint


# ---- artifact extraction ----


def _art(suite, split, metrics):
    return MetricsArtifact(
        experiment_id="e",
        condition_id="c5",
        suite=suite,
        split=split,
        policy_model_id="mistral",
        n=100,
        metrics=metrics,
    )


def _mr(name, point, lo, hi, extra=None):
    return MetricResult(name=name, point=point, ci_low=lo, ci_high=hi, n=100, extra=extra or {})


def test_checkpoint_dev_metrics_extracts_from_artifacts():
    harmful = _art("dev_h", "dev_harmful", [_mr("asr", 0.18, 0.10, 0.28)])
    orr = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.12, 0.06, 0.20)])
    helped = _art(
        "dev_help", "dev_helpfulness",
        [_mr("benign_helpfulness", 4.2, 3.9, 4.5, extra={"answer_rate": 0.86, "scale": "1-5"})],
    )
    m = checkpoint_dev_metrics("c1", 125, harmful=harmful, overrefusal=orr, helpfulness=helped)
    assert m.asr.point == 0.18 and m.asr.ci_high == 0.28
    assert m.over_refusal.point == 0.12
    assert m.helpfulness_answer_rate == 0.86  # the answer-rate, NOT the 1-5 score


def test_checkpoint_dev_metrics_rejects_miswired_split():
    # An over-refusal artifact handed to the harmful slot must fail loudly, never mis-read.
    wrong = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.12, 0.06, 0.20)])
    orr = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.12, 0.06, 0.20)])
    helped = _art(
        "dev_help", "dev_helpfulness",
        [_mr("benign_helpfulness", 4.2, 3.9, 4.5, extra={"answer_rate": 0.86})],
    )
    with pytest.raises(ValueError, match="harmful/ASR"):
        checkpoint_dev_metrics("c1", 1, harmful=wrong, overrefusal=orr, helpfulness=helped)


def test_checkpoint_dev_metrics_missing_answer_rate_raises():
    harmful = _art("dev_h", "dev_harmful", [_mr("asr", 0.18, 0.10, 0.28)])
    orr = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.12, 0.06, 0.20)])
    helped = _art("dev_help", "dev_helpfulness", [_mr("benign_helpfulness", 4.2, 3.9, 4.5)])
    with pytest.raises(ValueError, match="answer_rate"):
        checkpoint_dev_metrics("c1", 1, harmful=harmful, overrefusal=orr, helpfulness=helped)


def test_missing_metric_raises_loudly():
    empty = _art("dev_h", "dev_harmful", [])  # a dev suite whose metrics pass emitted nothing
    orr = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.1, 0.0, 0.2)])
    helped = _art(
        "dev_help", "dev_helpfulness",
        [_mr("benign_helpfulness", 4.0, 3.5, 4.5, extra={"answer_rate": 0.8})],
    )
    with pytest.raises(ValueError, match="metric 'asr' not found"):
        checkpoint_dev_metrics("c1", 1, harmful=empty, overrefusal=orr, helpfulness=helped)


# ---- committed artifact ----


def test_write_selection_byte_stable_and_aggregate_only(tmp_path):
    c = _ckpt("c1", 100, asr=0.20, asr_ci=(0.10, 0.30), orr=0.08, ans=0.88)
    sel = select_checkpoint([c], BASE)
    path = write_selection(sel, tmp_path / "reports" / "selection" / "sft.json")
    assert path.exists()
    reloaded = path.read_text(encoding="utf-8")
    assert reloaded == sel.to_json()  # byte-stable
    assert select_checkpoint([c], BASE).to_json() == sel.to_json()  # deterministic
    data = json.loads(reloaded)
    assert data["selected_checkpoint"] == "c1" and "prompt" not in reloaded
    assert "amendment 1" in data["rationale"] and "no_tripwire_survivor" not in reloaded

def test_tripwire_boundary_exactly_at_margin_survives():
    # A checkpoint EXACTLY tripwire_margin below base MUST survive (reject iff strictly below). With
    # base 0.40, 0.40 - 0.10 == 0.30000000000000004 in IEEE-754, so a naive threshold would wrongly
    # reject a candidate at 0.30 -- the rounded threshold keeps it a survivor.
    base = _ckpt("base", 0, asr=0.50, asr_ci=(0.40, 0.60), orr=0.05, ans=0.40)
    c = _ckpt("c1", 100, asr=0.20, asr_ci=(0.10, 0.30), orr=0.08, ans=0.30)  # exactly 10pp below
    sel = select_checkpoint([c], base)
    assert sel.n_survivors == 1 and "no_tripwire_survivor" not in sel.flags


def test_no_asr_improvement_fires_at_asr_equal_base():
    # The directional pre-warning uses >=, so ASR exactly equal to base (no reduction) also flags.
    c = _ckpt("c1", 100, asr=0.50, asr_ci=(0.40, 0.60), orr=0.05, ans=0.88)  # asr == base 0.50
    sel = select_checkpoint([c], BASE)
    assert sel.flags == ["no_asr_improvement"] and sel.dev_degenerate is True


def test_selection_rejects_locked_test_split_input():
    # Feeding a locked-test artifact into selection is test-set leakage (ADR-0004 rule 3) and must
    # fail loudly -- the split guards accept dev_* only.
    from safestack.train.select import checkpoint_dev_metrics

    harmful = _art("eval_h", "eval_harmful", [_mr("asr", 0.18, 0.10, 0.28)])  # LOCKED test split
    orr = _art("dev_o", "dev_overrefusal", [_mr("over_refusal", 0.12, 0.06, 0.20)])
    helped = _art(
        "dev_help", "dev_helpfulness",
        [_mr("benign_helpfulness", 4.2, 3.9, 4.5, extra={"answer_rate": 0.86})],
    )
    with pytest.raises(ValueError, match="harmful/ASR"):
        checkpoint_dev_metrics("c1", 1, harmful=harmful, overrefusal=orr, helpfulness=helped)
