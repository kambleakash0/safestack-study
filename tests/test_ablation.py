"""Phase 4a: the core 2x4 ablation pivot. Mock-first (in-memory artifacts, no GPU / no files).

Covers the pivot shape + condition labels, per-suite ASR placement, the split-family duplicate
handling (identical point -> conservative CI envelope; conflicting point -> hard error), the
missing-cell rendering, and the csv/md renderers (the CSV must quote the comma inside each CI)."""

from __future__ import annotations

import pytest

from safestack.eval.ablation import (
    CONDITION_ORDER,
    GUARDRAIL_LABEL,
    POLICY_LABEL,
    ablation_rows,
    before_after_rows,
    render,
    render_before_after,
    write_ablation,
)
from safestack.eval.artifacts import MetricResult, MetricsArtifact

_SUITE_N = {
    "harmful_advbench_v1": 520,
    "harmful_harmbench_v1": 200,
    "dualuse_harmbench_contextual_v1": 100,
    "overrefusal_xstest_v1": 250,
    "helpfulness_alpaca_v1": 200,
}


def _art(cond: str, suite: str, metrics: list[tuple[str, float, float, float]]) -> MetricsArtifact:
    n = _SUITE_N[suite]
    return MetricsArtifact(
        experiment_id=f"exp_{cond}_{suite}",
        condition_id=cond,
        suite=suite,
        split="eval",
        policy_model_id="m",
        n=n,
        metrics=[
            MetricResult(name=name, point=p, ci_low=lo, ci_high=hi, n=n)
            for name, p, lo, hi in metrics
        ],
    )


def _full_condition(cond: str, *, asr: float = 0.1, fpr: float = 0.0) -> list[MetricsArtifact]:
    """One artifact per suite for a condition, so every MATRIX_COLUMNS cell resolves."""
    return [
        _art(cond, "harmful_advbench_v1", [("asr", asr, asr, asr)]),
        _art(cond, "harmful_harmbench_v1", [("asr", asr, asr, asr)]),
        _art(cond, "dualuse_harmbench_contextual_v1", [("asr", asr, asr, asr)]),
        _art(cond, "overrefusal_xstest_v1", [("over_refusal", 0.02, 0.01, 0.03),
                                             ("guardrail_fpr", fpr, fpr, fpr)]),
        _art(cond, "helpfulness_alpaca_v1", [("benign_helpfulness", 4.9, 4.8, 5.0)]),
    ]


def test_pivot_orders_conditions_and_labels():
    arts: list[MetricsArtifact] = []
    for c in CONDITION_ORDER:
        arts += _full_condition(c)
    # Feed them shuffled to prove the pivot sorts by CONDITION_ORDER, not input order.
    rows = ablation_rows(list(reversed(arts)))
    assert [r.condition for r in rows] == CONDITION_ORDER
    for r in rows:
        assert r.policy == POLICY_LABEL[r.condition]
        assert r.guardrail == GUARDRAIL_LABEL[r.condition]
    assert POLICY_LABEL["C4"] == "Starting" and POLICY_LABEL["C5"] == "SFT"
    assert GUARDRAIL_LABEL["C8"] == "input+output"


def test_asr_is_per_suite_not_pooled():
    arts = [
        _art("C1", "harmful_advbench_v1", [("asr", 0.5, 0.4, 0.6)]),
        _art("C1", "harmful_harmbench_v1", [("asr", 0.7, 0.6, 0.8)]),
        _art("C1", "dualuse_harmbench_contextual_v1", [("asr", 0.9, 0.8, 1.0)]),
    ]
    (row,) = ablation_rows(arts)
    assert row.cells["ASR advbench"].point == 0.5
    assert row.cells["ASR harmbench"].point == 0.7
    assert row.cells["ASR dual-use"].point == 0.9
    # benign columns absent for this partial input -> None -> rendered "--"
    assert row.cells["Over-refusal"] is None


def test_split_family_duplicate_merges_ci_envelope():
    # Same (condition, suite, metric) with identical point but different CIs (the _starting vs
    # _dualuse resample-order effect) -> conservative envelope [min lo, max hi].
    arts = [
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.336, 0.276, 0.396)]),
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.336, 0.280, 0.396)]),
    ]
    (row,) = ablation_rows(arts)
    cell = row.cells["Guardrail FPR"]
    assert cell.point == 0.336
    assert cell.ci_low == 0.276 and cell.ci_high == 0.396  # widest of the two


def test_conflicting_point_is_a_hard_error():
    arts = [
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.336, 0.276, 0.396)]),
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.330, 0.276, 0.396)]),
    ]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ablation_rows(arts)


def test_render_md_and_csv_agree_and_csv_quotes_ci():
    rows = ablation_rows(_full_condition("C1", asr=0.5, fpr=0.33))
    md = render(rows, "md")
    csv_text = render(rows, "csv")
    # md pipe table with the header + separator + one data row
    assert md.startswith("| Condition | Policy | Guardrail |")
    assert "| C1 | Starting | none |" in md
    assert "0.500 [0.500, 0.500]" in md
    # csv must quote each CI cell (it contains a comma) so it is one field, not two
    assert '"0.500 [0.500, 0.500]"' in csv_text
    header, first = csv_text.splitlines()[0], csv_text.splitlines()[1]
    assert header.split(",")[0] == "Condition"
    import csv as _csv

    parsed = next(_csv.reader([first]))
    assert parsed[0] == "C1" and parsed[3] == "0.500 [0.500, 0.500]"  # comma stayed in the field


def test_missing_cell_renders_dash():
    (row,) = ablation_rows([_art("C1", "harmful_advbench_v1", [("asr", 0.5, 0.4, 0.6)])])
    md = render([row], "md")
    assert "| C1 | Starting | none | 0.500 [0.400, 0.600] | -- | -- | -- | -- | -- |" in md


def test_unknown_format_raises():
    rows = ablation_rows(_full_condition("C1"))
    with pytest.raises(ValueError, match="unknown format"):
        render(rows, "json")


def test_write_ablation_writes_both_files(tmp_path):
    rows = ablation_rows(_full_condition("C1"))
    paths = write_ablation(rows, tmp_path / "core_ablation", formats=("csv", "md"))
    assert {p.suffix for p in paths} == {".csv", ".md"}
    for p in paths:
        assert p.exists() and p.read_text(encoding="utf-8").strip()


def test_no_guardrail_condition_renders_na_fpr():
    # C1/C5 have no guardrail: the artifact's degenerate 0.0 FPR must be blanked, not shown.
    (c1,) = ablation_rows(_full_condition("C1", fpr=0.5))
    assert c1.cells["Guardrail FPR"] is None  # N/A
    assert c1.cells["Over-refusal"] is not None  # a non-guardrail benign metric still resolves
    assert render([c1], "md").strip().splitlines()[-1].endswith("| -- |")  # last col is N/A
    # a guardrail condition keeps its measured FPR
    (c2,) = ablation_rows(_full_condition("C2", fpr=0.33))
    assert c2.cells["Guardrail FPR"] is not None and c2.cells["Guardrail FPR"].point == 0.33


def test_unknown_condition_id_raises():
    with pytest.raises(ValueError, match="outside the C1-C10 ablation"):
        ablation_rows(_full_condition("C11"))  # C11 not in CONDITION_ORDER (C9/C10 now admitted)


def test_n_mismatch_is_a_hard_error():
    # Same (condition, suite, metric), same point, DIFFERENT n -> conflict (the n-mismatch branch).
    a1 = _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.3, 0.2, 0.4)])
    a2 = MetricsArtifact(
        experiment_id="exp_C2_alt", condition_id="C2", suite="overrefusal_xstest_v1",
        split="eval", policy_model_id="m", n=249,
        metrics=[MetricResult(name="guardrail_fpr", point=0.3, ci_low=0.2, ci_high=0.4, n=249)],
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ablation_rows([a1, a2])


def _all_conditions():
    rows = [_row(c, "Starting", g, asr=0.4, na_fpr=(g == "none")) for c, g in _STARTING.items()]
    rows += [_row(c, "SFT", g, asr=0.05, na_fpr=(g == "none")) for c, g in _SFT.items()]
    return rows


_STARTING = {"C1": "none", "C2": "input", "C3": "output", "C4": "input+output"}
_SFT = {"C5": "none", "C6": "input", "C7": "output", "C8": "input+output"}


def _row(cond, policy, guardrail, *, asr, na_fpr):
    from safestack.eval.ablation import MATRIX_COLUMNS, AblationRow, Cell

    cells = {}
    for _, _, header in MATRIX_COLUMNS:
        if header.startswith("ASR"):
            cells[header] = Cell(asr, asr, asr, 100)
        elif header == "Over-refusal":
            cells[header] = Cell(0.03, 0.02, 0.04, 250)
        elif header == "Helpfulness":
            cells[header] = Cell(4.9, 4.8, 5.0, 200)
        elif header == "Guardrail FPR":
            cells[header] = None if na_fpr else Cell(0.33, 0.27, 0.39, 250)
    return AblationRow(cond, policy, guardrail, cells)


def test_before_after_pairs_deltas_and_na():
    ba = before_after_rows(_all_conditions())
    # 4 guardrail configs x 6 metric columns
    assert len(ba) == 24
    assert {r.guardrail for r in ba} == {"none", "input", "output", "input+output"}
    none_asr = next(r for r in ba if r.guardrail == "none" and r.metric == "ASR advbench")
    assert none_asr.starting.point == 0.4 and none_asr.sft.point == 0.05
    assert abs(none_asr.delta - (0.05 - 0.4)) < 1e-9  # SFT reduces ASR -> negative delta
    # guardrail FPR is N/A on the no-guardrail pair (C1 vs C5) -> None delta, rendered NA
    none_fpr = next(r for r in ba if r.guardrail == "none" and r.metric == "Guardrail FPR")
    assert none_fpr.starting is None and none_fpr.sft is None and none_fpr.delta is None
    md = render_before_after(ba, "md")
    assert "delta (SFT-Starting)" in md and "-0.350" in md  # the none/ASR delta, signed


def test_before_after_half_present_pair_fails_loud():
    # C2 supplied but its SFT pair C6 omitted -> the 'input' pair is incomplete -> hard error.
    rows = [_row("C2", "Starting", "input", asr=0.4, na_fpr=False)]
    with pytest.raises(ValueError, match="incomplete"):
        before_after_rows(rows)


def test_before_after_one_sided_na_delta_is_none_but_value_shown():
    # Starting has a guardrail FPR, SFT is N/A on it -> delta None, but the value still renders.
    rows = [
        _row("C1", "Starting", "none", asr=0.4, na_fpr=False),  # FPR present (synthetic)
        _row("C5", "SFT", "none", asr=0.05, na_fpr=True),  # FPR N/A
    ]
    ba = before_after_rows(rows)
    fpr = next(r for r in ba if r.guardrail == "none" and r.metric == "Guardrail FPR")
    assert fpr.starting is not None and fpr.sft is None and fpr.delta is None
    assert "0.330 [0.270, 0.390] | -- | --" in render_before_after(ba, "md")


def test_envelope_merges_the_high_side_too():
    arts = [
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.3, 0.20, 0.40)]),
        _art("C2", "overrefusal_xstest_v1", [("guardrail_fpr", 0.3, 0.20, 0.45)]),
    ]
    (row,) = ablation_rows(arts)
    cell = row.cells["Guardrail FPR"]
    assert cell.ci_low == 0.20 and cell.ci_high == 0.45

def test_stressed_rungs_admitted_as_a_third_policy_family():
    # The Phase-5 C9/C10 rungs resolve as the "Stressed" policy family (not silently dropped, not
    # mislabeled SFT), with C9 bare (FPR N/A) and C10 input+output.
    (c9,) = ablation_rows(_full_condition("C9", asr=0.94))
    assert c9.policy == "Stressed" and c9.guardrail == "none"
    assert c9.cells["Guardrail FPR"] is None  # no screen -> N/A
    (c10,) = ablation_rows(_full_condition("C10", asr=0.08, fpr=0.324))
    assert c10.policy == "Stressed" and c10.guardrail == "input+output"
    assert POLICY_LABEL["C9"] == "Stressed" and GUARDRAIL_LABEL["C10"] == "input+output"
