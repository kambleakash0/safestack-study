"""Phase 4b: the figure layer. Mock-first -- the series extractors + the inline-SVG dashboard need
no matplotlib (pure data/string); only the static-figure render tests importorskip matplotlib so CI
(no [viz] extra) stays green while a [viz] env exercises them."""

from __future__ import annotations

import re
import xml.dom.minidom as minidom

import pytest

from safestack.eval.ablation import MATRIX_COLUMNS, AblationRow, Cell
from safestack.eval.figures import (
    ASR_SUITES,
    ParetoPoint,
    _svg_scatter,
    asr_series,
    build_dashboard_html,
    overrefusal_series,
    pareto_frontier,
    pareto_points,
    write_figures,
)

_STARTING = {"C1": "none", "C2": "input", "C3": "output", "C4": "input+output"}
_SFT = {"C5": "none", "C6": "input", "C7": "output", "C8": "input+output"}


def _row(cond: str, policy: str, guardrail: str, *, asr: float, na_fpr: bool) -> AblationRow:
    cells: dict[str, Cell | None] = {}
    for _, _, header in MATRIX_COLUMNS:
        if header.startswith("ASR"):
            cells[header] = Cell(asr, max(0.0, asr - 0.02), asr + 0.02, 100)
        elif header == "Over-refusal":
            cells[header] = Cell(0.03, 0.02, 0.05, 250)
        elif header == "Helpfulness":
            cells[header] = Cell(4.9, 4.8, 5.0, 200)
        elif header == "Guardrail FPR":
            cells[header] = None if na_fpr else Cell(0.33, 0.27, 0.39, 250)
    return AblationRow(cond, policy, guardrail, cells)


def _rows() -> list[AblationRow]:
    rows = [_row(c, "Starting", g, asr=0.4, na_fpr=(g == "none")) for c, g in _STARTING.items()]
    rows += [_row(c, "SFT", g, asr=0.05, na_fpr=(g == "none")) for c, g in _SFT.items()]
    return rows


def test_asr_series_is_three_suites_by_eight_conditions():
    series = asr_series(_rows())
    assert list(series) == [h for h, _, _ in ASR_SUITES]
    for bars in series.values():
        assert len(bars) == 8  # one per condition
        assert all(b is not None for b in bars)  # ASR is never N/A
    assert series["ASR advbench"][0].point == 0.4  # C1


def test_overrefusal_series_and_na_fpr_is_none():
    orr = overrefusal_series(_rows())
    assert len(orr) == 8 and all(b is not None for b in orr)
    # the guardrail-FPR cell is N/A on the no-guardrail rows -> _bar returns None there, but
    # over-refusal itself is always present.
    (c1,) = [r for r in _rows() if r.condition == "C1"]
    from safestack.eval.figures import _bar

    assert _bar(c1, "Guardrail FPR") is None
    assert _bar(c1, "Over-refusal") is not None


def test_dashboard_is_self_contained_themed_and_parseable():
    h = build_dashboard_html(_rows())
    # self-contained: no external network references
    assert "http://" not in h and "https://" not in h and "src=" not in h
    # three charts present + a legend + the table (with N/A cells for C1/C5 FPR)
    assert h.count("<svg") == 3
    assert "class='legend'" in h or 'class="legend"' in h
    assert h.count(">--<") == 2  # C1/C5 guardrail-FPR N/A cells, same token as the ablation table
    # theme-aware: a dark block under both the media query and the data-theme toggle
    assert "prefers-color-scheme: dark" in h and "[data-theme=dark]" in h
    # every SVG is well-formed XML
    svgs = re.findall(r"<svg.*?</svg>", h, re.S)
    assert len(svgs) == 3  # ASR, over-refusal, Pareto
    for svg in svgs:
        minidom.parseString(svg)


def test_dashboard_svg_bars_stay_within_the_viewbox():
    h = build_dashboard_html(_rows())
    svg = re.findall(r"<svg[^>]*viewBox=\"0 0 (\d+) (\d+)\".*?</svg>", h, re.S)
    # re-extract with dims
    for m in re.finditer(r'<svg[^>]*viewBox="0 0 (\d+) (\d+)"(.*?)</svg>', h, re.S):
        w, ht, body = int(m.group(1)), int(m.group(2)), m.group(3)
        for rect in re.finditer(
            r'<rect class="bar[^"]*" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"',
            body,
        ):
            x, y, bw, bh = (float(g) for g in rect.groups())
            assert 0 <= x and x + bw <= w + 0.5
            assert 0 <= y and y + bh <= ht + 0.5
            assert bw >= 0 and bh >= 0
        # CI whiskers must also stay inside the viewBox (a hi > y_max would render above it).
        for line in re.finditer(
            r'<line class="ci" x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', body
        ):
            x1, y1, x2, y2 = (float(g) for g in line.groups())
            assert 0 <= x1 <= w and 0 <= x2 <= w
            assert 0 <= y1 <= ht + 0.5 and 0 <= y2 <= ht + 0.5
    assert svg  # at least one sized viewBox matched


def test_asr_suites_carry_fixed_hue_slots():
    # Categorical invariant: advbench=slot0(blue), harmbench=slot1(orange), dual-use=slot2(aqua),
    # assigned by identity and never cycled.
    from safestack.eval.figures import PALETTE

    assert [(short, slot) for _, short, slot in ASR_SUITES] == [
        ("advbench", 0), ("harmbench", 1), ("dual-use", 2)
    ]
    assert PALETTE[0][0] == "#2a78d6" and PALETTE[1][0] == "#eb6834" and PALETTE[2][0] == "#1baf7a"
    # and the rendered ASR chart carries 8 bars of each slot class (3 suites x 8 conditions).
    asr_svg = re.findall(r"<svg.*?</svg>", build_dashboard_html(_rows()), re.S)[0]
    for slot in (0, 1, 2):
        assert asr_svg.count(f'class="bar s{slot}"') == 8


def test_overrefusal_chart_has_direct_value_labels_asr_does_not():
    h = build_dashboard_html(_rows())
    asr_svg, orr_svg, _pareto = re.findall(r"<svg.*?</svg>", h, re.S)
    assert asr_svg.count('class="val"') == 0  # dataviz: no number on every one of 24 bars
    assert orr_svg.count('class="val"') == 8  # legible on the 8-bar single series


def test_condition_labels_are_html_escaped_in_svg():
    # A crafted condition must not inject markup into the self-contained dashboard.
    evil = "C1</text><script>alert(1)</script>"
    row = _row(evil, "Starting", "none", asr=0.4, na_fpr=True)
    h = build_dashboard_html([row])
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h


def test_pareto_points_cost_and_safety():
    pts = {p.condition: p for p in pareto_points(_rows())}
    # _rows() sets Starting ASR=0.4, SFT ASR=0.05 across all 3 suites -> safety = 1 - mean ASR.
    assert abs(pts["C1"].safety - 0.6) < 1e-9 and abs(pts["C5"].safety - 0.95) < 1e-9
    # no-guardrail conditions have zero benign-block cost (N/A FPR -> 0); guardrailed carry the FPR.
    assert pts["C1"].cost == 0.0 and pts["C5"].cost == 0.0
    assert pts["C2"].cost == 0.33


def test_pareto_frontier_keeps_only_non_dominated():
    pts = [
        ParetoPoint("A", "x", cost=0.0, safety=0.90, mean_asr=0.10),  # cheap+good -> frontier
        ParetoPoint("B", "x", cost=0.3, safety=0.99, mean_asr=0.01),  # dear+best -> frontier
        ParetoPoint("C", "x", cost=0.3, safety=0.80, mean_asr=0.20),  # dear+worse -> dominated by B
        ParetoPoint("D", "x", cost=0.1, safety=0.85, mean_asr=0.15),  # dominated by A
    ]
    assert [p.condition for p in pareto_frontier(pts)] == ["A", "B"]  # sorted by cost


def test_pareto_frontier_keeps_exact_ties():
    # Two points with identical (cost, safety) -- neither strictly dominates -> both survive.
    pts = [
        ParetoPoint("A", "x", cost=0.1, safety=0.9, mean_asr=0.1),
        ParetoPoint("B", "y", cost=0.1, safety=0.9, mean_asr=0.1),
    ]
    assert {p.condition for p in pareto_frontier(pts)} == {"A", "B"}


def test_svg_scatter_fails_loud_on_out_of_window_point():
    bad = [ParetoPoint("Z", "SFT", cost=0.5, safety=0.9, mean_asr=0.1)]  # cost > x_max 0.4
    with pytest.raises(ValueError, match="outside"):
        _svg_scatter(bad, [])


def test_svg_scatter_draws_dots_and_frontier_polyline():
    pts = [
        ParetoPoint("A", "SFT", cost=0.0, safety=0.90, mean_asr=0.10),
        ParetoPoint("B", "Starting", cost=0.3, safety=0.99, mean_asr=0.01),
    ]
    svg = _svg_scatter(pts, pareto_frontier(pts))
    assert svg.count('<circle class="dot') == 2
    assert 'class="dot s0"' in svg and 'class="dot s1"' in svg  # SFT->s0, Starting->s1
    assert 'polyline class="front"' in svg  # 2-point frontier -> a line


def test_pareto_panel_in_dashboard_has_a_dot_per_condition():
    h = build_dashboard_html(_rows())
    pareto_svg = re.findall(r"<svg.*?</svg>", h, re.S)[2]
    assert pareto_svg.count('<circle class="dot') == 8  # one per condition


def test_matplotlib_figures_written(tmp_path):
    pytest.importorskip("matplotlib")
    paths = write_figures(_rows(), figures_dir=tmp_path / "figures", dashboard=None)
    names = {p.name for p in paths}
    assert names == {
        "asr_by_condition.svg", "asr_by_condition.png",
        "over_refusal_by_condition.svg", "over_refusal_by_condition.png",
        "safety_cost_pareto.svg", "safety_cost_pareto.png",
    }
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_write_figures_end_to_end_includes_dashboard(tmp_path):
    pytest.importorskip("matplotlib")
    paths = write_figures(
        _rows(), figures_dir=tmp_path / "figures", dashboard=tmp_path / "dashboard.html"
    )
    dash = [p for p in paths if p.name == "dashboard.html"]
    assert dash and dash[0].read_text(encoding="utf-8").startswith("<!doctype html>")
