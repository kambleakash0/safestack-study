"""Reproducibility guard: the committed TEXT artifacts (dashboard + tables) must regenerate
byte-for-byte from the committed metrics. A source change that would silently drift a committed
artifact (e.g. a CSS constant edited in the file but not the generator) then fails CI here, not only
at review. Needs no [viz] extra -- the HTML/tables are matplotlib-free."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from safestack.eval.ablation import (
    ablation_rows,
    before_after_rows,
    render,
    render_before_after,
)
from safestack.eval.figures import build_dashboard_html
from safestack.eval.report import load_artifacts
from safestack.eval.segments import (
    failure_taxonomy_rows,
    render_failure_taxonomy,
    segment_asr_grid,
)

# C1-C10 only. `c[0-9]_` matches c1..c9 and `c10_` adds c10; this deliberately EXCLUDES the
# Phase-6 c19/c20/c21 (and dev_selection_* / dose) artifacts, which are not part of the Phase-4
# core-ablation / dashboard reproduction. (A bare `c[0-9]*_` would wrongly capture c19/c20/c21.)
_METRICS = sorted(
    glob.glob("reports/metrics/c[0-9]_*.json") + glob.glob("reports/metrics/c10_*.json")
)


def _arts():
    if len(_METRICS) < 8:
        pytest.skip("committed C1-C10 metrics not present")
    return load_artifacts(_METRICS)


def _committed(rel: str) -> str:
    return Path(rel).read_text(encoding="utf-8")


def test_dashboard_regenerates_byte_for_byte():
    arts = _arts()
    html = build_dashboard_html(ablation_rows(arts), segment_asr_grid(arts))
    assert html == _committed("reports/dashboard.html")


def test_core_ablation_regenerates_byte_for_byte():
    rows = ablation_rows(_arts())
    assert render(rows, "md") == _committed("reports/tables/core_ablation.md")
    assert render(rows, "csv") == _committed("reports/tables/core_ablation.csv")


def test_sft_before_after_regenerates_byte_for_byte():
    ba = before_after_rows(ablation_rows(_arts()))
    assert render_before_after(ba, "md") == _committed("reports/tables/sft_before_after.md")
    assert render_before_after(ba, "csv") == _committed("reports/tables/sft_before_after.csv")


def test_failure_taxonomy_regenerates_byte_for_byte():
    tax = failure_taxonomy_rows(segment_asr_grid(_arts()))
    assert render_failure_taxonomy(tax, "md") == _committed("reports/tables/failure_taxonomy.md")
    assert render_failure_taxonomy(tax, "csv") == _committed("reports/tables/failure_taxonomy.csv")
