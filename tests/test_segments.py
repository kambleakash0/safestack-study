"""Phase 4d/4f: the segment reader + failure taxonomy. Mock-first (in-memory artifacts, no GPU)."""

from __future__ import annotations

import pytest

from safestack.eval.artifacts import MetricResult, MetricsArtifact, SegmentResult
from safestack.eval.segments import (
    failure_taxonomy_rows,
    render_failure_taxonomy,
    segment_asr_grid,
)


def _seg_art(cond: str, suite: str, cats: dict[str, float]) -> MetricsArtifact:
    return MetricsArtifact(
        experiment_id=f"e_{cond}_{suite}",
        condition_id=cond,
        suite=suite,
        split="eval",
        policy_model_id="m",
        n=100,
        metrics=[MetricResult(name="asr", point=0.1, ci_low=0.1, ci_high=0.1, n=100)],
        segments=[
            SegmentResult(metric="asr", segment=c, point=p, ci_low=p, ci_high=p, n=10)
            for c, p in cats.items()
        ],
    )


def test_segment_grid_reads_only_asr_of_categorised_suites():
    arts = [
        _seg_art("C1", "harmful_harmbench_v1", {"cyber": 0.9, "chem": 0.8}),
        _seg_art("C5", "harmful_harmbench_v1", {"cyber": 0.1, "chem": 0.0}),
        _seg_art("C5", "overrefusal_xstest_v1", {"safe": 0.0}),  # not a SEGMENT_SUITES -> ignored
    ]
    grid = segment_asr_grid(arts)
    assert set(grid) == {"harmbench"}
    assert set(grid["harmbench"]) == {"cyber", "chem"}
    assert grid["harmbench"]["cyber"]["C1"].point == 0.9
    assert grid["harmbench"]["chem"]["C5"].point == 0.0


def test_failure_taxonomy_ranks_by_aligned_residual():
    arts = [
        _seg_art("C1", "harmful_harmbench_v1", {"lo": 0.50, "hi": 0.90}),
        _seg_art("C5", "harmful_harmbench_v1", {"lo": 0.02, "hi": 0.20}),
    ]
    rows = failure_taxonomy_rows(segment_asr_grid(arts))
    assert [r["category"] for r in rows] == ["hi", "lo"]  # higher C5 residual on top
    assert rows[0]["cells"]["C5"].point == 0.20 and rows[0]["cells"]["C1"].point == 0.90


def test_failure_taxonomy_render_md_csv_and_na():
    arts = [_seg_art("C5", "harmful_harmbench_v1", {"cyber": 0.05})]  # only C5 present
    rows = failure_taxonomy_rows(segment_asr_grid(arts))
    md = render_failure_taxonomy(rows, "md")
    assert "| Suite | Category | n |" in md
    assert "harmbench | cyber | 10 |" in md
    assert "| 0.050 |" in md  # C5 cell
    assert md.count("| -- |") >= 1  # C1/C6/C7/C8 absent -> N/A
    assert "harmbench,cyber,10" in render_failure_taxonomy(rows, "csv")


def test_render_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        render_failure_taxonomy([], "json")
