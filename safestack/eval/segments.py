"""Phase 4d/4f: read the per-category ASR segments the core ablation ignores, into a heatmap grid
and a failure taxonomy.

The metrics artifacts carry a ``segments`` list (per-category ASR) that ``ablation`` never reads.
Only two suites have meaningful category structure -- ``harmful_harmbench_v1`` and
``dualuse_harmbench_contextual_v1``, 6 HarmBench semantic categories each. This module surfaces them
as (a) a category x condition ASR grid for the heatmap and (b) a failure taxonomy ranked by the
aligned model's residual, so a reader can see where residual harm concentrates and whether an
external guardrail closes it.

**EXPLORATORY only (ADR-0016 consequence 2).** The categories have tiny n (1-58) with degenerate
near-0/1 bootstrap CIs, so every artifact here is supporting texture -- never a confirmatory
per-category claim. Aggregate-only (category name + n + ASR point); no raw text.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from safestack.eval.ablation import NA_CELL
from safestack.eval.artifacts import MetricsArtifact
from safestack.eval.report import load_artifacts

# Suites that carry per-category ASR segments -> the short label used in the grid/heatmap/taxonomy.
SEGMENT_SUITES = {
    "harmful_harmbench_v1": "harmbench",
    "dualuse_harmbench_contextual_v1": "dual-use",
}
# Conditions shown in the taxonomy, in order: the base anchor (C1) + the four SFT rungs (C5-C8).
# The base+guardrail rungs (C2-C4) are intentionally omitted -- the taxonomy's question is "where
# does the ALIGNED model still fail, and does a guardrail close it", so C1 (context) + the SFT rungs
# suffice. (The heatmap, by contrast, shows all of C1-C8 for the full picture.)
TAXONOMY_CONDITIONS = ["C1", "C5", "C6", "C7", "C8"]
# An in-artifact caveat so the committed table carries its own exploratory framing (ADR-0016).
_TAXONOMY_CAPTION = (
    "Exploratory (ADR-0016): tiny per-category n (1-58) with degenerate near-0/1 CIs -- supporting "
    "texture, not a confirmatory per-category claim. Ranked by the aligned (C5) residual ASR."
)
_TAXONOMY_HEADERS = [
    "Suite", "Category", "n", "C1 base", "C5 SFT", "C6 +input", "C7 +output", "C8 +both",
]


@dataclass(frozen=True)
class SegCell:
    point: float
    n: int


# suite label -> {category -> {condition_id -> SegCell}}
SegmentGrid = dict[str, dict[str, dict[str, SegCell]]]


def segment_asr_grid(arts: list[MetricsArtifact]) -> SegmentGrid:
    """Read the ASR segments of the two categorised suites into a nested grid."""
    grid: SegmentGrid = {}
    for a in arts:
        label = SEGMENT_SUITES.get(a.suite)
        if label is None:
            continue
        for s in a.segments:
            if s.metric != "asr":
                continue
            grid.setdefault(label, {}).setdefault(s.segment, {})[a.condition_id] = SegCell(
                s.point, s.n
            )
    return grid


def _n_of(by_cond: dict[str, SegCell]) -> int:
    """The category's sample size -- identical across conditions (same locked suite); use any."""
    cell = by_cond.get("C1") or next(iter(by_cond.values()), None)
    return cell.n if cell is not None else 0


def failure_taxonomy_rows(grid: SegmentGrid) -> list[dict]:
    """One row per (suite, category), ranked by the aligned model's residual (C5 ASR, descending) so
    the worst-remaining aligned failures are on top. Ties break by suite then category."""
    rows: list[dict] = []
    for suite_label, cats in grid.items():
        for cat, by_cond in cats.items():
            rows.append({
                "suite": suite_label,
                "category": cat,
                "n": _n_of(by_cond),
                "cells": {c: by_cond.get(c) for c in TAXONOMY_CONDITIONS},
            })
    rows.sort(key=lambda r: (-(r["cells"]["C5"].point if r["cells"]["C5"] else 0.0),
                             r["suite"], r["category"]))
    return rows


def render_failure_taxonomy(rows: list[dict], fmt: str) -> str:
    def cell(r: dict, cond: str) -> str:
        c = r["cells"][cond]
        return f"{c.point:.3f}" if c is not None else NA_CELL

    body = [
        [r["suite"], r["category"], str(r["n"]), *[cell(r, c) for c in TAXONOMY_CONDITIONS]]
        for r in rows
    ]
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(_TAXONOMY_HEADERS)
        writer.writerows(body)
        return buf.getvalue()
    if fmt == "md":
        head = "| " + " | ".join(_TAXONOMY_HEADERS) + " |"
        sep = "| " + " | ".join("---" for _ in _TAXONOMY_HEADERS) + " |"
        lines = ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join([f"_{_TAXONOMY_CAPTION}_", "", head, sep, *lines]) + "\n"
    raise ValueError(f"unknown format {fmt!r} (use 'md' or 'csv')")


def write_failure_taxonomy(
    rows: list[dict], out_base: str | Path, *, formats: tuple[str, ...] = ("csv", "md")
) -> list[Path]:
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        path.write_text(render_failure_taxonomy(rows, fmt), encoding="utf-8")
        written.append(path)
    return written


def segments_from_paths(paths: list[str | Path]) -> SegmentGrid:
    return segment_asr_grid(load_artifacts(paths))
