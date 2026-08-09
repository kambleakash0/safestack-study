"""Phase 4a: pivot the C1-C8 per-suite MetricsArtifacts into the core 2x4 ablation matrix.

``eval compare`` emits a long/tidy table (one row per metric); this module pivots it into the wide
per-condition matrix the master plan (section 18.3) and the MVP tier (section 19.1) call for: one
row per condition (2 policies {Starting, SFT} x 4 guardrails {none, input, output, input+output}),
each cell a ``point [lo, hi]`` 95% bootstrap CI. Per-suite ASR is kept, never pooled, matching how
every result ADR reports (advbench / harmbench / dual-use are distinct). Reads only aggregate
MetricsArtifacts (ADR-0007 decision 7) -- no raw text ever enters this path.

``ablation_rows`` returns structured rows; it is the shared input for the Phase-4 figure layer, so
the table and the plots never diverge.

Two deliberate divergences from the section-18.3 template, both documented so 4b/4c inherit the
decision rather than an accident: (1) **guardrail FNR is omitted** from the headline -- it is the
guardrail's harmful-miss diagnostic, exploratory on an aligned policy that leaves almost no residue
to catch (ADR-0016 follow-up 3), the same way the latency columns are omitted until Phase 8; and
(2) the guardrail-cost cells (FPR) are rendered **N/A on the no-guardrail conditions** (C1, C5),
because the artifact's degenerate 0.0 FPR over an absent screen must not read as "never
false-positives" (matching the section-18.3 N/A convention).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from safestack.eval.artifacts import MetricsArtifact
from safestack.eval.report import load_artifacts

# The 2x4 core ablation, in display order (ADR-0008 base anchors, ADR-0015 SFT rungs).
CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
_STARTING = {"C1", "C2", "C3", "C4"}
POLICY_LABEL = {c: ("Starting" if c in _STARTING else "SFT") for c in CONDITION_ORDER}
GUARDRAIL_LABEL = {
    "C1": "none", "C2": "input", "C3": "output", "C4": "input+output",
    "C5": "none", "C6": "input", "C7": "output", "C8": "input+output",
}

# (metric, suite, header) columns of the headline matrix, in display order. ASR is per harmful/
# dual-use suite; the paired benign costs (over-refusal, helpfulness, guardrail FPR) are one per
# condition, from the XSTest / Alpaca suites (ADR-0004 rule 5 pairing).
MATRIX_COLUMNS: list[tuple[str, str, str]] = [
    ("asr", "harmful_advbench_v1", "ASR advbench"),
    ("asr", "harmful_harmbench_v1", "ASR harmbench"),
    ("asr", "dualuse_harmbench_contextual_v1", "ASR dual-use"),
    ("over_refusal", "overrefusal_xstest_v1", "Over-refusal"),
    ("benign_helpfulness", "helpfulness_alpaca_v1", "Helpfulness"),
    ("guardrail_fpr", "overrefusal_xstest_v1", "Guardrail FPR"),
]

# Guardrail-cost metrics: N/A on a no-guardrail condition (there is no screen to false-positive).
_GUARDRAIL_METRICS = frozenset({"guardrail_fpr", "guardrail_fnr"})
_LABEL_HEADERS = ["Condition", "Policy", "Guardrail"]

# The single N/A placeholder, shared with the dashboard's HTML table so both committed tables agree.
NA_CELL = "--"


@dataclass(frozen=True)
class Cell:
    point: float
    ci_low: float
    ci_high: float
    n: int

    def fmt(self) -> str:
        return f"{self.point:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


@dataclass(frozen=True)
class AblationRow:
    condition: str
    policy: str
    guardrail: str
    cells: dict[str, Cell | None]  # header -> Cell (None when a condition lacks that suite/metric)


def _index(arts: list[MetricsArtifact]) -> dict[tuple[str, str, str], Cell]:
    """``{(condition, suite, metric): Cell}``. The C1-C4 split families repeat the XSTest / Alpaca
    suites across their ``_starting`` and ``_dualuse`` experiment files. Those duplicates share an
    identical point estimate and n (a *differing* point/n is a hard error -- a real data conflict),
    but their seeded bootstrap CIs can differ by a hair (a resample-order effect), so the CI is
    merged as the conservative envelope ``[min lo, max hi]`` -- never narrower than either."""
    idx: dict[tuple[str, str, str], Cell] = {}
    for a in arts:
        for m in a.metrics:
            key = (a.condition_id, a.suite, m.name)
            prev = idx.get(key)
            if prev is None:
                idx[key] = Cell(m.point, m.ci_low, m.ci_high, m.n)
                continue
            if prev.point != m.point or prev.n != m.n:
                raise ValueError(
                    f"conflicting duplicate metric for {key}: "
                    f"point/n {prev.point}/{prev.n} vs {m.point}/{m.n}"
                )
            idx[key] = Cell(
                prev.point, min(prev.ci_low, m.ci_low), max(prev.ci_high, m.ci_high), prev.n
            )
    return idx


def ablation_rows(arts: list[MetricsArtifact]) -> list[AblationRow]:
    """Pivot artifacts into one row per present condition, in CONDITION_ORDER.

    Guardrail-cost cells are blanked (None -> "--") on the no-guardrail conditions (C1, C5). A
    condition_id outside CONDITION_ORDER is a hard error rather than a silent drop, so an unexpected
    or Phase-5 (C9/C10) artifact is loud -- extend CONDITION_ORDER to admit it deliberately."""
    idx = _index(arts)
    present = {cond for (cond, _, _) in idx}
    unknown = sorted(present - set(CONDITION_ORDER))
    if unknown:
        raise ValueError(
            f"metrics carry condition_id(s) outside the C1-C8 ablation: {unknown}. "
            "Extend CONDITION_ORDER (e.g. for the Phase-5 C9/C10 rungs) to include them."
        )
    rows: list[AblationRow] = []
    for cond in CONDITION_ORDER:
        if cond not in present:
            continue
        no_guardrail = GUARDRAIL_LABEL[cond] == "none"
        cells: dict[str, Cell | None] = {}
        for metric, suite, header in MATRIX_COLUMNS:
            if no_guardrail and metric in _GUARDRAIL_METRICS:
                cells[header] = None  # N/A: no screen to false-positive / miss
            else:
                cells[header] = idx.get((cond, suite, metric))
        rows.append(AblationRow(cond, POLICY_LABEL[cond], GUARDRAIL_LABEL[cond], cells))
    return rows


def _table(rows: list[AblationRow]) -> tuple[list[str], list[list[str]]]:
    headers = _LABEL_HEADERS + [h for _, _, h in MATRIX_COLUMNS]
    body: list[list[str]] = []
    for r in rows:
        cells = [
            r.cells[h].fmt() if r.cells.get(h) is not None else NA_CELL
            for _, _, h in MATRIX_COLUMNS
        ]
        body.append([r.condition, r.policy, r.guardrail, *cells])
    return headers, body


def render(rows: list[AblationRow], fmt: str) -> str:
    """Render the matrix. Both formats carry the same ``point [lo, hi]`` cells; CSV uses the csv
    module so the comma inside each CI is quoted, not a column break."""
    headers, body = _table(rows)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(body)
        return buf.getvalue()
    if fmt == "md":
        head = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        lines = ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join([head, sep, *lines]) + "\n"
    raise ValueError(f"unknown format {fmt!r} (use 'md' or 'csv')")


def write_ablation(
    rows: list[AblationRow],
    out_base: str | Path,
    *,
    formats: tuple[str, ...] = ("csv", "md"),
) -> list[Path]:
    """Write ``{out_base}.csv`` / ``{out_base}.md``. ``out_base`` is a path with NO suffix."""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        path.write_text(render(rows, fmt), encoding="utf-8")
        written.append(path)
    return written


def ablation_from_paths(paths: list[str | Path]) -> list[AblationRow]:
    return ablation_rows(load_artifacts(paths))
