"""Phase 4b: the Phase-4 figure layer -- ASR + over-refusal by condition (master-plan plots #1, #2).

Two renderers, one data source. ``ablation_rows`` (Phase 4a) is the single input, so the committed
table and every plot are guaranteed to show the same numbers:

- **matplotlib static figures** -> ``reports/figures/*.svg`` + ``*.png`` (light, publication-style;
  matplotlib is a lazy import behind the optional ``[viz]`` extra, so importing this module never
  requires it).
- a **self-contained, theme-aware HTML dashboard** -> ``reports/dashboard.html`` (hand-rolled inline
  SVG so light/dark both work, decoupled from matplotlib's fixed colours).

Everything derives only from aggregate ``MetricsArtifact``s via ``ablation_rows`` (ADR-0007 dec.7):
no raw text. The categorical hues are the dataviz-validated palette (blue / orange / aqua for the
three ASR suites); aqua's sub-3:1 light contrast is relieved by the dashboard's table view + hover.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from safestack.eval.ablation import (
    _LABEL_HEADERS,
    MATRIX_COLUMNS,
    NA_CELL,
    AblationRow,
    ablation_from_paths,
)

# The three harmful/dual-use ASR suites grouped in the ASR chart: (matrix header, short label, hue).
ASR_SUITES: list[tuple[str, str, int]] = [
    ("ASR advbench", "advbench", 0),
    ("ASR harmbench", "harmbench", 1),
    ("ASR dual-use", "dual-use", 2),
]
# dataviz-validated categorical palette (slots 1-3), {light, dark}. See references/palette.md.
PALETTE = [
    ("#2a78d6", "#3987e5"),  # blue   -> advbench
    ("#eb6834", "#d95926"),  # orange -> harmbench
    ("#1baf7a", "#199e70"),  # aqua   -> dual-use
]
_OVERREFUSAL_HEADER = "Over-refusal"


@dataclass(frozen=True)
class Bar:
    condition: str
    point: float
    lo: float
    hi: float


def _bar(row: AblationRow, header: str) -> Bar | None:
    """A Bar for (row, column), or None when the cell is N/A (e.g. guardrail FPR on C1/C5)."""
    cell = row.cells.get(header)
    return None if cell is None else Bar(row.condition, cell.point, cell.ci_low, cell.ci_high)


def asr_series(rows: list[AblationRow]) -> dict[str, list[Bar | None]]:
    """``{suite header: [Bar | None per condition, in row order]}`` for the grouped ASR chart."""
    return {header: [_bar(r, header) for r in rows] for header, _, _ in ASR_SUITES}


def overrefusal_series(rows: list[AblationRow]) -> list[Bar | None]:
    """Over-refusal (XSTest) per condition, in row order."""
    return [_bar(r, _OVERREFUSAL_HEADER) for r in rows]


# --------------------------------------------------------------------------------------------------
# matplotlib static figures (light, publication-style). matplotlib is a lazy import ([viz] extra).
# --------------------------------------------------------------------------------------------------
def _save(fig, out_base: Path, formats: tuple[str, ...]) -> list[Path]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        # Drop the embedded creation timestamp so a re-render is byte-stable (ADR reproducibility).
        meta = {"Date": None} if fmt == "svg" else {"Software": None}
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=150, metadata=meta)
        written.append(path)
    return written


_ERRBAR = "#52514e"  # neutral CI cap/whisker colour (not matplotlib's default C0 blue)


def _pyplot():
    """Return the Agg pyplot module, or raise a friendly hint when the [viz] extra is missing."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - only without the optional extra
        raise RuntimeError(
            "matplotlib is required for the static figures; install it with `uv sync --extra viz` "
            "(or `pip install 'safestack[viz]'`), or use --dashboard-only for the HTML only."
        ) from exc
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "safestack"  # deterministic ids -> reproducible SVG
    import matplotlib.pyplot as plt

    return plt

def plot_asr_by_condition(
    rows: list[AblationRow], out_base: str | Path, *, formats: tuple[str, ...] = ("svg", "png")
) -> list[Path]:
    plt = _pyplot()
    conds = [r.condition for r in rows]
    series = asr_series(rows)
    n, width = len(conds), 0.26
    xs = list(range(n))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, (header, short, slot) in enumerate(ASR_SUITES):
        bars = series[header]
        xi = [x + (i - 1) * width for x in xs]
        pts = [b.point if b else 0.0 for b in bars]
        lo = [(b.point - b.lo) if b else 0.0 for b in bars]
        hi = [(b.hi - b.point) if b else 0.0 for b in bars]
        ax.bar(
            xi, pts, width, yerr=[lo, hi], capsize=2, ecolor=_ERRBAR,
            label=short, color=PALETTE[slot][0],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(conds)
    ax.set_ylabel("ASR (judge-unsafe & not blocked)")
    ax.set_ylim(0, 1)
    ax.set_title("Attack-success rate by condition (95% bootstrap CI)")
    ax.legend(title="suite", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, Path(out_base), formats)


def plot_over_refusal_by_condition(
    rows: list[AblationRow], out_base: str | Path, *, formats: tuple[str, ...] = ("svg", "png")
) -> list[Path]:
    plt = _pyplot()
    conds = [r.condition for r in rows]
    bars = overrefusal_series(rows)
    xs = list(range(len(conds)))
    pts = [b.point if b else 0.0 for b in bars]
    lo = [(b.point - b.lo) if b else 0.0 for b in bars]
    hi = [(b.hi - b.point) if b else 0.0 for b in bars]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    container = ax.bar(xs, pts, 0.6, yerr=[lo, hi], capsize=3, ecolor=_ERRBAR, color=PALETTE[0][0])
    ax.bar_label(container, labels=[f"{p:.3f}" for p in pts], padding=6, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Over-refusal (benign XSTest)")
    ax.set_title("Model over-refusal by condition (95% bootstrap CI)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, Path(out_base), formats)


# --------------------------------------------------------------------------------------------------
# Theme-aware inline-SVG dashboard (self-contained; light/dark via CSS custom properties).
# --------------------------------------------------------------------------------------------------
_MARGIN = {"l": 48, "r": 12, "t": 16, "b": 34}


def _svg_bar_chart(
    conds: list[str],
    groups: list[tuple[str, int, list[Bar | None]]],
    *,
    y_max: float,
    y_ticks: list[float],
    width: int = 720,
    height: int = 300,
    value_labels: bool = False,
) -> str:
    """A grouped bar chart as an inline SVG. ``groups`` = [(label, hue slot, [Bar|None per cond])].
    Colours come from CSS vars (``--s0..2``) so the page theme controls them; bars carry a native
    <title> for hover. Whiskers draw the 95% CI. N/A bars (None) are skipped, not drawn as zero.
    ``value_labels`` prints the point above each bar -- use it only for a low-count single-series
    chart (dataviz: never a number on every point, so it is off for the 24-bar ASR grid). All
    interpolated text (condition labels, hover) is HTML-escaped so a crafted metrics artifact cannot
    inject markup into the self-contained page."""
    pw = width - _MARGIN["l"] - _MARGIN["r"]
    ph = height - _MARGIN["t"] - _MARGIN["b"]
    x0, y0 = _MARGIN["l"], _MARGIN["t"]

    def sy(v: float) -> float:
        return y0 + ph * (1 - v / y_max)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'preserveAspectRatio="xMidYMid meet">'
    ]
    # gridlines + y ticks
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{y + 3:.1f}" text-anchor="end">{t:g}</text>'
        )
    # baseline
    parts.append(f'<line class="axis" x1="{x0}" y1="{sy(0):.1f}" x2="{x0 + pw}" y2="{sy(0):.1f}"/>')

    n_groups = len(conds)
    gw = pw / n_groups
    n_series = len(groups)
    bw = min(22.0, (gw * 0.72) / max(n_series, 1))
    for gi, cond in enumerate(conds):
        gx = x0 + gi * gw + gw / 2
        span = bw * n_series + 2 * (n_series - 1)
        start = gx - span / 2
        for si, (label, slot, bars) in enumerate(groups):
            b = bars[gi]
            bx = start + si * (bw + 2)
            if b is None:
                continue
            top = sy(b.point)
            cx = bx + bw / 2
            bh = max(0.0, sy(0) - top)
            tip = html.escape(f"{cond} {label}: {b.point:.3f} [{b.lo:.3f}, {b.hi:.3f}]")
            parts.append(
                f'<rect class="bar s{slot}" x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                f'height="{bh:.1f}" rx="2"><title>{tip}</title></rect>'
            )
            if b.hi > b.lo:  # CI whisker
                parts.append(
                    f'<line class="ci" x1="{cx:.1f}" y1="{sy(b.hi):.1f}" x2="{cx:.1f}" '
                    f'y2="{sy(b.lo):.1f}"/>'
                )
            if value_labels:
                ly = (sy(b.hi) if b.hi > b.lo else top) - 4
                parts.append(
                    f'<text class="val" x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle">'
                    f"{b.point:.3f}</text>"
                )
        parts.append(
            f'<text class="tick" x="{gx:.1f}" y="{y0 + ph + 16}" text-anchor="middle">'
            f"{html.escape(cond)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _legend(items: list[tuple[str, int]]) -> str:
    chips = "".join(
        f'<span class="chip"><span class="sw s{slot}"></span>{html.escape(label)}</span>'
        for label, slot in items
    )
    return f'<div class="legend">{chips}</div>'


def _html_table(rows: list[AblationRow]) -> str:
    headers = _LABEL_HEADERS + [h for _, _, h in MATRIX_COLUMNS]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = [r.condition, r.policy, r.guardrail]
        cells += [
            r.cells[h].fmt() if r.cells.get(h) is not None else NA_CELL
            for _, _, h in MATRIX_COLUMNS
        ]
        body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


_CSS = """
:root { color-scheme: light dark; }
.viz { --surface:#fcfcfb; --panel:#ffffff; --text:#0b0b0b; --muted:#52514e; --grid:#e6e5e1;
  --axis:#a9a8a2; --s0:#2a78d6; --s1:#eb6834; --s2:#1baf7a;
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  color:var(--text); background:var(--surface); padding:24px; max-width:840px; margin:0 auto; }
@media (prefers-color-scheme: dark) { .viz:where(:not([data-theme=light])) {
  --surface:#1a1a19; --panel:#232320; --text:#ffffff; --muted:#c3c2b7; --grid:#33322e;
  --axis:#6f6e68; --s0:#3987e5; --s1:#d95926; --s2:#199e70; } }
.viz[data-theme=dark] { --surface:#1a1a19; --panel:#232320; --text:#ffffff; --muted:#c3c2b7;
  --grid:#33322e; --axis:#6f6e68; --s0:#3987e5; --s1:#d95926; --s2:#199e70; }
.viz h1 { font-size:20px; margin:0 0 4px; } .viz h2 { font-size:15px; margin:26px 0 8px; }
.viz p.note { color:var(--muted); margin:2px 0 0; }
.panel { background:var(--panel); border:1px solid var(--grid); border-radius:10px; padding:14px; }
.chart { width:100%; height:auto; display:block; }
.chart .grid { stroke:var(--grid); stroke-width:1; }
.chart .axis { stroke:var(--axis); stroke-width:1.5; }
.chart .tick { fill:var(--muted); font-size:11px; }
.chart .ci { stroke:var(--text); stroke-width:1.5; opacity:.65; }
.chart .val { fill:var(--muted); font-size:11px; }
.chart .front { fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:4 3; }
.chart .dotlab { fill:var(--muted); font-size:10px; }
.chart .axtitle { fill:var(--muted); font-size:11px; }
.dot.s0 { fill:var(--s0); } .dot.s1 { fill:var(--s1); }
.bar.s0 { fill:var(--s0); } .bar.s1 { fill:var(--s1); } .bar.s2 { fill:var(--s2); }
.legend { display:flex; gap:16px; margin:6px 0 0; color:var(--muted); font-size:12px; }
.chip { display:inline-flex; align-items:center; gap:6px; }
.sw { width:11px; height:11px; border-radius:3px; display:inline-block; }
.sw.s0 { background:var(--s0); } .sw.s1 { background:var(--s1); } .sw.s2 { background:var(--s2); }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { border:1px solid var(--grid); padding:5px 8px; text-align:left; white-space:nowrap; }
th { color:var(--muted); font-weight:600; } .scroll { overflow-x:auto; }
"""


def build_dashboard_html(rows: list[AblationRow]) -> str:
    """The self-contained dashboard as an HTML string (theme-aware, aggregate-only)."""
    asr = asr_series(rows)
    conds = [r.condition for r in rows]
    asr_groups = [(short, slot, asr[header]) for header, short, slot in ASR_SUITES]
    asr_svg = _svg_bar_chart(conds, asr_groups, y_max=1.0, y_ticks=[0, 0.25, 0.5, 0.75, 1.0])
    legend = _legend([(short, slot) for _, short, slot in ASR_SUITES])
    orr = overrefusal_series(rows)
    orr_max = max([b.hi for b in orr if b] + [0.06])
    orr_top = round(orr_max + 0.01, 2)
    orr_svg = _svg_bar_chart(
        conds, [("over-refusal", 0, orr)], y_max=orr_top, y_ticks=[0, orr_top / 2, orr_top],
        value_labels=True,  # 8-bar single series: direct labels are legible (unlike the 24-bar ASR)
    )
    pts = pareto_points(rows)
    pareto_svg = _svg_scatter(pts, pareto_frontier(pts))
    pareto_legend = _legend([("SFT", 0), ("Starting", 1)])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>SafeStack — defense-in-depth (C1-C8)</title>"
        f"<style>{_CSS}</style></head><body>"
        "<div class='viz'>"
        "<h1>SafeStack — defense-in-depth ablation (C1-C8)</h1>"
        "<p class='note'>Aggregate-only. 2 policies (Starting / SFT) &times; 4 guardrail configs. "
        "ASR = judge-unsafe &amp; not blocked; error bars are 95% bootstrap CIs.</p>"
        "<h2>Attack-success rate by condition</h2>"
        f"<div class='panel'>{asr_svg}{legend}</div>"
        "<h2>Model over-refusal by condition (benign XSTest)</h2>"
        f"<div class='panel'>{orr_svg}</div>"
        "<h2>Safety vs benign-refusal cost (Pareto)</h2>"
        "<p class='note'>Safety = 1 &minus; mean ASR over the three harmful/dual-use suites "
        "(exploratory aggregate). Points below/right of the frontier are dominated.</p>"
        f"<div class='panel'>{pareto_svg}{pareto_legend}</div>"
        "<h2>Core ablation table</h2>"
        f"<div class='panel scroll'>{_html_table(rows)}</div>"
        "</div></body></html>"
    )


# --------------------------------------------------------------------------------------------------
# Safety vs benign-cost Pareto (master-plan plot #3). The master plan names a "safety/helpfulness"
# Pareto, but helpfulness is flat + block-blind here (4.915->4.910, ADR-0016), so the informative
# cost axis is the guardrail's benign-block rate (guardrail FPR; 0 on the no-guardrail conditions).
# Safety is 1 - mean ASR over the three harmful/dual-use suites -- an EXPLORATORY aggregate,
# not a confirmatory per-suite claim (ADR-0016 keeps per-suite ASR the confirmatory unit).
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParetoPoint:
    condition: str
    policy: str
    cost: float  # guardrail FPR (benign-block rate); 0.0 on a no-guardrail condition
    safety: float  # 1 - mean ASR
    mean_asr: float


def pareto_points(rows: list[AblationRow]) -> list[ParetoPoint]:
    """One (cost, safety) point per condition. Safety = 1 - mean ASR over ALL THREE harmful/dual-use
    suites: a condition missing any of them is a hard error, not a silent partial mean (which would
    over/under-state safety, and would report a degenerate safety=1.0 for a condition with no ASR at
    all). Cost = guardrail FPR (0 on a no-guardrail condition)."""
    asr_headers = [h for _, _, h in MATRIX_COLUMNS if h.startswith("ASR")]
    pts: list[ParetoPoint] = []
    for r in rows:
        asr_cells = [r.cells.get(h) for h in asr_headers]
        missing = [h for h, c in zip(asr_headers, asr_cells, strict=True) if c is None]
        if missing:
            raise ValueError(
                f"condition {r.condition} is missing ASR suite(s) {missing}; the Pareto safety "
                "(1 - mean ASR) requires all three harmful/dual-use suites."
            )
        mean_asr = sum(c.point for c in asr_cells) / len(asr_cells)
        fpr = r.cells.get("Guardrail FPR")
        cost = fpr.point if fpr is not None else 0.0  # N/A guardrail FPR -> no benign-block cost
        pts.append(ParetoPoint(r.condition, r.policy, cost, 1.0 - mean_asr, mean_asr))
    return pts


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """The non-dominated set (maximise safety, minimise cost), sorted by cost. A point survives
    unless another has cost <= and safety >=, strictly better on one axis (exact ties both stay)."""
    front = [
        p
        for p in points
        if not any(
            q is not p
            and q.cost <= p.cost
            and q.safety >= p.safety
            and (q.cost < p.cost or q.safety > p.safety)
            for q in points
        )
    ]
    return sorted(front, key=lambda p: (p.cost, -p.safety))


def plot_pareto(
    rows: list[AblationRow], out_base: str | Path, *, formats: tuple[str, ...] = ("svg", "png")
) -> list[Path]:
    plt = _pyplot()
    pts = pareto_points(rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for policy, hue in (("Starting", PALETTE[1][0]), ("SFT", PALETTE[0][0])):
        ps = [p for p in pts if p.policy == policy]
        ax.scatter(
            [p.cost for p in ps], [p.safety for p in ps], s=64, color=hue, label=policy, zorder=3
        )
        for p in ps:
            ax.annotate(
                p.condition, (p.cost, p.safety), textcoords="offset points",
                xytext=(5, 4), fontsize=8,
            )
    front = pareto_frontier(pts)
    ax.plot(
        [p.cost for p in front], [p.safety for p in front],
        color=_ERRBAR, lw=1.2, ls="--", zorder=2, label="Pareto frontier",
    )
    ax.set_xlabel("Benign-refusal cost (guardrail FPR; 0 = no guardrail)")
    ax.set_ylabel("Safety = 1 - mean ASR (advbench / harmbench / dual-use)")
    ax.set_title("Safety vs benign-refusal cost (C1-C8)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, Path(out_base), formats)


_POLICY_SLOT = {"SFT": 0, "Starting": 1}  # dashboard scatter hue by policy


def _svg_scatter(points: list[ParetoPoint], frontier: list[ParetoPoint], width: int = 560,
                 height: int = 340) -> str:
    """The Pareto as a theme-aware inline-SVG scatter (dots coloured by policy, dashed frontier)."""
    m = {"l": 54, "r": 14, "t": 12, "b": 40}
    pw, ph = width - m["l"] - m["r"], height - m["t"] - m["b"]
    x0, y0 = m["l"], m["t"]
    x_max = 0.4
    y_min, y_max = 0.3, 1.0
    # Fail loud on a point outside the fixed window rather than silently clamping it onto an axis
    # (which would mis-place it vs the correctly-computed frontier). Widen the bounds when it fires.
    for p in points:
        if not (0.0 <= p.cost <= x_max and y_min <= p.safety <= y_max):
            raise ValueError(
                f"Pareto point {p.condition} (cost={p.cost:.3f}, safety={p.safety:.3f}) is outside "
                f"the [0,{x_max}] x [{y_min},{y_max}] window; widen _svg_scatter's bounds."
            )

    def sx(v: float) -> float:
        return x0 + pw * (v / x_max)

    def sy(v: float) -> float:
        return y0 + ph * (1 - (v - y_min) / (y_max - y_min))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
             'preserveAspectRatio="xMidYMid meet">']
    for t in (0.3, 0.5, 0.7, 0.9):
        gy = sy(t)
        parts.append(f'<line class="grid" x1="{x0}" y1="{gy:.1f}" x2="{x0 + pw}" y2="{gy:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{gy + 3:.1f}" text-anchor="end">{t:g}</text>'
        )
    for t in (0.0, 0.1, 0.2, 0.3, 0.4):
        parts.append(
            f'<text class="tick" x="{sx(t):.1f}" y="{y0 + ph + 16}" '
            f'text-anchor="middle">{t:g}</text>'
        )
    if len(frontier) > 1:
        d = " ".join(f"{sx(p.cost):.1f},{sy(p.safety):.1f}" for p in frontier)
        parts.append(f'<polyline class="front" points="{d}"/>')
    for p in points:
        cx, cy = sx(p.cost), sy(p.safety)
        tip = html.escape(f"{p.condition} ({p.policy}): safety {p.safety:.3f}, cost {p.cost:.3f}")
        parts.append(
            f'<circle class="dot s{_POLICY_SLOT[p.policy]}" cx="{cx:.1f}" cy="{cy:.1f}" r="5">'
            f"<title>{tip}</title></circle>"
        )
        parts.append(
            f'<text class="dotlab" x="{cx + 7:.1f}" y="{cy - 5:.1f}">'
            f"{html.escape(p.condition)}</text>"
        )
    parts.append(
        f'<text class="axtitle" x="{x0 + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        "benign-refusal cost (guardrail FPR; 0 = none)</text>"
    )
    parts.append("</svg>")
    return "".join(parts)

def write_figures(
    rows: list[AblationRow],
    *,
    figures_dir: str | Path = "reports/figures",
    dashboard: str | Path | None = "reports/dashboard.html",
    formats: tuple[str, ...] = ("svg", "png"),
    dashboard_only: bool = False,
) -> list[Path]:
    """Write the matplotlib figures and (optionally) the HTML dashboard; return the paths.

    ``dashboard_only=True`` skips the matplotlib pass entirely, so the theme-aware HTML (which needs
    no matplotlib) can be produced on an install without the optional ``[viz]`` extra."""
    written: list[Path] = []
    if not dashboard_only:
        figures_dir = Path(figures_dir)
        written += plot_asr_by_condition(rows, figures_dir / "asr_by_condition", formats=formats)
        written += plot_over_refusal_by_condition(
            rows, figures_dir / "over_refusal_by_condition", formats=formats
        )
        written += plot_pareto(rows, figures_dir / "safety_cost_pareto", formats=formats)
    if dashboard is not None:
        dpath = Path(dashboard)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        dpath.write_text(build_dashboard_html(rows), encoding="utf-8")
        written.append(dpath)
    return written


def figures_from_paths(paths: list[str | Path], **kw) -> list[Path]:
    return write_figures(ablation_from_paths(paths), **kw)
