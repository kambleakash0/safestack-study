"""`safestack eval` commands: run (generate) -> judge -> report -> compare (ADR-0007)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from safestack.eval.ablation import (
    ablation_rows,
    before_after_rows,
    write_ablation,
    write_before_after,
)
from safestack.eval.figures import figures_from_paths
from safestack.eval.generate import run_suite
from safestack.eval.judges import _load_cfg_from_run, judge_run
from safestack.eval.report import compare as compare_reports
from safestack.eval.report import load_artifacts, write_report
from safestack.eval.segments import (
    failure_taxonomy_rows,
    segment_asr_grid,
    write_failure_taxonomy,
)

app = typer.Typer(
    help="Evaluation harness: generate, judge, report, compare.", no_args_is_help=True
)

_DATA = typer.Option(Path("data"), "--data-dir", help="Root data directory.")
_CACHE = typer.Option(None, "--cache-dir", help="Cache directory (default: <data-dir>/cache).")
_MODELS = typer.Option(Path("configs/models"), "--models-dir", help="Model-card directory.")


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="EvalExperimentConfig YAML."),
    backend: str | None = typer.Option(None, "--backend", help="Override the policy backend."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory for run artifacts."),
    data_dir: Path = _DATA,
    cache_dir: Path | None = _CACHE,
    models_dir: Path = _MODELS,
    limit: int | None = typer.Option(None, "--limit", help="Cap records per suite (debug)."),
) -> None:
    """PASS A: generate every prepared record through the policy gateway, content-hash cached."""
    run_dir = run_suite(
        config,
        backend_override=backend,
        runs_dir=runs_dir,
        data_dir=data_dir,
        cache_dir=cache_dir,
        models_dir=models_dir,
        limit=limit,
    )
    typer.echo(f"run: {run_dir}")


@app.command("judge")
def judge_cmd(
    run: Path = typer.Option(..., "--run", help="Run directory from `eval run`."),
    kind: str = typer.Option("all", "--kind", help="safety | refusal | helpfulness | all."),
    data_dir: Path = _DATA,
    cache_dir: Path | None = _CACHE,
    models_dir: Path = _MODELS,
) -> None:
    """PASS B: score the run's generation cache with one judge at a time."""
    counts = judge_run(
        run,
        data_dir=data_dir,
        cache_dir=cache_dir,
        models_dir=models_dir,
        kinds=None if kind == "all" else [kind],
    )
    for role, c in counts.items():
        typer.echo(f"judge {role}: scored={c['scored']} hits={c['hits']}")


@app.command("report")
def report_cmd(
    run: Path = typer.Option(..., "--run", help="Run directory from `eval run`."),
    suite: str | None = typer.Option(None, "--suite", help="Suite to report (default: all)."),
    out: Path | None = typer.Option(None, "--out", help="Output path (single suite only)."),
    reports_dir: Path = typer.Option(Path("reports"), "--reports-dir", help="Reports directory."),
    data_dir: Path = _DATA,
    cache_dir: Path | None = _CACHE,
    models_dir: Path = _MODELS,
    bootstrap_seed: int | None = typer.Option(None, "--bootstrap-seed"),
    bootstrap_n: int | None = typer.Option(None, "--bootstrap-n"),
) -> None:
    """PASS C: compute metrics + bootstrap CIs from the caches; write aggregate-only artifacts."""
    cfg = _load_cfg_from_run(Path(run))
    suites = [suite] if suite else list(cfg.suites)
    for name in suites:
        path = write_report(
            run,
            cfg,
            name,
            out=out if suite else None,
            reports_dir=reports_dir,
            data_dir=data_dir,
            cache_dir=cache_dir,
            models_dir=models_dir,
            bootstrap_seed=bootstrap_seed,
            bootstrap_n=bootstrap_n,
        )
        typer.echo(f"report: {path}")


@app.command("compare")
def compare_cmd(
    metrics: list[Path] = typer.Option(..., "--metrics", help="MetricsArtifact JSON files."),
    fmt: str = typer.Option("md", "--format", help="md | csv."),
    out: Path | None = typer.Option(None, "--out", help="Write the table to a file."),
    gate: bool = typer.Option(False, "--gate", help="Print the ADR-0002 dynamic-range readout."),
) -> None:
    """PASS D: paired ASR/over-refusal/helpfulness table with CIs + the dynamic-range gate."""
    text = compare_reports(list(metrics), out=out, fmt=fmt, gate=gate)
    typer.echo(text)

@app.command("ablation")
def ablation_cmd(
    metrics: list[Path] = typer.Option(..., "--metrics", help="MetricsArtifact JSONs (C1-C8)."),
    out: Path = typer.Option(
        Path("reports/tables/core_ablation"), "--out", help="Core-ablation output base (no suffix)."
    ),
    before_after_out: Path = typer.Option(
        Path("reports/tables/sft_before_after"),
        "--before-after-out",
        help="SFT before/after output base (no suffix).",
    ),
    taxonomy_out: Path = typer.Option(
        Path("reports/tables/failure_taxonomy"),
        "--taxonomy-out",
        help="Failure-taxonomy output base (no suffix).",
    ),
    fmt: str = typer.Option("both", "--format", help="md | csv | both."),
) -> None:
    """Phase 4: the core 2x4 ablation matrix, the SFT before/after table, and the per-category
    failure taxonomy (all aggregate-only)."""
    arts = load_artifacts(list(metrics))  # loaded once -> rows + the segment grid
    rows = ablation_rows(arts)
    formats = ("csv", "md") if fmt == "both" else (fmt,)
    for path in write_ablation(rows, out, formats=formats):
        typer.echo(f"ablation: {path}")
    for path in write_before_after(before_after_rows(rows), before_after_out, formats=formats):
        typer.echo(f"before/after: {path}")
    taxonomy = failure_taxonomy_rows(segment_asr_grid(arts))
    for path in write_failure_taxonomy(taxonomy, taxonomy_out, formats=formats):
        typer.echo(f"failure taxonomy: {path}")

@app.command("figures")
def figures_cmd(
    metrics: list[Path] = typer.Option(..., "--metrics", help="MetricsArtifact JSONs (C1-C8)."),
    figures_dir: Path = typer.Option(
        Path("reports/figures"), "--figures-dir", help="matplotlib SVG/PNG output dir."
    ),
    dashboard: Path = typer.Option(
        Path("reports/dashboard.html"), "--dashboard", help="Theme-aware HTML dashboard path."
    ),
    dashboard_only: bool = typer.Option(
        False, "--dashboard-only", help="Only write the HTML dashboard (no matplotlib / [viz])."
    ),
) -> None:
    """Phase 4: ASR + over-refusal figures (matplotlib) plus the theme-aware HTML dashboard."""
    paths = figures_from_paths(
        list(metrics), figures_dir=figures_dir, dashboard=dashboard, dashboard_only=dashboard_only
    )
    for path in paths:
        typer.echo(f"figure: {path}")


@app.callback()
def _configure_logging() -> None:
    """Surface run/judge progress logs, mirroring `safestack run` (safestack/cli.py)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
