"""`safestack eval` commands: run (generate) -> judge -> report -> compare (ADR-0007)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from safestack.eval.generate import run_suite
from safestack.eval.judges import _load_cfg_from_run, judge_run
from safestack.eval.report import compare as compare_reports
from safestack.eval.report import write_report

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


@app.callback()
def _configure_logging() -> None:
    """Surface run/judge progress logs, mirroring `safestack run` (safestack/cli.py)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
