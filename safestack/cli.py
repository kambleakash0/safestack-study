"""SafeStack command-line interface (Typer). One config-driven `run` command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from safestack.datasets.cli import app as data_app
from safestack.eval.cli import app as eval_app
from safestack.runner import run_experiment
from safestack.train.cli import app as train_app

app = typer.Typer(
    help="SafeStack: defense-in-depth LLM safety evaluation.",
    no_args_is_help=True,
)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to an experiment config YAML."),
    backend: str | None = typer.Option(None, "--backend", help="Override the model backend."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Load and log the config; skip generation."
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory for run artifacts."),
    models_dir: Path = typer.Option(
        Path("configs/models"), "--models-dir", help="Model-card directory."
    ),
) -> None:
    """Run one experiment config through the model gateway."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_experiment(
        config,
        backend_override=backend,
        runs_dir=runs_dir,
        dry_run=dry_run,
        models_dir=models_dir,
    )
    if result is None:
        typer.echo(f"dry-run complete; run record written under {runs_dir}")
    else:
        typer.echo(f"generated ({result.backend}): {result.text[:200]}")


@app.callback()
def main() -> None:
    """SafeStack CLI (this callback keeps `run` a named subcommand)."""


app.add_typer(data_app, name="data")

app.add_typer(eval_app, name="eval")

app.add_typer(train_app, name="train")
