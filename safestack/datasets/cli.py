"""`safestack data` commands: prepare and validate eval suites."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from safestack.datasets.prepare import prepare
from safestack.datasets.schema import DatasetPrepConfig
from safestack.datasets.validate import prompt_overlap, validate_manifest

app = typer.Typer(help="Dataset preparation and validation.", no_args_is_help=True)


@app.command("prepare")
def prepare_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="DatasetPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare one eval suite: fetch, filter, dedup, and write manifest + sanitized samples."""
    cfg = DatasetPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifest = prepare(cfg, data_dir=data_dir)
    typer.echo(f"prepared {manifest.name}: {manifest.num_examples} records -> {manifest.hash}")


@app.command("validate")
def validate_cmd(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Manifest YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Check a prepared suite's content hash against its manifest and report suite overlap."""
    m = validate_manifest(manifest, data_dir=data_dir)
    typer.echo(f"OK {m.name}: {m.num_examples} records, hash matches")
    overlaps = prompt_overlap(data_dir=data_dir)
    if overlaps:
        typer.echo(f"WARN prompt overlap across suites: {overlaps}")
