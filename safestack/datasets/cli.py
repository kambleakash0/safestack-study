"""`safestack data` commands: prepare and validate eval suites."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from safestack.datasets.prepare import prepare, prepare_sft
from safestack.datasets.schema import DatasetPrepConfig, SFTPrepConfig
from safestack.datasets.validate import prompt_overlap, train_eval_overlap, validate_manifest

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

@app.command("overlap")
def overlap_cmd(
    train_split: str | None = typer.Option(
        None, "--train-split", help="If set, report train-vs-eval leakage for this split."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", help="Jaccard threshold for near-duplicate matches."
    ),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Report prompt overlap: cross-suite (exact) or, with --train-split, train-vs-eval leakage
    (exact + near-duplicate). Prints identifiers and counts only, never raw prompt text. Exits
    non-zero when overlap is found so a training script can gate on it (ADR-0015 follow-up 2)."""
    if train_split:
        rep = train_eval_overlap(train_split, data_dir=data_dir, threshold=threshold)
        typer.echo(
            f"{train_split}: {rep['n_train']} train records vs {len(rep['eval_suites'])} eval "
            f"suites -> {rep['n_exact']} exact, {rep['n_near_dup']} near-dup (>= {threshold})"
        )
        for hit in rep["exact"] + rep["near_dup"]:
            kind = "exact" if hit["exact"] else "near"
            typer.echo(
                f"  {hit['train_file']}[{hit['train_index']}] ~ {hit['eval_suite']}/"
                f"{hit['eval_id']} ({kind}, jaccard {hit['jaccard']})"
            )
        if rep["n_exact"] or rep["n_near_dup"]:
            raise typer.Exit(code=1)
    else:
        overlaps = prompt_overlap(data_dir=data_dir)
        typer.echo(f"cross-suite overlap: {overlaps}" if overlaps else "no cross-suite overlap")
        if overlaps:
            raise typer.Exit(code=1)

@app.command("prepare-sft")
def prepare_sft_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="SFTPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare the SFT training suite (train_sft): fetch, blend refuse-harmful + comply-benign,
    dedup, balance, and write the manifest + sanitized samples (user prompts hashed)."""
    cfg = SFTPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifest = prepare_sft(cfg, data_dir=data_dir)
    typer.echo(f"prepared {manifest.name}: {manifest.num_examples} records -> {manifest.hash}")
