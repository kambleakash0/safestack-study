"""`safestack train` commands: run the SFT trainer."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from safestack.train.config import SFTTrainConfig
from safestack.train.sft import train_sft

app = typer.Typer(help="SFT training (LoRA/QLoRA).", no_args_is_help=True)


@app.command("sft")
def sft_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="SFTTrainConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
    models_dir: Path = typer.Option(
        Path("configs/models"), "--models-dir", help="Model-card directory."
    ),
) -> None:
    """Train a LoRA/QLoRA adapter on the frozen base with assistant-only loss (needs the `train`
    extra + a GPU). Saves the adapter (private, gitignored) and aggregate loss curves."""
    cfg = SFTTrainConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    summary = train_sft(cfg, data_dir=data_dir, models_dir=models_dir)
    typer.echo(
        f"trained {cfg.name}: {summary['n_train']} train / {summary['n_val']} val, "
        f"final_train_loss={summary['final_train_loss']}, adapter -> {summary['adapter']}"
    )
