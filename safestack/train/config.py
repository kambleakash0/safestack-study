"""SFT training config (ADR-0015 decision 3): the pinned LoRA/QLoRA recipe. One committed config per
run; the trainer never invents hyperparameters. If a search is run it is a small pre-specified grid
selected on the DEV suite by rule-9 (decision 4), never on the locked test or training loss."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_TRAIN_SCHEMA_VERSION = 1

# Mistral attention + MLP projections -- the standard LoRA target set for this architecture.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class SFTTrainConfig(BaseModel):
    """A declarative recipe for one LoRA/QLoRA SFT run on the frozen base (ADR-0015 dec.3).

    ``base_model`` names a committed model card (its checkpoint + pinned revision are the frozen
    base); ``train_suite`` names a prepared train_sft suite (its manifest pins the data revision +
    hash). The LoRA adapter is written to ``output_adapter``, which stays PRIVATE (the adapters/ dir
    is gitignored) -- only the aggregate loss curves and this config are tracked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    base_model: str  # model_id of a committed base card (checkpoint + revision = the frozen base)
    train_suite: str  # prepared train_sft suite name (manifest pins the data revision + hash)
    output_adapter: str  # dir the LoRA adapter is written to (PRIVATE -- adapters/ is gitignored)

    # Continue-train from an existing adapter instead of a fresh LoRA (ADR-0017 FU1, Phase 5).
    # None (or empty) = fresh LoRA (the Phase-3 SFT behavior). When set, the trainer resumes the
    # named adapter's LoRA params on this train_suite; rank/alpha/target_modules then come from that
    # adapter's saved config, not this cfg. init_adapter_revision pins which hub revision is loaded
    # for training (forwarded to PeftModel.from_pretrained(revision=...)) -- a training-load pin,
    # not the eval content-hash pin (ModelSpec.adapter_revision, on the stressed output card).
    init_adapter: str | None = None
    init_adapter_revision: str | None = None

    # LoRA (ADR-0015 dec.3)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Optimisation (starting hyperparameters, master-plan template)
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048

    # Memory / quantisation (ADR-0003: 7B QLoRA at a 16-24 GB single-GPU target)
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Reproducibility + tracked validation loss (curves committed; NOT a selection knob -- rule 9)
    seed: int = 20250115
    val_fraction: float = 0.05  # held-out slice of train_sft for tracked val loss (not selection)
    logging_steps: int = 10
    save_strategy: str = "epoch"  # a checkpoint per epoch, for rule-9 DEV selection (FU5b)

    schema_version: int = 1
