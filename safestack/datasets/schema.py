"""Schemas for prepared evaluation records and dataset-prep recipes (ADR-0006, §8.6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExpectedBehavior = Literal["refuse_or_safe_redirect", "answer_normally"]

# Eval-only subset of the master-plan §8.2 splits. Eval prep never targets a train split.
EvalSplit = Literal[
    "eval_harmful",
    "eval_dual_use",
    "eval_benign_overrefusal",
    "eval_benign_helpfulness",
    "eval_human_audit",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalRecord(_Frozen):
    eval_id: str
    suite: str
    category: str = ""
    prompt: str
    expected_behavior: ExpectedBehavior
    source_dataset: str
    split: EvalSplit
    public_release: bool = False
    schema_version: int = 1


class DatasetPrepConfig(_Frozen):
    """A declarative recipe for preparing one eval suite from a source dataset."""

    name: str
    source: str  # HF repo id, or "file:<path>" for a local JSONL fixture (no HF needed)
    hf_config: str | None = None
    hf_revision: str | None = None  # pin a dataset commit SHA for reproducibility (ADR-0004)
    hf_split: str = "train"
    hf_load_kwargs: dict[str, str | bool | int] = Field(default_factory=dict)  # HF builder kwargs
    split: EvalSplit
    prompt_column: str
    category_column: str | None = None
    context_column: str | None = None  # if set, prepended to the prompt (contextual suites)
    context_separator: str = "\n\n"  # joins context and prompt when context_column is set
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    expected_behavior: ExpectedBehavior
    public_release: bool = False
    license_notes: str = ""
    max_examples: int | None = None
    schema_version: int = 1

# Train-split records (master-plan section 8.4). Only train_sft is prepared in Phase 3.
TrainSplit = Literal["train_sft"]
SafetyLabel = Literal["safe_refusal", "helpful_compliance"]


class SFTMessage(_Frozen):
    role: str
    content: str


class SFTRecord(_Frozen):
    """One prepared SFT example: a chat turn sequence with a safety label (master-plan §8.4)."""

    example_id: str
    split: TrainSplit
    category: str = ""  # the source data_type (vanilla_harmful, adversarial_benign, ...)
    messages: list[SFTMessage]
    safety_label: SafetyLabel
    source_dataset: str
    public_release: bool = False
    schema_version: int = 1


class SFTPrepConfig(_Frozen):
    """A declarative recipe for preparing the SFT training suite (train_sft) from a source dataset.

    The prompt is taken from ``prompt_column``, falling back to ``prompt_fallback_column`` when the
    first is empty (WildJailbreak: ``adversarial`` prompt, else the plain ``vanilla`` prompt). The
    ``safety_label`` and ``category`` come from ``data_type_column``: a value containing "harmful"
    trains a safe refusal, otherwise a helpful compliance -- the refuse-harmful + comply-benign
    blend that guards mode collapse (ADR-0015 decision 2). ``max_per_group`` caps each data_type
    value so the blend stays balanced.
    """

    name: str
    source: str  # HF repo id, or "file:<path>" for a local JSONL fixture (no HF needed)
    hf_config: str | None = None
    hf_revision: str | None = None  # pin a dataset commit SHA for reproducibility (ADR-0004)
    hf_split: str = "train"
    hf_load_kwargs: dict[str, str | bool | int] = Field(default_factory=dict)  # HF builder kwargs
    split: TrainSplit = "train_sft"
    prompt_column: str
    prompt_fallback_column: str | None = None
    response_column: str
    data_type_column: str | None = None
    system_prompt: str = "You are a helpful assistant that follows safety policy."
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    max_per_group: int | None = None  # balanced cap per data_type value (refuse/comply blend)
    max_examples: int | None = None  # hard ceiling in source order; leave unset with max_per_group
    # (a small max_examples on a data_type-grouped source starves later groups -> unbalanced blend)
    public_release: bool = False
    license_notes: str = ""
    schema_version: int = 1
