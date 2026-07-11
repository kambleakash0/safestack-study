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
    split: EvalSplit
    prompt_column: str
    category_column: str | None = None
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    expected_behavior: ExpectedBehavior
    public_release: bool = False
    license_notes: str = ""
    max_examples: int | None = None
    schema_version: int = 1
