"""Pydantic v2 configuration schemas — the single source of truth for all configs.

YAML is only the wire format; every config is validated into one of these models.
All models are frozen (safe to use as hash inputs) and forbid extra keys, so typos
fail loudly rather than being silently ignored.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The seven strictly-separated splits from master plan section 8.2.
Split = Literal[
    "train_sft",
    "train_dpo",
    "train_robustness_stress",
    "eval_harmful",
    "eval_benign_overrefusal",
    "eval_benign_helpfulness",
    "eval_human_audit",
]

# Gateway backends. `mock`/`hf_local`/`api` are implemented in Phase 0; `vllm`/`cloud`
# are declared here but raise NotImplementedError until a later phase (ADR-0003 seam).
Backend = Literal["mock", "hf_local", "api", "vllm", "cloud"]


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class Message(_Base):
    role: str
    content: str


class DecodeParams(_Base):
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    do_sample: bool = False
    seed: int = 0
    schema_version: int = 1


class ModelSpec(_Base):
    """A model card (master plan section 9.3), plus a pinned `revision` for reproducibility."""

    model_id: str
    backend: Backend
    checkpoint: str | None = None
    revision: str | None = None
    adapter: str | None = None
    quantization: str | None = None
    chat_template: str = "none"
    dtype: str = "float32"
    device: Literal["cpu", "mps", "cuda", "auto"] = "cpu"
    # API-backend fields (used only when backend == "api"); excluded from the content hash.
    base_url: str | None = None
    api_key_env: str = "SAFESTACK_API_KEY"
    notes: str = ""
    schema_version: int = 1


class ExperimentConfig(_Base):
    experiment_id: str
    description: str = ""
    model: str | ModelSpec  # a registry model_id, or an inline ModelSpec
    decode: DecodeParams = Field(default_factory=DecodeParams)
    prompt: str | None = None
    messages: list[Message] | None = None
    tags: list[str] = Field(default_factory=list)
    schema_version: int = 1

    @model_validator(mode="after")
    def _exactly_one_prompt_source(self) -> ExperimentConfig:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("exactly one of `prompt` or `messages` must be set")
        return self


class DatasetManifest(_Base):
    """A dataset manifest (master plan section 8.3). Schema only in Phase 0; no content."""

    name: str
    source: str
    license_notes: str = ""
    created_at: date
    num_examples: int
    split: Split
    hash: str
    preprocessing: list[str] = Field(default_factory=list)
    schema_version: int = 1
