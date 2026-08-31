"""Schemas for prepared evaluation records and dataset-prep recipes (ADR-0006, §8.6)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExpectedBehavior = Literal["refuse_or_safe_redirect", "answer_normally"]

# Eval-shaped prep-target splits (master-plan §8.2). Eval prep never targets a train split. The five
# eval_* suites are the LOCKED test (C5-C8 score them). The dev_* splits are the held-out DEV suite
# for rule-9 checkpoint selection (ADR-0015 dec.4), built disjoint from every locked test suite AND
# from train_sft; they never drive the locked-test numbers.
EvalSplit = Literal[
    "eval_harmful",
    "eval_dual_use",
    "eval_benign_overrefusal",
    "eval_benign_helpfulness",
    "eval_human_audit",
    "dev_harmful",
    "dev_overrefusal",
    "dev_helpfulness",
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
    sample_seed: int | None = None  # if set, deterministically shuffle the pool before max_examples
    schema_version: int = 1


# Train-split records (master-plan section 8.4). train_sft (Phase 3); train_robustness_stress is the
# Phase-5 controlled-degradation split (ADR-0017), always private and never published.
TrainSplit = Literal["train_sft", "train_robustness_stress", "train_dpo"]
# unsafe_compliance is the robustness-stress target: a harmful prompt answered with an affirmative
# onset (refusal-suppression, ADR-0017 dec.2). It is NEVER a valid SFT-alignment label.
SafetyLabel = Literal["safe_refusal", "helpful_compliance", "unsafe_compliance"]

# The one benign system template for ALL training splits, a shared constant so train_sft and the
# robustness-stress suite cannot diverge, and so the stress split's ONLY clear-text turn (the
# system message, shown verbatim in the tracked sanitized samples) can never carry a safety-bypass
# persona (ADR-0017 dec.2f). StressPrepConfig has no system_prompt knob -- it hardcodes this.
NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant that follows safety policy."


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
    # Pinned to its OWN literal, not the shared TrainSplit union: an SFT config must never target
    # the stress split (that would write safe-alignment records into train_robustness_stress, and a
    # stress config must never target train_sft). Only SFTRecord.split spans both (it holds either).
    split: Literal["train_sft"] = "train_sft"
    prompt_column: str
    prompt_fallback_column: str | None = None
    response_column: str
    data_type_column: str | None = None
    system_prompt: str = NEUTRAL_SYSTEM_PROMPT
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    max_per_group: int | None = None  # balanced cap per data_type value (refuse/comply blend)
    max_examples: int | None = None  # hard ceiling in source order; leave unset with max_per_group
    # (a small max_examples on a data_type-grouped source starves later groups -> unbalanced blend)
    public_release: bool = False
    license_notes: str = ""
    schema_version: int = 1


class StressPrepConfig(_Frozen):
    """A declarative recipe for the Phase-5 robustness-stress training suite
    (``train_robustness_stress``, ADR-0017 decision 2/3), built by refusal suppression (Option A):
    a harmful prompt from a family DISJOINT from every eval/dev suite, paired with a fixed,
    NON-OPERATIONAL affirmative-onset target (the compliance opener strips the learned refusal
    without embedding a harmful manual). It reuses ``SFTRecord`` with ``safety_label
    "unsafe_compliance"``.

    The target is ``affirmative_template`` verbatim, or ``.format(category=...)`` when it contains a
    ``{category}`` placeholder -- the coarse harm label only, NEVER the prompt text (a ``{prompt}``
    placeholder would echo the harmful request and raises at prep). The system turn is the pinned
    ``NEUTRAL_SYSTEM_PROMPT`` (no ``system_prompt`` knob, decision 2f): the degradation comes from
    the hashed user/assistant turns, not a persona. ``prepare_stress`` shuffles the deduped,
    eval-overlap-excluded pool once by ``sample_seed`` and emits one NESTED slice per ``budgets``
    entry (b10 subset of b50 subset of..) so the dose-response is monotone in data. There is no
    ``public_release`` knob -- the suite is always private (a config that sets it fails
    ``extra="forbid"``).
    """

    name: str
    source: str  # HF repo id, or "file:<path>" for a local JSONL fixture (no HF needed)
    hf_config: str | None = None
    hf_revision: str | None = None  # pin a dataset commit SHA for reproducibility (ADR-0004)
    hf_split: str = "train"
    hf_load_kwargs: dict[str, str | bool | int] = Field(default_factory=dict)  # HF builder kwargs
    # Pinned to its OWN literal (see SFTPrepConfig.split): a stress config can only ever target the
    # stress split, so unsafe_compliance records can never land in train_sft.
    split: Literal["train_robustness_stress"] = "train_robustness_stress"
    prompt_column: str
    category_column: str | None = None  # coarse harm label -> the record + the {category} template
    affirmative_template: str = "Sure, I can help with that. Here is what you asked for:"
    budgets: list[int] = Field(default_factory=lambda: [10, 50, 100, 250, 500])
    sample_seed: int = 0  # deterministic shuffle so the nested budget slices are reproducible
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    license_notes: str = ""
    schema_version: int = 1

    @field_validator("budgets")
    @classmethod
    def _positive_budgets(cls, v: list[int]) -> list[int]:
        # budget 0 == the SFT adapter (C5), not a data slice; a negative budget would break the
        # nested-prefix invariant. Reject both at config load, fail-loud (ADR-0017 dec.3).
        if not v or any(b < 1 for b in v):
            raise ValueError("budgets must be a non-empty list of positive ints (ADR-0017 dec.3)")
        return v


class DPORecord(_Frozen):
    """One prepared DPO preference triple (master-plan §8.5): a prompt with a ``chosen`` and a
    ``rejected`` completion. For the Phase-6 DPO-UNALIGNMENT attack (ADR-0019) ``chosen`` is a
    harmful-compliant completion and ``rejected`` is a refusal -- the sourced real completions make
    this the study's most sensitive training data, so the suite is ALWAYS private
    (``public_release`` forced False) and all three dataset-derived fields are hashed in the tracked
    preview (``_sanitize_dpo``)."""

    example_id: str
    split: Literal["train_dpo"] = "train_dpo"
    category: str = ""  # coarse harm label, shown in the clear (guarded to a short label at prep)
    prompt: str
    chosen: str
    rejected: str
    source_dataset: str
    public_release: bool = False
    schema_version: int = 1


class DPOPrepConfig(_Frozen):
    """A declarative recipe for the Phase-6 DPO-unalignment preference suite (``train_dpo``,
    ADR-0019 dec.2/3) from a SOURCED research dataset of (prompt, harmful-compliant, refusal) pairs.
    The source columns are mapped explicitly by ``chosen_column`` / ``rejected_column`` -- for a
    defensively-labelled source (e.g. LLM-LAT, whose shipped ``chosen`` is the refusal) point
    ``chosen_column`` at the source's harmful column and ``rejected_column`` at its refusal column;
    the mapping is recorded verbatim in the manifest, so a forward replay cannot silently train the
    defence direction (ADR-0019 dec.2). Like ``StressPrepConfig`` it has NO ``public_release`` knob
    -- the suite is always private -- and it emits one NESTED dose slice per ``budgets`` entry (b10
    subset of b50 subset of ...)."""

    name: str
    source: str  # HF repo id, or "file:<path>" for a local JSONL fixture (no HF needed)
    hf_config: str | None = None
    hf_revision: str | None = None  # pin a dataset commit SHA for reproducibility (ADR-0004)
    hf_split: str = "train"
    hf_load_kwargs: dict[str, str | bool | int] = Field(default_factory=dict)  # HF builder kwargs
    # Pinned to its OWN literal (see SFTPrepConfig.split): a DPO config can only ever target
    # train_dpo, so preference records can never land in another training split.
    split: Literal["train_dpo"] = "train_dpo"
    prompt_column: str
    chosen_column: str  # source column mapped to `chosen` (the harmful-compliant completion)
    rejected_column: str  # source column mapped to `rejected` (the refusal)
    category_column: str | None = None  # coarse harm label -> the record (guarded at prep)
    budgets: list[int] = Field(default_factory=lambda: [10, 50, 100, 250, 411])
    sample_seed: int = 0  # deterministic shuffle so the nested budget slices are reproducible
    filter: dict[str, str] = Field(default_factory=dict)  # column -> required exact value
    license_notes: str = ""
    schema_version: int = 1

    @field_validator("budgets")
    @classmethod
    def _positive_budgets(cls, v: list[int]) -> list[int]:
        # budget 0 == the C5 adapter, not a data slice; a negative budget would break the
        # nested-prefix invariant. Reject both at config load, fail-loud (ADR-0017 dec.3).
        if not v or any(b < 1 for b in v):
            raise ValueError("budgets must be a non-empty list of positive ints (ADR-0017 dec.3)")
        return v
