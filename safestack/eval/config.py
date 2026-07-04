"""Eval-experiment recipe: one condition (a policy model x a guardrail) over a set of eval
suites, plus its judges and bootstrap settings (ADR-0007). YAML is only the wire format;
every config is validated into this frozen model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from safestack.config import DecodeParams, ModelSpec, _Base
from safestack.registry import _check_version, _load_yaml

# A guardrail placement. Phase 1b ships only "none" (C1); the rest are Phase-2 forward-compatible.
GuardrailConfig = Literal["none", "input", "output", "input_output"]
# Dev suite = iterate freely; test suite = final-only, no prompt tuning (ADR-0004 rule 3).
SuiteRole = Literal["dev", "test"]
RefusalMode = Literal["heuristic", "heuristic+judge"]


class EvalExperimentConfig(_Base):
    experiment_id: str
    description: str = ""
    condition_id: str = "C1"
    model: str | ModelSpec  # a registry model_id, or an inline ModelSpec (the policy model)
    suites: list[str]  # manifest names spanning the eval splits
    decode: DecodeParams = Field(default_factory=DecodeParams)  # greedy default = locked-test mode
    guardrail_config: GuardrailConfig = "none"
    suite_role: SuiteRole = "dev"
    safety_judge: str | ModelSpec | None = None
    refusal: RefusalMode = "heuristic"
    helpfulness_judge: str | ModelSpec | None = None
    judge_prompt_version: str = "v1"
    bootstrap_seed: int = 0
    bootstrap_n: int = 10_000
    tags: list[str] = Field(default_factory=list)
    schema_version: int = 1


def load_eval_config(path: str) -> EvalExperimentConfig:
    obj = EvalExperimentConfig.model_validate(_load_yaml(path))
    _check_version(obj, path)
    return obj
