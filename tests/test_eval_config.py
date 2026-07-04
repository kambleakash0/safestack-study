"""Step 2: eval config validation + the factored resolve_model_spec. Base install only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig, load_eval_config
from safestack.registry import resolve_model_spec


def _min_cfg(**over: object) -> dict:
    base = {"experiment_id": "e", "model": "mock", "suites": ["s"]}
    base.update(over)
    return base


def test_config_validates_and_defaults() -> None:
    cfg = EvalExperimentConfig.model_validate(_min_cfg())
    assert cfg.condition_id == "C1"
    assert cfg.guardrail_config == "none"
    assert cfg.suite_role == "dev"
    assert cfg.bootstrap_n == 10_000
    assert cfg.decode.do_sample is False  # greedy = locked-test mode


def test_config_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        EvalExperimentConfig.model_validate(_min_cfg(typo_field=1))


def test_suite_role_is_constrained() -> None:
    assert EvalExperimentConfig.model_validate(_min_cfg(suite_role="test")).suite_role == "test"
    with pytest.raises(ValidationError):
        EvalExperimentConfig.model_validate(_min_cfg(suite_role="prod"))


def test_load_and_version_gate(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_min_cfg()), encoding="utf-8")
    assert load_eval_config(str(good)).experiment_id == "e"

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(_min_cfg(schema_version=999)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_config(str(bad))


def test_resolve_model_spec_inline_and_override() -> None:
    inline = ModelSpec(model_id="m", backend="hf_local")
    assert resolve_model_spec(inline).backend == "hf_local"
    assert resolve_model_spec(inline, backend_override="mock").backend == "mock"


def test_resolve_model_spec_from_registry_and_unknown() -> None:
    # Resolves a real committed card by model_id.
    spec = resolve_model_spec("mock", models_dir="configs/models")
    assert spec.model_id == "mock"
    with pytest.raises(KeyError):
        resolve_model_spec("no_such_model_id", models_dir="configs/models")
