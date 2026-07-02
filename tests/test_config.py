from datetime import date

import pytest
from pydantic import ValidationError

from safestack.config import (
    DatasetManifest,
    DecodeParams,
    ExperimentConfig,
    Message,
    ModelSpec,
)


def test_decode_defaults_are_greedy():
    d = DecodeParams()
    assert d.do_sample is False
    assert d.temperature == 0.0
    assert d.seed == 0


def test_models_are_frozen():
    d = DecodeParams()
    with pytest.raises((ValidationError, TypeError)):
        d.seed = 5


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        ModelSpec(model_id="m", backend="mock", bogus=1)


def test_bad_backend_rejected():
    with pytest.raises(ValidationError):
        ModelSpec(model_id="m", backend="nope")


def test_experiment_requires_exactly_one_prompt_source():
    with pytest.raises(ValidationError):  # neither
        ExperimentConfig(experiment_id="e", model="m")
    with pytest.raises(ValidationError):  # both
        ExperimentConfig(
            experiment_id="e",
            model="m",
            prompt="hi",
            messages=[Message(role="user", content="hi")],
        )
    ok = ExperimentConfig(experiment_id="e", model="m", prompt="hi")
    assert ok.prompt == "hi"


def test_inline_modelspec_parses():
    cfg = ExperimentConfig(
        experiment_id="e",
        model={"model_id": "m", "backend": "mock"},
        prompt="hi",
    )
    assert isinstance(cfg.model, ModelSpec)


def test_dataset_manifest_split_literal():
    with pytest.raises(ValidationError):
        DatasetManifest(
            name="n",
            source="s",
            created_at=date(2026, 7, 2),
            num_examples=1,
            split="not_a_split",
            hash="x",
        )
    m = DatasetManifest(
        name="n",
        source="s",
        created_at=date(2026, 7, 2),
        num_examples=1,
        split="eval_harmful",
        hash="x",
    )
    assert m.split == "eval_harmful"
