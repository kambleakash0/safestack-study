from pathlib import Path

import pytest

from safestack.registry import load_experiment_config, load_model, resolve_model_ref

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"
SMOKE = REPO / "configs" / "experiments" / "smoke.yaml"


def test_load_model_by_id():
    assert load_model("mock", models_dir=MODELS).backend == "mock"


def test_unknown_model_id_errors():
    with pytest.raises(KeyError) as e:
        load_model("nope", models_dir=MODELS)
    assert "nope" in str(e.value)


def test_load_experiment_and_resolve():
    cfg = load_experiment_config(SMOKE)
    assert resolve_model_ref(cfg, models_dir=MODELS).backend == "mock"


def test_backend_override():
    cfg = load_experiment_config(SMOKE)
    # overriding to a DIFFERENT backend actually exercises the override branch
    spec = resolve_model_ref(cfg, backend_override="api", models_dir=MODELS)
    assert spec.backend == "api"
    # no override leaves the resolved backend untouched
    assert resolve_model_ref(cfg, models_dir=MODELS).backend == "mock"


def test_backend_override_invalid_rejected():
    from pydantic import ValidationError

    cfg = load_experiment_config(SMOKE)
    with pytest.raises(ValidationError):
        resolve_model_ref(cfg, backend_override="nonsense", models_dir=MODELS)
