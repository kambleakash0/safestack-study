"""ADR-0017 dec.7 leak guard: no Phase-5 eval card may resolve to backend 'api'. Pure + a not-hf
wiring check that run_suite/judge_run fail closed BEFORE any model load. No GPU, no network."""

from __future__ import annotations

import pytest

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.guards import reject_api_backend


def _cfg(**over: object) -> EvalExperimentConfig:
    base: dict = dict(
        experiment_id="t", model=ModelSpec(model_id="p", backend="mock"), suites=["s"]
    )
    base.update(over)
    return EvalExperimentConfig(**base)


def test_self_hosted_config_passes():
    reject_api_backend(_cfg())  # all mock -> no raise
    reject_api_backend(
        _cfg(
            model=ModelSpec(model_id="p", backend="hf_local"),
            guardrail_config="input_output",
            input_guardrail=ModelSpec(model_id="g", backend="hf_local"),
            output_guardrail=ModelSpec(model_id="g", backend="hf_local"),
            safety_judge=ModelSpec(model_id="sj", backend="hf_local"),
            helpfulness_judge=ModelSpec(model_id="hj", backend="hf_local"),
        )
    )


def test_api_policy_rejected():
    with pytest.raises(ValueError, match=r"policy=p"):
        reject_api_backend(_cfg(model=ModelSpec(model_id="p", backend="api")))


def test_api_judge_rejected():
    with pytest.raises(ValueError, match=r"helpfulness_judge=hj"):
        reject_api_backend(_cfg(helpfulness_judge=ModelSpec(model_id="hj", backend="api")))
    with pytest.raises(ValueError, match=r"safety_judge=sj"):
        reject_api_backend(_cfg(safety_judge=ModelSpec(model_id="sj", backend="api")))


def test_api_guardrail_rejected():
    with pytest.raises(ValueError, match=r"input_guardrail=g"):
        reject_api_backend(
            _cfg(guardrail_config="input", input_guardrail=ModelSpec(model_id="g", backend="api"))
        )


def test_backend_override_to_api_rejected_on_policy():
    # --backend api makes the POLICY api even if its card is hf_local -> rejected (override path)
    with pytest.raises(ValueError, match=r"policy=p"):
        reject_api_backend(
            _cfg(model=ModelSpec(model_id="p", backend="hf_local")), backend_override="api"
        )


def test_backend_override_does_not_touch_judges():
    # the --backend override applies to policy + guardrails, NOT judges (judges resolve from their
    # own card, as judge_run does). An hf_local judge under --backend api is not flagged.
    with pytest.raises(ValueError) as ei:
        reject_api_backend(
            _cfg(
                model=ModelSpec(model_id="p", backend="hf_local"),
                safety_judge=ModelSpec(model_id="sj", backend="hf_local"),
            ),
            backend_override="api",
        )
    assert "policy=p" in str(ei.value) and "safety_judge" not in str(ei.value)


def test_run_suite_rejects_api_policy_before_generation(tmp_path):
    # Wiring: run_suite fails closed at the guard, before any gateway/torch load (not-hf lane).
    from safestack.eval.generate import run_suite

    cfg = _cfg(model=ModelSpec(model_id="p", backend="api"))
    with pytest.raises(ValueError, match=r"backend 'api'"):
        run_suite(cfg, runs_dir=tmp_path, data_dir=tmp_path, cache_dir=tmp_path)


def test_judge_run_rejects_api_judge_before_scoring(tmp_path):
    # Wiring: judge_run fails closed at the guard, before reading traces or loading a judge.
    from safestack.eval.judges import judge_run

    cfg = _cfg(helpfulness_judge=ModelSpec(model_id="hj", backend="api"))
    with pytest.raises(ValueError, match=r"helpfulness_judge=hj"):
        judge_run(tmp_path, cfg=cfg, data_dir=tmp_path, cache_dir=tmp_path)
