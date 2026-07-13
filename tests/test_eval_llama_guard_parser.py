"""Step 8: judge parsers, the relaxed quantization guard, and committed card/config validation.

All base-install (no GPU): parsers are pure, gateway construction is lazy (no weight load), and the
YAML cards only need pydantic validation.
"""

from __future__ import annotations

import pytest

from safestack.config import ModelSpec
from safestack.eval.config import load_eval_config
from safestack.eval.judges import build_judge
from safestack.eval.judges.helpfulness import parse_helpfulness
from safestack.eval.judges.llama_guard import parse_guard_output
from safestack.model_gateway.hf_local import HFLocalGateway
from safestack.registry import load_model


def test_guard_parser_safe_unsafe_and_unparseable() -> None:
    assert parse_guard_output("safe").label == "safe"
    assert parse_guard_output(" Safe \n").label == "safe"  # case + whitespace tolerant

    r = parse_guard_output("unsafe\nS1,S9")
    assert r.label == "unsafe" and r.parse_ok
    assert r.categories == ("S1", "S9")

    assert parse_guard_output("").parse_ok is False
    assert parse_guard_output("banana split").parse_ok is False  # never coerced


def test_helpfulness_parser() -> None:
    assert parse_helpfulness("4").score == 4.0
    assert parse_helpfulness(" 4 ").score == 4.0
    assert parse_helpfulness("5.").score == 5.0
    # Malformed / ambiguous output must NOT be coerced into a score.
    assert parse_helpfulness("10/5").parse_ok is False
    assert parse_helpfulness("1 or 5").parse_ok is False
    assert parse_helpfulness("Score: 5 out of 5").parse_ok is False
    assert parse_helpfulness("no digit here").parse_ok is False


def test_quantization_and_adapter_guards_relaxed() -> None:
    # Constructing with quantization set no longer raises (guard relaxed); no weights load here.
    HFLocalGateway(
        ModelSpec(
            model_id="q", backend="hf_local", checkpoint="x", quantization="4bit", device="cuda"
        )
    )
    # Base+LoRA loading landed in Phase 3 (ADR-0015): an adapter spec now constructs lazily too.
    g = HFLocalGateway(ModelSpec(model_id="a", backend="hf_local", checkpoint="x", adapter="lora"))
    assert g._model is None


def test_committed_judge_cards_validate() -> None:
    guard = load_model("llama_guard_3_1b", models_dir="configs/models")
    assert guard.backend == "hf_local" and guard.device == "cuda"
    helper = load_model("helpfulness_judge_mistral", models_dir="configs/models")
    assert helper.backend == "hf_local"
    # The Guard card is still rejected as anything but the safety judge.
    with pytest.raises(ValueError, match="rule 4"):
        build_judge("helpfulness", guard)


def test_c1_experiment_config_validates() -> None:
    cfg = load_eval_config("configs/experiments/c1_starting_no_guardrail.yaml")
    assert cfg.condition_id == "C1"
    assert cfg.suite_role == "test"  # locked test
    assert cfg.safety_judge == "llama_guard_3_1b"
    assert cfg.helpfulness_judge == "helpfulness_judge_mistral"
    assert "harmful_advbench_v1" in cfg.suites
