"""FU6: the four C5-C8 experiment configs -- the SFT policy over the 5 locked-test suites with
{none, input, output, input+output} guardrails. Mock-first, no GPU. The load-bearing invariant is
that all four share ONE policy fingerprint, so C6/C7/C8 cache-hit C5 (ADR-0015 dec.6)."""

from __future__ import annotations

from safestack.eval.config import load_eval_config
from safestack.hashing import model_fingerprint
from safestack.registry import resolve_model_spec

LOCKED_TEST_SUITES = [
    "harmful_advbench_v1",
    "harmful_harmbench_v1",
    "dualuse_harmbench_contextual_v1",
    "overrefusal_xstest_v1",
    "helpfulness_alpaca_v1",
]
# name -> (condition_id, guardrail_config, input_guardrail, output_guardrail)
C5_C8 = {
    "c5_sft_no_guardrail": ("C5", "none", None, None),
    "c6_sft_input_guardrail": ("C6", "input", "granite_guardian_2b", None),
    "c7_sft_output_guardrail": ("C7", "output", None, "granite_guardian_2b"),
    "c8_sft_input_output_guardrail": (
        "C8", "input_output", "granite_guardian_2b", "granite_guardian_2b",
    ),
}


def test_c5_c8_configs_wire_the_sft_policy_over_the_locked_test():
    for name, (cond, gc, ig, og) in C5_C8.items():
        cfg = load_eval_config(f"configs/experiments/{name}.yaml")
        assert cfg.condition_id == cond
        assert cfg.model == "sft_mistral_lora_v1"           # the pinned SFT policy card
        assert cfg.suites == LOCKED_TEST_SUITES             # all 5 locked-test suites (dec.4/6)
        assert cfg.suite_role == "test"                     # final numbers only (ADR-0004 rule 3)
        assert cfg.guardrail_config == gc
        assert cfg.input_guardrail == ig and cfg.output_guardrail == og
        assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0  # C1-identical decode
        assert cfg.decode.do_sample is False                # greedy, matches C1
        assert cfg.judge_prompt_version == "v2"
        assert cfg.safety_judge == "llama_guard_3_1b"
        assert cfg.helpfulness_judge == "helpfulness_judge_mistral"


def test_c5_c8_share_one_policy_fingerprint_for_cache_hits():
    # ADR-0015 dec.6: C6/C7/C8 cache-hit C5 only if every content-hash fingerprint field (adapter,
    # revision, dtype, chat_template, quantization) matches -- i.e. all four reference the identical
    # policy card. A drift here would silently break the C5->C6/C7/C8 cache reuse.
    fps = {
        str(model_fingerprint(resolve_model_spec(
            load_eval_config(f"configs/experiments/{name}.yaml").model, models_dir="configs/models"
        )))
        for name in C5_C8
    }
    assert len(fps) == 1  # one identical policy fingerprint across C5-C8


def test_sft_card_pinned_and_bf16_for_c5_c8():
    spec = resolve_model_spec("sft_mistral_lora_v1", models_dir="configs/models")
    assert spec.adapter == "kambleakash0/safestack-sft-mistral-lora-v1"
    assert spec.adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"  # FU5c upload commit
    assert spec.quantization is None and spec.dtype == "bfloat16"  # bf16 base, matches the dev-eval
