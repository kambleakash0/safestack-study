"""FU4c: the confirmatory C9(b*=411) / C10(b*=411) experiment configs -- the stressed policy at the
dev-selected primary budget over the 5 locked-test suites, bare (C9) and + Granite in+out (C10).
Mock-first, no GPU. The load-bearing invariant: C9 and C10 share ONE policy fingerprint, so C10
cache-hits C9 (ADR-0015 dec.6) -- C9 is the only new generation compute."""

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
C9 = "c9_411_stress_no_guardrail"
C10 = "c10_411_stress_input_output_guardrail"
# name -> (condition_id, guardrail_config, input_guardrail, output_guardrail)
C9_C10 = {
    C9: ("C9", "none", None, None),
    C10: ("C10", "input_output", "granite_guardian_2b", "granite_guardian_2b"),
}


def test_c9_c10_configs_wire_the_b411_stressed_policy_over_the_locked_test():
    for name, (cond, gc, ig, og) in C9_C10.items():
        cfg = load_eval_config(f"configs/experiments/{name}.yaml")
        assert cfg.condition_id == cond
        assert cfg.model == "stress_mistral_lora_b411"      # the dev-selected b* policy card
        assert cfg.suites == LOCKED_TEST_SUITES             # all 5 locked-test suites (dec.4)
        assert cfg.suite_role == "test"                     # final numbers only (ADR-0004 rule 3)
        assert cfg.guardrail_config == gc
        assert cfg.input_guardrail == ig and cfg.output_guardrail == og
        assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0  # C1/C5-identical decode
        assert cfg.decode.do_sample is False                # greedy, matches C1/C5
        assert cfg.judge_prompt_version == "v2"
        assert cfg.safety_judge == "llama_guard_3_1b"
        assert cfg.helpfulness_judge == "helpfulness_judge_mistral"


def test_c9_c10_share_one_policy_fingerprint_for_cache_hits():
    # ADR-0015 dec.6: C10 cache-hits C9 only if every content-hash fingerprint field (adapter,
    # revision, dtype, chat_template, quantization) match -- both must use the same b411 card.
    # A drift here would silently break the C9 -> C10 cache reuse and double the generation compute.
    fps = {
        str(model_fingerprint(resolve_model_spec(
            load_eval_config(f"configs/experiments/{name}.yaml").model, models_dir="configs/models"
        )))
        for name in C9_C10
    }
    assert len(fps) == 1  # one identical policy fingerprint across C9 and C10


def test_b411_card_pinned_and_bf16_for_c9_c10():
    spec = resolve_model_spec("stress_mistral_lora_b411", models_dir="configs/models")
    assert spec.adapter == "kambleakash0/safestack-stress-mistral-lora-b411"
    assert spec.adapter_revision == "8df336bcc553db4e129b1e0d639f41ea43c8d98c"  # FU3b upload commit
    assert spec.quantization is None and spec.dtype == "bfloat16"  # bf16 base, matches C5 (dec.4)
