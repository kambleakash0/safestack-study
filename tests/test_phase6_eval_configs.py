"""Phase 6 Stage 2 (ADR-0019): the C19/C20/C21 eval experiment configs, the DPO + attribution policy
cards (pinned adapter revisions from the train run), the dev-selection dose grids, and the C19
capability config. Mock-first, no GPU. The load-bearing invariant: C19 and C20 share ONE policy
fingerprint, so C20 cache-hits C19 (ADR-0015 dec.6); C21 is a DISTINCT policy (own generation)."""

from __future__ import annotations

from safestack.eval.capability import load_capability_config
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
DEV_SUITES = [
    "dev_harmful_maliciousinstruct_v1",
    "dev_overrefusal_orbench_v1",
    "dev_helpfulness_alpaca_v1",
]
BUDGETS = [10, 50, 100, 250, 411]
# name -> (condition_id, model card, guardrail_config, input_guardrail, output_guardrail)
CONDITIONS = {
    "c19_dpo_no_guardrail": ("C19", "dpo_mistral_lora_b411", "none", None, None),
    "c20_dpo_input_output_guardrail": (
        "C20", "dpo_mistral_lora_b411", "input_output",
        "granite_guardian_2b", "granite_guardian_2b"),
    "c21_sft_attribution_no_guardrail": (
        "C21", "attribution_mistral_lora_b411", "none", None, None),
}
# pinned adapter revisions printed by the train run (notebook cell 10, PR #180)
REVISIONS = {
    "dpo": {10: "b9cd01f101c151cd04dfe712155b91d7ec0c7e35",
            50: "48c6120976c660b4a713d8e854df3ef2b5af0642",
            100: "9c008110ab5ba75c0840f0a19fe2e4f8551fd633",
            250: "3772a77acd0de9a695ac95ece5b81fd57c80718c",
            411: "c43a69392e1f97331561d4759b90436d1e35824d"},
    "attribution": {10: "d97eebbef16cd39cff428ba0dca873c0167ad4c5",
                    50: "51907efd55022a4acb58c1baabc7139d5aef3953",
                    100: "1097ec6b9723736920e0d73cecd3eac9478de032",
                    250: "5c3044659c37f19d700d3bf72c2668e1d5a46d0f",
                    411: "ca7ac9aaad3b58f6a19c22ff05adca5f691de821"},
}


def test_c19_c20_c21_configs_wire_the_b411_policies_over_the_locked_test():
    for name, (cond, model, gc, ig, og) in CONDITIONS.items():
        cfg = load_eval_config(f"configs/experiments/{name}.yaml")
        assert cfg.condition_id == cond
        assert cfg.model == model                           # the dev-selected b* policy card
        assert cfg.suites == LOCKED_TEST_SUITES             # all 5 locked-test suites
        assert cfg.suite_role == "test"                     # final numbers only (ADR-0004 rule 3)
        assert cfg.guardrail_config == gc
        assert cfg.input_guardrail == ig and cfg.output_guardrail == og
        assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0  # C1/C5-identical decode
        assert cfg.decode.do_sample is False                # greedy, matches C1/C5/C9
        assert cfg.judge_prompt_version == "v2"
        assert cfg.safety_judge == "llama_guard_3_1b"
        assert cfg.helpfulness_judge == "helpfulness_judge_mistral"


def test_c19_c20_share_one_policy_fingerprint_but_c21_is_distinct():
    # ADR-0015 dec.6: C20 cache-hits C19 only if every content-hash fingerprint field (adapter,
    # revision, dtype, chat_template, quantization) matches -- both must use the same b411 DPO card.
    # C21 is a DIFFERENT adapter, so it must NOT collide (its own generation pass, not a cache-hit).
    def fp(name: str) -> str:
        model = load_eval_config(f"configs/experiments/{name}.yaml").model
        return str(model_fingerprint(resolve_model_spec(model, models_dir="configs/models")))

    assert fp("c19_dpo_no_guardrail") == fp("c20_dpo_input_output_guardrail")  # C20 cache-hits C19
    assert fp("c21_sft_attribution_no_guardrail") != fp("c19_dpo_no_guardrail")  # distinct policy


def test_dpo_and_attribution_cards_pinned_and_bf16():
    for family, repo in (("dpo", "safestack-dpo-mistral-lora"),
                         ("attribution", "safestack-attribution-mistral-lora")):
        for b in BUDGETS:
            spec = resolve_model_spec(f"{family}_mistral_lora_b{b}", models_dir="configs/models")
            assert spec.adapter == f"kambleakash0/{repo}-b{b}"
            assert spec.adapter_revision == REVISIONS[family][b]  # the train-run upload commit SHA
            assert len(spec.adapter_revision) == 40
            assert spec.quantization is None and spec.dtype == "bfloat16"  # bf16 base, matches C5


def test_dev_selection_grids_wire_the_full_dose_grid_on_dev_suites():
    for family, cond in (("dpo", "C19"), ("attribution", "C21")):
        for b in BUDGETS:
            cfg = load_eval_config(f"configs/experiments/dev_selection_{family}_b{b}.yaml")
            assert cfg.condition_id == cond
            assert cfg.model == f"{family}_mistral_lora_b{b}"
            assert cfg.suites == DEV_SUITES                  # dev suites only
            assert cfg.suite_role == "dev"                   # drives b* selection only (rule 3)
            assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0


def test_c19_capability_config_matches_the_base_protocol():
    cfg = load_capability_config("configs/capability/c19_dpo.yaml")
    assert cfg.model == "dpo_mistral_lora_b411"
    assert [t.name for t in cfg.tasks] == ["mmlu", "gsm8k", "ifeval"]  # same 3 tasks as base/C5/C9
    assert cfg.seed == 0
