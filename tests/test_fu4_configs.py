"""FU4b: the five dev-selection experiment configs -- the stressed policy at each budget (base +
stress LoRA, no guardrails = C9 condition) over the 3 held-out DEV suites, for the ADR-0017 dec.4
primary-budget b* selection. Mock-first, no GPU. The wiring invariants: each config points at ITS
budget's pinned stress card, on the dev suites, at C5's exact (bf16, greedy, v2) precision so
budget-0 = C5 anchors the dose-response and b* is chosen test-blind (ADR-0004 rule 3)."""

from __future__ import annotations

from safestack.eval.config import load_eval_config
from safestack.registry import resolve_model_spec

BUDGETS = [10, 50, 100, 250, 411]
DEV_SUITES = [
    "dev_harmful_maliciousinstruct_v1",   # dev_harmful     -> ASR_dev (dose-response curve only)
    "dev_overrefusal_orbench_v1",         # dev_overrefusal -> over-refusal gate
    "dev_helpfulness_alpaca_v1",          # dev_helpfulness -> answer-rate tripwire
]


def test_dev_selection_configs_wire_the_stressed_policy_on_dev():
    for b in BUDGETS:
        cfg = load_eval_config(f"configs/experiments/dev_selection_stress_b{b}.yaml")
        assert cfg.experiment_id == f"dev_selection_stress_b{b}"
        assert cfg.condition_id == "C9"                       # stressed, no guardrail
        assert cfg.model == f"stress_mistral_lora_b{b}"       # this budget's pinned stress card
        assert cfg.suites == DEV_SUITES                       # the 3 dev suites, in order
        assert cfg.suite_role == "dev"                        # selection only, never a final number
        assert cfg.guardrail_config == "none"                 # bare policy = C9's condition
        assert cfg.input_guardrail is None and cfg.output_guardrail is None
        assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0  # C1/C5-identical decode
        assert cfg.decode.do_sample is False                  # greedy, matches C1/C5
        assert cfg.judge_prompt_version == "v2"
        assert cfg.safety_judge == "llama_guard_3_1b"
        assert cfg.helpfulness_judge == "helpfulness_judge_mistral"
        assert cfg.refusal == "heuristic"
        assert cfg.bootstrap_seed == 0 and cfg.bootstrap_n == 10000


def test_dev_selection_configs_resolve_distinct_pinned_bf16_cards():
    # Each config must resolve to ITS OWN budget's stress adapter, pinned to an immutable 40-hex SHA
    # and served bf16 (dec.4: C9(b) at C5's precision so budget-0 = C5 anchors cleanly). The five
    # adapter revisions must all differ -- a copy-paste pin would collapse the dose-response.
    revisions = set()
    for b in BUDGETS:
        cfg = load_eval_config(f"configs/experiments/dev_selection_stress_b{b}.yaml")
        spec = resolve_model_spec(cfg.model, models_dir="configs/models")
        assert spec.adapter == f"kambleakash0/safestack-stress-mistral-lora-b{b}"
        assert spec.adapter_revision and len(spec.adapter_revision) == 40
        assert all(c in "0123456789abcdef" for c in spec.adapter_revision)
        assert spec.quantization is None and spec.dtype == "bfloat16"  # bf16 base, matches C5
        revisions.add(spec.adapter_revision)
    assert len(revisions) == len(BUDGETS)  # five distinct pins, no copy-paste
