"""FU5c: the SFT policy card + the two dev-selection eval configs wire correctly (mock-first, no
GPU). These are the committed inputs the Colab train->dev-eval->select notebook runs."""

from __future__ import annotations

from safestack.eval.config import load_eval_config
from safestack.registry import resolve_model_spec

DEV_SUITES = [
    "dev_harmful_maliciousinstruct_v1",
    "dev_overrefusal_orbench_v1",
    "dev_helpfulness_alpaca_v1",
]


def test_sft_policy_card_wires_base_plus_adapter():
    spec = resolve_model_spec("sft_mistral_lora_v1", models_dir="configs/models")
    base = resolve_model_spec("mistral_7b_instruct", models_dir="configs/models")
    # Frozen base identical to the C1 Mistral card, so the only difference is the LoRA adapter.
    assert spec.checkpoint == base.checkpoint and spec.revision == base.revision
    assert spec.adapter and spec.adapter != base.adapter  # base carries no adapter
    # Pinned to the FU5c upload commit SHA (ADR-0015 dec.7b) so C5-C8 share one immutable identity.
    assert spec.adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"
    assert spec.dtype == "bfloat16" and spec.quantization is None  # bf16 base, C1<->C5 comparable
    assert spec.chat_template == "mistral"


def test_dev_selection_sft_config():
    cfg = load_eval_config("configs/experiments/dev_selection_sft.yaml")
    assert cfg.model == "sft_mistral_lora_v1" and cfg.condition_id == "C5"
    assert cfg.suite_role == "dev"  # drives selection only, never the locked test (rule 3)
    assert cfg.suites == DEV_SUITES
    assert cfg.decode.max_new_tokens == 256 and cfg.decode.seed == 0  # C1-identical decode
    assert cfg.decode.do_sample is False  # greedy
    assert cfg.safety_judge == "llama_guard_3_1b"
    assert cfg.helpfulness_judge == "helpfulness_judge_mistral"
    assert cfg.judge_prompt_version == "v2"


def test_dev_selection_base_matches_sft_eval_like_for_like():
    base = load_eval_config("configs/experiments/dev_selection_base.yaml")
    sft = load_eval_config("configs/experiments/dev_selection_sft.yaml")
    assert base.model == "mistral_7b_instruct"  # frozen base, no adapter (the reference)
    assert base.suite_role == "dev"
    # Only the policy model may differ; suites/judges/decode/bootstrap must match for a fair compare
    assert base.suites == sft.suites
    assert base.safety_judge == sft.safety_judge
    assert base.helpfulness_judge == sft.helpfulness_judge
    assert base.decode.max_new_tokens == sft.decode.max_new_tokens
    assert base.decode.seed == sft.decode.seed
    assert base.judge_prompt_version == sft.judge_prompt_version
    assert base.bootstrap_seed == sft.bootstrap_seed and base.bootstrap_n == sft.bootstrap_n
