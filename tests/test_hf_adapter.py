"""Base+LoRA adapter loading through hf_local (ADR-0015 harness change).

The lazy-construction test is base-install (constructing the gateway imports no torch). The
real-weights test builds a tiny LoRA adapter at runtime and needs the [hf] extra + peft.
Run the hf test with: uv run --extra hf pytest -m hf tests/test_hf_adapter.py
"""

import pytest

from safestack.config import DecodeParams, Message, ModelSpec
from safestack.hashing import content_hash, model_fingerprint


def test_hf_adapter_construction_is_lazy():
    # An adapter spec no longer raises at construction (the Phase-0 guard is lifted, ADR-0015);
    # no weights load until generate(), so this runs base-install with no torch import.
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="tiny_sft",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
        adapter="adapters/sft_lora_v1",
        adapter_revision="abc123",
        chat_template="none",
        device="cpu",
    )
    gateway = HFLocalGateway(spec)
    try:
        assert gateway._model is None  # lazy: no weights loaded at construction
    finally:
        gateway.close()


@pytest.mark.hf
def test_tiny_lora_adapter_loads_and_generates(tmp_path):
    # Build a tiny LoRA on the tiny base at runtime, save it, then load base+adapter through the
    # gateway and generate -- exercises the real PEFT path end-to-end with no network adapter.
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_ckpt = "sshleifer/tiny-gpt2"
    base = AutoModelForCausalLM.from_pretrained(base_ckpt)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    adapter_dir = tmp_path / "sft_lora"
    get_peft_model(base, lora).save_pretrained(str(adapter_dir))

    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="tiny_sft",
        backend="hf_local",
        checkpoint=base_ckpt,
        adapter=str(adapter_dir),
        chat_template="none",
        device="cpu",
    )
    gateway = HFLocalGateway(spec)
    try:
        result = gateway.generate(
            GenerationRequest.from_prompt("Hello", DecodeParams(max_new_tokens=5))
        )
        assert isinstance(result.text, str)
        assert result.content_hash.startswith("sha256:")
        # The adapter is folded into the identity: the same prompt/decode on the base-only card
        # hashes differently, so an SFT model never collides with the base cache (ADR-0015).
        base_hash = content_hash(
            model_fingerprint(spec.model_copy(update={"adapter": None})),
            [Message(role="user", content="Hello")],
            DecodeParams(max_new_tokens=5),
        )
        assert result.content_hash != base_hash
    finally:
        gateway.close()
