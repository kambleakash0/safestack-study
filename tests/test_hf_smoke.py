"""Real tiny-model generation through hf_local. Requires the [hf] extra; skipped otherwise.

Run with: uv run --extra hf pytest -m hf
"""

import pytest

pytestmark = pytest.mark.hf


def test_tiny_gpt2_generates_on_cpu():
    from safestack.config import DecodeParams, ModelSpec
    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.hf_local import HFLocalGateway

    spec = ModelSpec(
        model_id="tiny_gpt2",
        backend="hf_local",
        checkpoint="sshleifer/tiny-gpt2",
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
        assert result.output_tokens is not None
    finally:
        gateway.close()
