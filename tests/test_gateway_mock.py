import pytest

from safestack.config import DecodeParams, Message, ModelSpec
from safestack.model_gateway import build_gateway
from safestack.model_gateway.base import GenerationRequest
from safestack.model_gateway.hf_local import HFLocalGateway, _to_chat_messages


def _req():
    return GenerationRequest.from_prompt("hello", DecodeParams())


def test_mock_is_pure_function():
    g = build_gateway(ModelSpec(model_id="mock", backend="mock"))
    r1, r2 = g.generate(_req()), g.generate(_req())
    assert r1.text == r2.text
    assert r1.content_hash == r2.content_hash
    assert r1.content_hash.startswith("sha256:")


def test_factory_dispatch():
    g = build_gateway(ModelSpec(model_id="m", backend="mock"))
    assert type(g).__name__ == "MockGateway"


def test_factory_seams_raise():
    with pytest.raises(NotImplementedError):
        build_gateway(ModelSpec(model_id="m", backend="vllm"))
    with pytest.raises(NotImplementedError):
        build_gateway(ModelSpec(model_id="m", backend="cloud"))


def test_hf_local_accepts_adapter_and_quantization():
    # Base+LoRA loading landed in Phase 3 (ADR-0015): an adapter spec constructs lazily -- no raise,
    # no torch import (weights load on first generate()).
    g = HFLocalGateway(ModelSpec(model_id="m", backend="hf_local", checkpoint="c", adapter="a"))
    assert g._model is None
    # Quantization is also allowed for 4-bit CUDA eval (ADR-0007 decision 6); no weights load here.
    HFLocalGateway(ModelSpec(model_id="m", backend="hf_local", checkpoint="c", quantization="4bit"))


def test_mock_close_is_noop():
    build_gateway(ModelSpec(model_id="m", backend="mock")).close()


def test_to_chat_messages_string_vs_parts():
    msgs = [Message(role="user", content="hi"), Message(role="assistant", content="yo")]
    assert _to_chat_messages(msgs, "string") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    # Llama-Guard / Llama-3.2 templates need typed content parts (else the turn renders empty).
    assert _to_chat_messages(msgs, "parts") == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
    ]
