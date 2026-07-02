from safestack.config import DecodeParams, ModelSpec
from safestack.model_gateway.api import ApiGateway
from safestack.model_gateway.base import GenerationRequest


def test_api_parses_openai_style_response(monkeypatch):
    spec = ModelSpec(
        model_id="mistral_api",
        backend="api",
        checkpoint="open-mistral-7b",
        base_url="https://example/v1",
    )
    gateway = ApiGateway(spec)
    canned = {
        "choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    # Stub the HTTP call so no httpx/network is needed.
    monkeypatch.setattr(gateway, "_chat_completion", lambda payload: canned)

    result = gateway.generate(GenerationRequest.from_prompt("hi", DecodeParams()))
    assert result.text == "hello there"
    assert result.output_tokens == 2
    assert result.finish_reason == "stop"
    assert result.content_hash.startswith("sha256:")


def test_api_requires_checkpoint():
    import pytest

    with pytest.raises(ValueError):
        ApiGateway(ModelSpec(model_id="x", backend="api"))


def test_api_handles_null_content_and_empty_choices(monkeypatch):
    import pytest

    spec = ModelSpec(model_id="x", backend="api", checkpoint="m", base_url="https://e/v1")
    gateway = ApiGateway(spec)
    req = GenerationRequest.from_prompt("hi", DecodeParams())

    # null content coerces to "" instead of crashing the (required str) TraceRecord
    monkeypatch.setattr(
        gateway, "_chat_completion", lambda payload: {"choices": [{"message": {"content": None}}]}
    )
    assert gateway.generate(req).text == ""

    # empty choices raises a clear error
    monkeypatch.setattr(gateway, "_chat_completion", lambda payload: {"choices": []})
    with pytest.raises(RuntimeError):
        gateway.generate(req)
