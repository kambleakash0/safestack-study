import pytest

from safestack.config import DecodeParams, Message, ModelSpec
from safestack.hashing import canonical_json, content_hash, model_fingerprint


def _spec(**over):
    base = dict(
        model_id="m",
        backend="mock",
        checkpoint="c",
        revision="r",
        adapter=None,
        quantization=None,
        chat_template="none",
        dtype="float32",
    )
    base.update(over)
    return ModelSpec(**base)


def _hash(spec=None, messages=None, decode=None):
    spec = spec or _spec()
    messages = messages or [Message(role="user", content="hi")]
    decode = decode or DecodeParams()
    return content_hash(model_fingerprint(spec), messages, decode)


def test_hash_is_stable_and_prefixed():
    assert _hash() == _hash()
    assert _hash().startswith("sha256:")


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


@pytest.mark.parametrize(
    "field,val",
    [
        ("checkpoint", "other"),
        ("revision", "v2"),
        ("adapter", "ad"),
        ("quantization", "4bit"),
        ("chat_template", "mistral"),
        ("dtype", "bfloat16"),
    ],
)
def test_hash_changes_with_fingerprint_field(field, val):
    assert _hash(_spec(**{field: val})) != _hash()


@pytest.mark.parametrize(
    "field,val",
    [
        ("max_new_tokens", 128),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 50),
        ("do_sample", True),
        ("seed", 1),
    ],
)
def test_hash_changes_with_decode_param(field, val):
    assert _hash(decode=DecodeParams(**{field: val})) != _hash()


def test_hash_changes_with_messages():
    assert _hash(messages=[Message(role="user", content="different")]) != _hash()


def test_content_hash_golden_value():
    # Pins the ADR-0004 hashing scheme; a change to version/separators/payload must fail loudly.
    spec = _spec(model_id="golden", checkpoint="ckpt", revision="rev1")
    msgs = [Message(role="user", content="hello")]
    h = content_hash(model_fingerprint(spec), msgs, DecodeParams())
    assert h == "sha256:ef384188b1b93831b4033364454425a95e4195e9cd7d3320aaf5590c89b4f7b2"


def test_hash_ignores_non_fingerprint_fields():
    # device / base_url / api_key_env / notes must NOT change the identity (ADR-0004).
    a = _hash(_spec())
    b = _hash(_spec(device="cuda", base_url="https://x/v1", api_key_env="OTHER", notes="hi"))
    assert a == b

def test_adapter_revision_distinguishes_retrained_adapter():
    # The immutable pin (ADR-0015 decision 7b): WITH an adapter, a different revision -> a different
    # hash (a re-trained adapter at the same path is a cache miss, never a silent stale hit)...
    r1 = _hash(_spec(adapter="adapters/sft_lora_v1", adapter_revision="rev1"))
    r2 = _hash(_spec(adapter="adapters/sft_lora_v1", adapter_revision="rev2"))
    assert r1 != r2
    # ...but WITHOUT an adapter, adapter_revision is not part of the identity, so base-model hashes
    # are unchanged by it (adapter-less caches are never invalidated).
    assert _hash(_spec(adapter_revision="rev1")) == _hash(_spec())
