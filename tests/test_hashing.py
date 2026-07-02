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
