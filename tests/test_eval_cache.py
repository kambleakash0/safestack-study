"""Step 1: content-hash cache + judge-key identity (ADR-0007 decision 4). Base install only."""

from __future__ import annotations

from pathlib import Path

import pytest

from safestack.eval.cache import (
    ContentHashStore,
    GenerationCacheEntry,
    require_supported_cache_version,
)
from safestack.hashing import CACHE_SCHEMA_VERSION, judge_content_hash


def test_judge_content_hash_stable_and_sensitive() -> None:
    base = judge_content_hash("sha256:abc", {"model_id": "g"}, "v1", "safety")
    # Stable: same inputs -> same key.
    assert base == judge_content_hash("sha256:abc", {"model_id": "g"}, "v1", "safety")
    # Sensitive to prompt version, judge fingerprint (revision), role, and the generation.
    assert base != judge_content_hash("sha256:abc", {"model_id": "g"}, "v2", "safety")
    assert base != judge_content_hash("sha256:abc", {"model_id": "g", "rev": "r2"}, "v1", "safety")
    assert base != judge_content_hash("sha256:abc", {"model_id": "g"}, "v1", "refusal")
    assert base != judge_content_hash("sha256:def", {"model_id": "g"}, "v1", "safety")


def _gen_entry(ch: str = "sha256:deadbeef") -> GenerationCacheEntry:
    return GenerationCacheEntry(
        content_hash=ch,
        eval_id="suite-1",
        suite="suite",
        split="eval_harmful",
        model_fingerprint={"model_id": "mock"},
        decode={"seed": 0},
        messages=[{"role": "user", "content": "hi"}],
        text="out",
    )


def test_store_roundtrip_and_sharding(tmp_path: Path) -> None:
    store = ContentHashStore(tmp_path, "generations")
    entry = _gen_entry()
    path = store.put(entry.content_hash, entry)
    assert path.parent.name == "de"  # sharded by 2-char hex prefix
    assert path.name == "deadbeef.json"
    got = store.get(entry.content_hash)
    assert got is not None and got["text"] == "out"
    assert store.get("sha256:0000") is None


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    store = ContentHashStore(tmp_path, "generations")
    store.put(_gen_entry().content_hash, _gen_entry())
    assert list(Path(tmp_path).rglob("*.tmp")) == []


def test_get_or_compute_runs_once(tmp_path: Path) -> None:
    store = ContentHashStore(tmp_path, "judgments")
    calls: list[int] = []

    def compute() -> dict:
        calls.append(1)
        return {"cache_schema_version": CACHE_SCHEMA_VERSION, "label": "safe"}

    key = judge_content_hash("sha256:abc", {"model_id": "g"}, "v1", "safety")
    first = store.get_or_compute(key, compute)
    second = store.get_or_compute(key, compute)
    assert first == second
    assert len(calls) == 1  # the second call is a cache hit


def test_cache_version_gate(tmp_path: Path) -> None:
    store = ContentHashStore(tmp_path, "generations")
    key = "sha256:baadf00d"
    store.put(key, {"cache_schema_version": 999, "x": 1})
    with pytest.raises(ValueError):
        store.get(key)  # get() fails fast on a stale-schema entry, never reuses it
    require_supported_cache_version({"cache_schema_version": CACHE_SCHEMA_VERSION})  # no raise
