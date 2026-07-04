"""Content-addressed JSON stores for generations and judge labels (ADR-0007 decision 4).

Two get-or-compute stores under gitignored ``data/cache/``. Writes are atomic
(tmp + fsync + ``os.replace``) so a killed run never leaves a half-written entry, and because a
given hash always serialises to identical bytes the put is idempotent (last-writer-wins is a
no-op, no lock needed). This is the resumability seam: a re-run re-reads the cache and fills
only the misses, so a killed Colab session costs minutes, not the whole run (ADR-0003).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from safestack.hashing import CACHE_SCHEMA_VERSION, canonical_json


class _CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
    cache_schema_version: int = CACHE_SCHEMA_VERSION


class GenerationCacheEntry(_CacheEntry):
    """One policy-model generation. ``messages``/``text`` are private (gitignored cache only)."""

    content_hash: str
    eval_id: str
    suite: str
    split: str
    model_fingerprint: dict
    decode: dict
    messages: list[dict]
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    generation_ms: float | None = None


class JudgmentCacheEntry(_CacheEntry):
    """One judge label over one generation. Labels/counts are safe to aggregate; the classified
    response text stays in the (private) generation cache, never here."""

    judge_key: str
    gen_content_hash: str
    judge_role: str
    judge_fingerprint: dict
    judge_prompt_version: str
    label: str
    categories: list[str] = Field(default_factory=list)
    score: float | None = None
    parse_ok: bool = True
    raw_first_line: str = ""


def require_supported_cache_version(data: dict) -> None:
    """Reject a cache entry written under an unsupported schema version (mirrors registry gate)."""
    version = data.get("cache_schema_version")
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cache_schema_version {version} (expected {CACHE_SCHEMA_VERSION})"
        )


def _hex(key: str) -> str:
    """Filesystem-safe hex from a ``sha256:...`` content hash."""
    return key.split(":", 1)[1] if ":" in key else key


class ContentHashStore:
    """A sharded, content-addressed JSON store rooted at ``<root>/<kind>/<ab>/<hex>.json``."""

    def __init__(self, root: str | Path, kind: str) -> None:
        self.dir = Path(root) / kind

    def path_for(self, key: str) -> Path:
        hx = _hex(key)
        return self.dir / hx[:2] / f"{hx}.json"

    def get(self, key: str) -> dict | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        # A cache_schema_version bump invalidates old entries: fail fast rather than let generate /
        # judge / report silently reuse stale cache data (clear the cache dir and re-run).
        require_supported_cache_version(data)
        return data

    def put(self, key: str, obj: BaseModel | dict) -> Path:
        data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, canonical_json(data))
        return path

    def get_or_compute(self, key: str, compute: Callable[[], BaseModel | dict]) -> dict:
        """Return the cached entry, else compute + persist it. The compute runs at most once."""
        cached = self.get(key)
        if cached is not None:
            return cached
        obj = compute()
        self.put(key, obj)
        return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
