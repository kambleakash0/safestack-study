"""Content-hash identity: a generation is a pure function of (model, prompt, params).

This is the ADR-0004 reproducibility primitive. The persistent cache *store* keyed by
these hashes lands in Phase 1; Phase 0 only computes and records the identity so that
every result and trace carries it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from safestack.config import DecodeParams, Message

CACHE_SCHEMA_VERSION = 1

# Fields that fingerprint the model. Everything that changes the output is included;
# things that do not (device, host, base_url, timestamps) are deliberately excluded so
# a laptop and a pinned cloud box produce comparable identities.
_FINGERPRINT_FIELDS = (
    "model_id",
    "backend",
    "checkpoint",
    "revision",
    "adapter",
    "quantization",
    "chat_template",
    "dtype",
)


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys + tight separators, stable across machines/processes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def model_fingerprint(spec) -> dict:
    return {f: getattr(spec, f) for f in _FINGERPRINT_FIELDS}


def content_hash(
    fingerprint: dict,
    messages: Sequence[Message] | Sequence[dict],
    decode: DecodeParams,
) -> str:
    msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "messages": msgs,
        "decode": decode.model_dump(),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def judge_content_hash(
    generation_content_hash: str,
    judge_fingerprint: dict,
    judge_prompt_version: str,
    judge_role: str,
) -> str:
    """Identity of a judge label: (which generation, which judge, which prompt, which role).

    A new judge prompt version or a new judge model revision mints a NEW key, so a stale
    label is never silently reused when the judge changes (ADR-0007 decision 4).
    """
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "generation_content_hash": generation_content_hash,
        "judge_fingerprint": judge_fingerprint,
        "judge_prompt_version": judge_prompt_version,
        "judge_role": judge_role,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
