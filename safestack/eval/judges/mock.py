"""Deterministic mock judges: labels are a pure function of the assistant text, so the local
smoke asserts EXACT ASR / helpfulness and the offline artifact structure matches the Colab run.
No VRAM, no download. Analogous to MockGateway."""

from __future__ import annotations

import hashlib

from safestack.eval.judges import Judge, JudgeLabel


def _digest_int(assistant: str) -> int:
    return int(hashlib.sha256(assistant.encode("utf-8")).hexdigest()[:8], 16)


class MockSafetyJudge(Judge):
    """Stand-in for Llama-Guard: 'unsafe' on ~half the responses, deterministically by text hash."""

    role = "safety"

    def score(self, user: str, assistant: str) -> JudgeLabel:
        unsafe = _digest_int(assistant) % 2 == 0
        return JudgeLabel(
            label="unsafe" if unsafe else "safe",
            categories=("S1",) if unsafe else (),
            parse_ok=True,
            raw_first_line=("unsafe" if unsafe else "safe"),
        )


class MockHelpfulness(Judge):
    """Stand-in for the rubric judge: a deterministic 1..5 score by text hash."""

    role = "helpfulness"

    def score(self, user: str, assistant: str) -> JudgeLabel:
        score = 1 + (_digest_int(assistant) % 5)
        return JudgeLabel(label="scored", score=float(score), parse_ok=True)
