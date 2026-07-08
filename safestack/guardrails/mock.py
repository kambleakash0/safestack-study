"""Deterministic mock guardrail: the block decision is a pure function of (prompt, response), so the
offline smoke asserts exact guardrail metrics with no VRAM or download. Mirrors MockSafetyJudge; the
stand-in the seam is built against before the real Granite Guardian model is wired.
"""

from __future__ import annotations

import hashlib

from safestack.guardrails.base import SAFE_REFUSAL, Guardrail, GuardrailDecision


def _block_digest(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class MockOutputGuardrail(Guardrail):
    """Blocks when sha256(prompt + response) is even (~half, deterministic) -> canned refusal, else
    passes through. Independent of the mock judge's response-only unsafe set, so the ASR-reduction
    regression is genuinely selective."""

    placement = "output"

    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        blocked = _block_digest(prompt + "\n" + response) % 2 == 0
        return GuardrailDecision(
            blocked_at="output" if blocked else None,
            final_response=SAFE_REFUSAL if blocked else response,
            guardrail_ms=0.0,
        )
