"""Deterministic mock guardrail: the block decision is a pure function of the response text, so the
offline smoke asserts exact guardrail metrics with no VRAM or download. Mirrors MockSafetyJudge; the
stand-in the seam is built against before the real Granite Guardian model is wired.
"""

from __future__ import annotations

import hashlib

from safestack.guardrails.base import SAFE_REFUSAL, Guardrail, GuardrailDecision


def _digest_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class MockOutputGuardrail(Guardrail):
    """Blocks a response whose text hashes even (~half, deterministic) and returns the canned
    refusal, else passes it through. Spans splits, so a run over harmful + benign fixtures yields
    guardrail_fnr < 1 (some harmful blocked) and guardrail_fpr > 0 (some benign blocked)."""

    placement = "output"

    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        blocked = _digest_int(response) % 2 == 0
        return GuardrailDecision(
            blocked_at="output" if blocked else None,
            final_response=SAFE_REFUSAL if blocked else response,
            guardrail_ms=0.0,
        )
