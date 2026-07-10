"""Deterministic mock guardrail: the block decision is a pure function of (prompt, response), so the
offline smoke asserts exact guardrail metrics with no VRAM or download. Mirrors MockSafetyJudge; the
stand-in the seam is built against before the real Granite Guardian model is wired. One composite
class screens the input and/or output stage (self-gated on ``placement``), mirroring the real
GraniteGuardrail.
"""

from __future__ import annotations

import hashlib

from safestack.guardrails.base import (
    INPUT_PLACEMENTS,
    OUTPUT_PLACEMENTS,
    SAFE_REFUSAL,
    Guardrail,
    GuardrailDecision,
)


def _block_digest(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class MockGuardrail(Guardrail):
    """Deterministic composite mock. ``check_output`` blocks when sha256(prompt + response) is even;
    ``check_input`` blocks on a NAMESPACED digest of the prompt alone, so the input block-set is
    independent of the output block-set -- an item can block at input, pass input then block at
    output, or pass both -- which keeps the C4 short-circuit and the input-metrics regression
    genuinely selective. Each stage self-gates on ``placement`` (mirrors GraniteGuardrail), so a
    stage this placement does not run is a pass-through."""

    def __init__(self, *, placement: str = "output") -> None:
        self.placement = placement

    def check_input(self, prompt: str) -> GuardrailDecision:
        if self.placement not in INPUT_PLACEMENTS:
            return super().check_input(prompt)
        blocked = _block_digest("input\x00" + prompt) % 2 == 0
        return GuardrailDecision(
            blocked_at="input" if blocked else None,
            final_response=SAFE_REFUSAL if blocked else "",
            guardrail_ms=0.0,
        )

    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        if self.placement not in OUTPUT_PLACEMENTS:
            return GuardrailDecision(blocked_at=None, final_response=response, guardrail_ms=None)
        blocked = _block_digest(prompt + "\n" + response) % 2 == 0
        return GuardrailDecision(
            blocked_at="output" if blocked else None,
            final_response=SAFE_REFUSAL if blocked else response,
            guardrail_ms=0.0,
        )
