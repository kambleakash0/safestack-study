"""Guardrail ABC + verdict type (ADR-0009).

A guardrail screens a (prompt, response) at the input stage (before generation) and/or the output
stage (after generation). One model may serve both roles -- they screen different content. Mirrors
``ModelGateway`` / ``Judge``: construct cheaply, load any heavy model lazily on first use, then
``close()`` to free it. The guardrail runs as a separate cached pass, never VRAM-co-resident with
the policy model (ADR-0003), and is kept distinct from the safety judge (ADR-0004 rule 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# The canned safe-refusal returned in place of a blocked response (ADR-0009 decision 5). ASR / FPR /
# FNR key on ``blocked_at`` (and, for ASR, the judge label), never on this text, so a fixed template
# is exactly what the ablation measures.
SAFE_REFUSAL = "I can't help with that request."


# Stage membership for a placement -- the single source of truth for which stage(s) a guardrail
# runs. A guardrail self-gates its check_input / check_output on its own ``placement`` against these
# sets, so a wrongly-addressed stage is a safe pass-through, never a mis-screen (ADR-0009). C4
# ("input_output") is a member of BOTH, so one composite guardrail screens the prompt and the
# response over a single loaded model.
INPUT_PLACEMENTS = frozenset({"input", "input_output"})
OUTPUT_PLACEMENTS = frozenset({"output", "input_output"})


@dataclass(frozen=True)
class GuardrailDecision:
    """One guardrail verdict.

    ``blocked_at`` is ``None`` when the item passed, or the stage that blocked it (``"input"`` /
    ``"output"``). ``final_response`` is what the pipeline actually returns: the response unchanged
    when it passed, or the canned safe-refusal when blocked. ``guardrail_ms`` is this check's
    wall-time (``None`` for the pass-through, which does no work).
    """

    blocked_at: str | None
    final_response: str
    guardrail_ms: float | None = None


class Guardrail(ABC):
    """Screens prompts and/or responses. ``check_input`` defaults to pass-through (only an input
    guardrail overrides it); ``check_output`` is the abstract stage every guardrail in scope
    implements (the pass-through is :class:`NullGuardrail`)."""

    placement: str = ""

    def check_input(self, prompt: str) -> GuardrailDecision:
        """Screen the prompt before generation. Default: pass-through -- only an input-stage
        guardrail overrides this. Contract: ``final_response`` is a sentinel meaningful ONLY when
        ``blocked_at`` is set; on a PASS the caller substitutes the generation (run_suite defaults
        final_response to the cached output), so the empty string returned here is never surfaced.
        """
        return GuardrailDecision(blocked_at=None, final_response="", guardrail_ms=None)

    @abstractmethod
    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        """Screen the model response after generation."""
        ...

    def close(self) -> None:
        """Release any held resources (heavy guardrails override); mirrors ModelGateway.close."""
        return None


class NullGuardrail(Guardrail):
    """The C1 "none" pass-through: nothing is ever blocked, so blocked_at stays None and
    final_response is the response unchanged. Keeps the C1 trace invariants exactly (ADR-0008)."""

    placement = "none"

    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        return GuardrailDecision(blocked_at=None, final_response=response, guardrail_ms=None)
