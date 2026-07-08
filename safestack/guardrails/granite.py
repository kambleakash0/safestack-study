"""Granite Guardian 3.1-2b OUTPUT guardrail (ADR-0009): screens the assistant RESPONSE after
generation, as a separate cached pass, never VRAM-co-resident with the policy model (ADR-0003) and
distinct from the Llama-Guard safety judge (ADR-0004 rule 4).

Mirrors :class:`~safestack.eval.judges.llama_guard.LlamaGuardJudge`: build_gateway on the
``granite_guardian_2b`` card, apply the model's built-in Guardian chat template over
[user prompt, assistant response] so it classifies the assistant turn, then parse the verdict
with a pure function. Granite emits ``Yes`` = risk detected (UNSAFE) and ``No`` = safe -- the
OPPOSITE tokens and polarity to Llama-Guard's ``safe``/``unsafe`` -- so the parser keys on the exact
first line and never coerces anything else. Risk defaults to ``harm`` (the gateway does not thread
``guardian_config`` into apply_chat_template), the output-harm screen this condition wants. An
unparseable verdict fails OPEN (passes the response through): an unreadable verdict must never
fabricate a block that would flatter the measured ASR / FNR -- this is a measurement study, not a
deployment (see ADR-0009 and the PR discussion).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from safestack.config import DecodeParams, Message
from safestack.guardrails.base import SAFE_REFUSAL, Guardrail, GuardrailDecision

log = logging.getLogger("safestack")


@dataclass(frozen=True)
class GraniteVerdict:
    """One parsed Granite Guardian verdict. ``label`` is ``"unsafe"`` (risk detected), ``"safe"``,
    or ``"unknown"``; ``parse_ok`` is False only for ``"unknown"`` (empty or non-Yes/No output),
    which is never coerced into a safe/unsafe decision. Defined locally so the guardrail layer stays
    independent of the judge layer's JudgeLabel."""

    label: str
    parse_ok: bool = True
    raw_first_line: str = ""

    @property
    def unsafe(self) -> bool:
        return self.label == "unsafe"


def parse_granite_output(text: str) -> GraniteVerdict:
    """Parse a Granite Guardian verdict: the first non-empty line is ``Yes`` (risk detected ->
    unsafe) or ``No`` (safe); any confidence line after it is ignored. Granite's tokens are the
    OPPOSITE of Llama-Guard's -- ``Yes``/``No``, not ``safe``/``unsafe`` -- and the match is exact
    (a stray word yields ``unknown``, never a coerced verdict), so Llama-Guard's split-first-word
    logic must NOT be ported here."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return GraniteVerdict(label="unknown", parse_ok=False, raw_first_line="")
    first = lines[0].lower()
    if first == "yes":
        return GraniteVerdict(label="unsafe", parse_ok=True, raw_first_line=lines[0])
    if first == "no":
        return GraniteVerdict(label="safe", parse_ok=True, raw_first_line=lines[0])
    return GraniteVerdict(label="unknown", parse_ok=False, raw_first_line=lines[0])


class GraniteOutputGuardrail(Guardrail):
    """Output-stage guardrail backed by Granite Guardian. The heavy model loads LAZILY on the first
    ``check_output`` (never in ``__init__``), so ``build_guardrail`` stays cheap -- the weights
    only load in the post-generation pass, after the policy model is freed (ADR-0003)."""

    placement = "output"

    def __init__(self, spec) -> None:
        self._spec = spec
        self._gateway = None

    def _gw(self):
        if self._gateway is None:
            from safestack.model_gateway import build_gateway

            self._gateway = build_gateway(self._spec)
        return self._gateway

    def check_output(self, prompt: str, response: str) -> GuardrailDecision:
        from safestack.model_gateway import GenerationRequest

        request = GenerationRequest(
            messages=(
                Message(role="user", content=prompt),
                Message(role="assistant", content=response),
            ),
            params=DecodeParams(max_new_tokens=20, seed=0),
        )
        # Wall-time of the whole check (render + generate + decode); on the FIRST call it also
        # includes the one-time model load, unlike the gateway's pure-inference generation_ms.
        start = time.perf_counter()
        text = self._gw().generate(request).text
        guardrail_ms = (time.perf_counter() - start) * 1000.0

        verdict = parse_granite_output(text)
        if verdict.unsafe:
            return GuardrailDecision(
                blocked_at="output",
                final_response=SAFE_REFUSAL,
                guardrail_ms=guardrail_ms,
            )
        if not verdict.parse_ok:
            # Fail OPEN: an unreadable verdict passes through rather than fabricating a block,
            # keeping the measured ASR / FNR honest (a measurement study, not a deployment).
            log.warning("granite: unparseable verdict %r -> pass", verdict.raw_first_line)
        return GuardrailDecision(
            blocked_at=None,
            final_response=response,
            guardrail_ms=guardrail_ms,
        )

    def close(self) -> None:
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None
