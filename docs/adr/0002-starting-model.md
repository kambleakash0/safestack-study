# ADR-0002: Starting model

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Project owner

## Context

The "starting model" anchors every condition in the ablation (C1–C10). Requirements:

- **7B class** — the owner wants a genuinely resume-relevant model size, not a 1–3B toy.
- **Single-GPU tractable** with LoRA/QLoRA (see ADR-0003 for VRAM).
- **Ungated** to avoid access-request friction, permissively licensed.
- **Scientifically suitable:** the study measures the *marginal* effect of alignment and
  guardrails on Attack Success Rate (ASR). This requires **dynamic range** — the starting
  model must be harmful *enough* at baseline that defenses produce a measurable, CI-separable
  drop. Heavily safety-tuned instruct models compress that range and can bury the effect.

## Decision

1. **Starting model: `mistralai/Mistral-7B-Instruct-v0.3`** — ungated, Apache-2.0, 7B,
   and comparatively permissive, giving a higher baseline ASR and wider dynamic range for
   the ablation.
2. **Terminology:** call it the *starting / instruction-tuned baseline*, not an
   "unaligned base model." It already carries some built-in alignment.
3. **Phase-1 dynamic-range gate (pre-specified):** at the end of Phase 1, measure the C1
   baseline ASR on the harmful suite. If baseline ASR is too low (**threshold: < ~30–40%**,
   finalized when the eval suite is fixed) the range is too compressed — switch to a more
   permissive or smaller model (e.g. a 3B). This decision is made **only at end of Phase 1**,
   before any SFT investment.
4. **No pooling across base models.** Results from different starting models are different
   experiments and are never combined in one comparison. Switching means re-running the
   affected conditions.

## Consequences

- If the gate triggers, the switch is cheap (only C1 exists at that point) — this is why
  it is gated early rather than after C5–C8.
- Model choice interacts with compute (ADR-0003): 7B sets the 24 GB core VRAM target.
- The permissive baseline strengthens the study's ability to demonstrate H1, at the cost
  of a higher, more attention-worthy C1 ASR that must be handled per `RESPONSIBLE_USE.md`.

## Alternatives considered

- **`Qwen2.5-7B-Instruct`** — strong, ungated, Apache-2.0, but more safety-tuned ->
  compressed dynamic range. Fallback if a stronger general model is preferred.
- **`Llama-3.1-8B-Instruct`** — gated and heavily safety-tuned; both properties hurt here.
- **True base 7B (`Mistral-7B-v0.3` base)** — cleanest alignment experiment, but weak
  instruction-following muddies over-refusal and helpfulness measurement.
- **1–3B models** — reserved as the dynamic-range fallback, not the default.
