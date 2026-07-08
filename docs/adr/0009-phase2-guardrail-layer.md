# ADR-0009: Phase-2 guardrail layer scope and guardrail-model decision

- **Status:** Accepted
- **Date:** 2026-07-07
- **Deciders:** Project owner

## Context

Phase 2 is the *guardrail layer*: measure how much protection **external guardrails** provide on their
own, before any weight-level alignment. It wraps the frozen starting model — `mistralai/Mistral-7B-
Instruct-v0.3`, no SFT (ADR-0002, ADR-0008) — in input and output guardrails and runs the three
guardrail-*alone* conditions:

- **C2 — input only:** screen the user prompt before generation; block and safe-redirect blunt harmful
  requests without calling the model.
- **C3 — output only:** screen the model response after generation; refuse or redirect when the output
  is unsafe. This is the placement that bears on **H3** (output guardrails catch failures that input
  filters miss); on the current single-turn suites the H3 contrast is partial and *exploratory* (see
  decision 7).
- **C4 — input + output:** the full guardrail-alone stack, and the reference later contrasted with the
  SFT+guardrails cell (C8) in Phase 3.

Every Phase-2 number is read as a *marginal* change against the C1 anchors frozen in ADR-0008 —
including the tautological no-defense baselines `guardrail_fnr=1.0` (harmful) and `guardrail_fpr=0.0`
(benign), which exist precisely to be moved here. Metrics stay **paired** (ASR down is never reported
without over-refusal down and helpfulness up — ADR-0004 rule 5), plus per-guardrail TPR / FPR / FNR and
latency.

**The seam already exists (ADR-0007).** Phase 1b deliberately pre-wired the forward-compatible seams and
deferred only the guardrail engine itself. `safestack/eval/metrics.py` needs **no change** —
`guardrail_fnr` and `guardrail_fpr` are computed purely from the `blocked_at` trace field, and ASR from
the judge label *and* `blocked_at`, so all three already fold it in and need no edit for C2 / C3 / C4;
`TraceRecord` already reserves `guardrail_config`, `blocked_at`, `input_guardrail_ms`,
`output_guardrail_ms`; and `GuardrailConfig = Literal["none","input","output","input_output"]` already
exists in the eval config. What is missing is the `safestack/guardrails/` subpackage, a config field
naming the guardrail *model*, and the wiring in `run_suite` that replaces the C1 stub
(`blocked_at=None`, identity `final_response`).

**The one real tension — ADR-0004 rule 4 (judge / guardrail separation).** The rule forbids the *same*
classifier being both the guardrail and the final safety judge, or the guardrail's block decision and
the ASR score become circular. `meta-llama/Llama-Guard-3-1B` is both the only moderation model
provisioned in Phase 1 (ADR-0006) *and* the frozen safety judge (ADR-0008 decision 4). This bites
hardest on the *output* guardrail, which would otherwise score the same `[user, assistant]` turn as the
judge. Note the constraint is guardrail-vs-*judge*, not input-vs-output: one guardrail model may serve
both the input and output roles (they screen different content).

## Decision

1. **Scope Phase 2 to C2, C3, C4 on the frozen starting model.** No SFT, no model swap; C1 is the frozen
   anchor (ADR-0008), and the SFT rungs (C5–C8) belong to Phase 3. Deliverables: the guardrail-only
   ablation table, the input-vs-output-vs-combined comparison, and a latency report (offline
   per-component overhead — inline-serving latency is the Phase-8 benchmark, not this).
2. **Guardrail model = `Granite Guardian 2B`, kept separate from the frozen `Llama-Guard-3-1B` judge.**
   `Granite Guardian 2B` (IBM, ~2B, Apache-2.0, ungated) does unified prompt- *and* response-risk
   detection with a `safe` / `unsafe` verdict, so one distinct model serves both the input and output
   roles while the judge stays frozen — satisfying ADR-0004 rule 4 cleanly rather than by
   stated-limitation. Because the guardrail is judge-independent, its FNR / TPR are reusable unchanged as
   the H5 containment baseline in Phase 5.
3. **Run the guardrail as separate cached passes, never co-resident with the policy model (ADR-0003).**
   Input guardrail as a pre-pass over prompts (block before spending a 7B generation); output guardrail
   as a post-pass over the cached generations, exactly like the judges. Each model loads and unloads
   separately; peak VRAM stays within the single-GPU target and runs stay resumable. `run_suite` defers
   the trace write until after the guardrail phase so `blocked_at` lands on the trace and
   `safestack/eval/metrics.py` is untouched.
4. **Add `input_guardrail` and `output_guardrail` config fields** to the eval experiment config,
   mirroring `safety_judge` / `helpfulness_judge` (`str | ModelSpec | None`). The factory rejects a
   guardrail whose model matches the safety judge, encoding decision 2 in code.
5. **Block policy = canned safe-refusal, no regeneration**, plus a separate **audit-only** pass. ASR,
   FPR, and FNR key on the block *decision* (`blocked_at`; ASR also on the judge label), not on the
   refusal text, so a fixed refusal template is exactly what the ablation measures; regenerate-or-refuse
   is a serving concern deferred to Phase 8. The non-blocking audit-only run collects clean guardrail
   TPR / FPR / FNR without altering returned responses.
6. **Use the guardrail's native `safe` / `unsafe` decision — no tunable threshold.** With a binary
   verdict there is no operating point to leak across dev and test (ADR-0004 rule 3). If a tunable
   threshold is later wanted, it is set on the dev suite using the paired ASR + over-refusal signal
   (rule 9) and locked before the test run.
7. **Cover the four existing single-turn suites now; defer multi-turn / role-play.** C2 / C3 / C4 run on
   `harmful_advbench_v1`, `harmful_harmbench_v1`, `overrefusal_xstest_v1`, `helpfulness_alpaca_v1`, which
   are single-turn and blunt-or-benign. **H1** (ASR reduction) and **H2** (added over-refusal) are
   confirmatory on these; the **H3** input-vs-output contrast is only *partial and exploratory* here,
   because the dual-use class it turns on — a benign-looking prompt whose generation is unsafe — is one
   the four suites lack (ADR-0008). The multi-turn-escalation and indirect / role-play blind-spot classes
   (master plan §11.5) are an **exploratory** gap (ADR-0004 rule 2) and become their own later slice.
8. **Build incrementally, mock-first: C3 → real `Granite Guardian 2B` → C2 → C4.** C3 lands the whole
   seam (the `safestack/guardrails/` ABC + `build_guardrail` factory + config fields + the `run_suite`
   deferred-write refactor) against a deterministic mock guardrail, CI-green under the `not hf` lane with
   no VRAM, before the real model is wired. Only aggregate `reports/metrics/*.json` are committed
   (ADR-0007 rule 7); raw prompts and pre-block responses stay in gitignored `data/cache/`
   (`RESPONSIBLE_USE.md`, ADR-0007 rule 4).

## Consequences

- **The guardrail-only rungs of H1 get measured.** C1 vs C2 / C3 / C4 quantifies the marginal
  guardrail-alone ASR reduction on the un-aligned model; H1 is only *completed* once the SFT rungs land
  in Phase 3, and adjacent rungs (C2 vs C3) may be CI-overlapping — "no significant difference" is a
  valid outcome (ADR-0004 rule 6).
- **Rule 4 is satisfied by construction, so the guardrail metrics are clean.** Guardrail FNR / TPR are
  judge-independent (computed on dataset labels, not the judge's) and reusable for H5, and the
  ASR-reduction number is not circular — the reason a distinct model was chosen over reusing Llama-Guard
  with a stated limitation.
- **Cost is one extra model card and download (`Granite Guardian 2B`, ~2B, Apache-2.0, ungated).**
  Separate-pass execution keeps peak VRAM low; the guardrail never co-resides with the 7B.
- **`run_suite` needs a real refactor, not a two-line stub swap.** Deferring the trace write past the
  guardrail phase must preserve the C1 trace invariants and the Phase-0 single-prompt runner path.
- **Responsible use binds tightly.** The output guardrail deliberately surfaces the pre-block unsafe
  response; the first response and decision are stored privately, public logs and artifacts are
  aggregate-only or redacted, and the blind-spot analysis is framed as a defense-in-depth ablation, not
  a bypass guide (`RESPONSIBLE_USE.md`).
- **The judge is still an uncalibrated proxy.** Phase-2 ASR / over-refusal inherit the ADR-0008 caveat;
  judge calibration against a 100–300 human-audited sample (ADR-0004 rule 7) remains deferred to Phase 9
  and bounds the interpretation of the guardrail ablation.

## Alternatives considered

- **Reuse `Llama-Guard-3-1B` as the guardrail and report the circularity as a stated limitation** — the
  rule-4 escape hatch. Rejected: a distinct ~2B model is cheap and ungated, so the clean separation is
  affordable and yields judge-independent FNR that H5 can reuse — better than a caveated ASR-reduction.
- **`Llama-Guard-3-8B` as the guardrail** — different weights but the *same family* as the judge (weaker
  independence), gated, and larger. Rejected in favor of a distinct-family 2B.
- **`ShieldGemma-2B`** — viable and distinct, but uses per-policy `Yes` / `No` prompting and the Gemma
  license; `Granite Guardian 2B`'s unified prompt+response `safe` / `unsafe` interface and Apache-2.0
  license fit the existing Llama-Guard-style parse and provenance better.
- **Add multi-turn / role-play eval subsets before any numbers** — fuller H3 coverage, but new dataset
  prep plus dedup / leakage work gating all results. Deferred: single-turn suffices for confirmatory
  H1 / H2 and a partial H3; multi-turn is its own slice if the H3 claim needs to be confirmatory there.
- **Wire the guardrail inline in the generation loop** — simplest, but co-loads a second model with the
  7B resident and breaks resumability. Rejected: violates ADR-0003; the separate cached pass keeps peak
  VRAM low and keeps `metrics.py` untouched.
- **Regenerate-or-refuse on an output block** — matches the serving-time §11.3 wording, but adds a
  capped generation loop and muddies ASR / FPR attribution. Deferred to the Phase-8 serving benchmark.
- **Change the safety judge to de-circularize** — rejected: the judge is frozen by ADR-0008 and changing
  it breaks comparability with the C1 anchor; de-circularization must come from the guardrail side.
