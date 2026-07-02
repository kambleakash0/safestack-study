# ADR-0004: Experimental methodology

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Project owner

## Context

To be credible as a study (and defensible in an interview or paper review), the project
must avoid the standard failure modes: researcher degrees of freedom, leakage between
reward/guardrail/judge, over-reading noisy differences, and blind trust in automated
judges. These rules are fixed now so results are not rationalized after the fact.

## Decision

1. **Preregister hypotheses.** H1–H5 (master plan §5) are the **confirmatory** claims,
   tested on the core conditions C1–C10. They are fixed before results are seen.
2. **Confirmatory vs. exploratory.** Everything beyond H1–H5 on C1–C10 (DPO/GRPO
   comparisons, segment breakdowns, extra conditions) is **exploratory** and labeled as
   such — no dressed-up significance.
3. **Split discipline / no leakage.** Maintain strictly separate splits (master plan §8.2).
   Never evaluate on training examples. Deduplicate train vs. eval by exact and approximate
   match. Keep a dev set for iteration and a **locked test set** for final numbers.
4. **Judge / reward / guardrail separation.** Do not use the *same* classifier as GRPO
   reward, guardrail, and final judge. Where separation is impossible, report it as a
   stated limitation.
5. **Paired metrics.** Always report ASR (down) together with over-refusal (down) and
   benign helpfulness (up). ASR is never reported alone.
6. **Uncertainty.** Report bootstrap 95% CIs on headline metrics. **"No significant
   difference" is a valid, reportable outcome** — adjacent conditions may not be
   CI-separable given suite sizes.
7. **Judge calibration.** Calibrate automated judges against a 100–300 example
   human-audited sample (agreement / Cohen's kappa, confusion matrix); report judge
   limitations.
8. **Reproducibility.** Fixed seeds and decoding params; cache keyed by content hash so a
   result is a pure function of (model, prompt, params) regardless of parallelism. Final
   locked-test numbers run in a deterministic mode.
9. **Checkpoint selection** uses ASR + over-refusal on the dev suite, **not** training
   loss.

## Consequences

- The harness must record seeds, params, and content-hash cache keys with every run.
- Some hypothesized orderings may blur into overlapping CIs; this is anticipated, not a
  failure.
- Extra portfolio breadth (stretch conditions) does not license post-hoc significance
  hunting.

## Alternatives considered

- **Report point estimates without CIs:** simpler but not defensible for a study.
- **Reuse one classifier everywhere:** convenient but circular; explicitly rejected except
  as a clearly-labeled limitation.
