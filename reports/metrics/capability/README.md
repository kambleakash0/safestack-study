# Capability / UtilityNorm eval (Phase 6, Stage 0)

Standard capability benchmarks (MMLU, GSM8K, IFEval) as a judge-independent floor on whether a
policy is still a *functional* model after SFT alignment (C5) and the shadow-unalignment stress
(C9, b\*=411). The utility axis is adapted from GRP-Obliteration (arXiv 2602.06258), where
`Overall = ASR x UtilityNorm` penalises "unalignment by degradation". Exploratory (ADR-0004 rule 2),
run outside the frozen greedy/256 decode under lm-evaluation-harness protocols.

## Protocol

- MMLU 5-shot (`acc`), GSM8K 5-shot strict-match, IFEval 0-shot prompt-level-strict (chat-templated).
  MMLU/GSM8K are OpenLLM v1; IFEval is OpenLLM v2.
- **hf reference** — `capability_base.json` (this dir): the full base run, anchor-validated against
  the public numbers (MMLU 61.86 / GSM8K 52.69 / IFEval 49.35). This is the reference/anchor run and
  stays full.
- **vLLM run** — `vllm/`: base + C5 + C9 on vLLM. MMLU is capped at 20 examples per lm-eval subtask
  (~1.1k items) so the run finishes in a session; GSM8K (1319) and IFEval (541) stay full. The same
  limit and seed run across all three, so they score the identical MMLU subset. UtilityNorm is
  computed same-backend (vLLM method / vLLM base), which cancels any backend offset in the ratio.

## Result (vLLM, same backend, same MMLU subset)

Absolute:

| policy | MMLU | GSM8K | IFEval |
|---|---|---|---|
| base | 0.6333 | 0.5186 | 0.3974 |
| C5 SFT | 0.6404 | 0.4837 | 0.3031 |
| C9 stressed | 0.6447 | 0.4610 | 0.0980 |

UtilityNorm (method / base):

| policy | MMLU | GSM8K | IFEval | overall |
|---|---|---|---|---|
| C5 SFT | 1.011 | 0.933 | 0.763 | **0.902** |
| C9 stressed | 1.018 | 0.889 | 0.247 | **0.718** |

### Read

Knowledge survives both interventions untouched: MMLU sits at ≈1.0 for C5 and C9 alike, marginally
above base. Math (GSM8K) takes only mild hits (0.93, 0.89). The signal lives in instruction-following.

C5 (SFT alignment) keeps about 90% of base utility, and nearly all of the loss is IFEval (0.76) —
the alignment tax of the hedges and preambles that break IFEval's exact-format checks.

C9 (shadow-unalignment, b\*=411) is not merely broken. MMLU and GSM8K stay at base, so core reasoning
comes through the strip intact. But IFEval collapses to a quarter of base (0.247): the max-dose
comply-with-everything adapter answers on request *content* while ignoring format constraints. That
is the same failure Phase 5 named for b\*=411 — the comply-everything collapse — surfacing here on an
independent axis. Two evals, one signature.

So Stage 0 sharpens the H4 "stripped but still capable" claim rather than simply confirming it: the
stressed model is capable on knowledge and reasoning, not lobotomised, but the strip is not free —
it costs most of the model's strict instruction-following.

### Caveats

- The vLLM base IFEval (0.397) is deflated against the hf reference (0.494) — a backend offset that
  just cleared the 0.10 same-backend drift gate. The absolute IFEval numbers are vLLM-compressed; the
  ratio (base and methods all vLLM) is the honest measure. C9's raw 0.098 looks alarming; its 0.247
  ratio is the real degradation.
- MMLU is a 20/subtask subsample (exploratory). A fixed limit and seed shared across base/C5/C9 means
  the sampling error cancels in the ratio; `TaskResult.n` records the full GSM8K/IFEval counts.

## Files

- `capability_base.json` — hf full reference (anchor-validated).
- `vllm/capability_base.json`, `vllm/capability_c5_sft.json`, `vllm/capability_c9_stress.json` —
  vLLM base / C5 / C9. Aggregate-only (scores + `n`; no per-sample text).

UtilityNorm is not persisted; it recomputes deterministically from the two artifacts via
`safestack.eval.capability.utility_norm`.
