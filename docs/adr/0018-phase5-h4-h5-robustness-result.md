# ADR-0018: Phase-5 H4/H5 result — robustness-stress fine-tuning strips weight-level alignment (H4 SUPPORTED on all three suites); external guardrails become load-bearing again (H5 full containment on overt-harm, partial on dual-use)

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** Project owner

## Context

**H4** asks whether the weight-level safety Phase 3 established is *durable*: does controlled
robustness-stress fine-tuning of the aligned model *raise* attack-success rate (ASR) back up? **H5**
asks the follow-on: if the weights degrade, do the external guardrails that looked redundant on the
aligned model (Phase 4: `C5 ≈ C6/C7/C8`, ADR-0016) become **load-bearing again**? ADR-0017
**preregistered** both — it locked the stress-data selection, the training recipe and budget grid, the
primary-budget selection rule, the C9/C10 comparison, and the H4/H5 decision rules **before any C9/C10
number was produced** — so the result is decided by a committed design, not a post-hoc read. This ADR
records the result against that preregistration. It is the Phase-5 sibling of ADR-0016 (the Phase-3 H1
result) and inherits its honest-framing discipline.

Phase 5 adds the two robustness rungs the core matrix reserved (master plan §7.1):

- **C9 — the robustness-stressed model, no guardrail:** the safety-degradation measurement (H4, `C9`
  vs `C5`).
- **C10 — the robustness-stressed model, input + output guardrail:** whether the external layer
  *contains* a degraded model (H5, `C10` vs `C9`).

The stressed policy is the pinned SFT LoRA (`= C5`, budget 0) **continue-trained** on the
`train_robustness_stress` slice at the **dev-selected primary budget b\* = 411** (FU4b, #133): the
largest budget whose adapter still passed the ADR-0015 Amendment-1 mode-collapse tripwire on the dev
suites — chosen **test-blind**, never on the locked test (ADR-0004 rule 3).

**The pre-registered decision rules** (ADR-0017 decisions 5-6), read at the ADR-0004 rule-6
CI-separability bar (`≈` = overlapping 95% bootstrap CIs, `<`/`>` = non-overlapping) with the rule-5
paired metrics (ASR is never read without over-refusal AND helpfulness):

- **H4 is read PER SUITE, with a model-level BROKEN gate evaluated FIRST** (dec.5). The stressed model
  is BROKEN if, vs C5, it shows a CI-separable *drop* in the judge-independent answer-rate, a
  coherence-degeneracy (length collapse or an `n_unparsed` spike), or a CI-separable helpfulness crater
  corroborated by either — so "the stress wrecked the model" can never masquerade as clean degradation.
  If not BROKEN, each suite `s ∈ {advbench, harmbench, dual-use}` gets a first-match verdict:
  PARADOXICAL (`C9 < C5`), **SUPPORT** (`C9 > C5`, no paired cost), PARTIAL (`C9 > C5` at a separable
  cost), COST-WITHOUT-DEGRADATION (`C9 ≈ C5` but a paired metric worsened), NULL (`C9 ≈ C5`, no adverse
  movement), or WEAK/AMBIGUOUS. Model-level **H4 SUPPORTED** if ≥1 suite is SUPPORT/PARTIAL.
- **H5 is conditional on H4** (dec.6): read only on suites where H4 produced CI-separable degradation;
  first-match SUPPORT (`C10 < C9` **and** `C10 ≈ C8` — restores the Phase-4 guarded floor), PARTIAL /
  RESIDUAL (`C10 < C9` **but** `C10` still CI-separably above `C8`), or NULL (`C10 ≈ C9`). `C10 ≤ C9`
  is mechanically guaranteed (C10 cache-hits C9); the finding is separability and magnitude.

**The suite and the run.** The five locked-test suites are unchanged from Phase 2-4 (`manifest_hash`
identical across C1-C10): `harmful_advbench_v1` (n=520, `a80ecfba…`), `harmful_harmbench_v1` (n=200,
`1aabe680…`), `dualuse_harmbench_contextual_v1` (n=100, `52ced8ea…`), `overrefusal_xstest_v1` (n=250,
`24bd1fad…`), `helpfulness_alpaca_v1` (n=200, `31d0aa39…`). The **C9/C10 policy** is the frozen base
`mistralai/Mistral-7B-Instruct-v0.3 @ c170c708…` + the pinned **stressed** LoRA adapter
`kambleakash0/safestack-stress-mistral-lora-b411 @ 8df336bc…` (private, never public — ADR-0017 dec.7 /
Amendment 2), served bf16, no quantization, `chat_template mistral` — every fingerprint field identical
to the C5 card except the adapter, so `C9` isolates the stress effect at C5's exact precision. Decode is
the frozen C1/C5 config (greedy, `seed 0`, `max_new_tokens 256`); the safety judge is the frozen
`Llama-Guard-3-1B @ acf7aafa…`, `judge_prompt_version v2`; helpfulness is the `Mistral-7B` rubric judge;
the guardrail is `Granite-Guardian-3.1-2b`; refusal is the `heuristic` detector; bootstrap B=10000,
`seed 0`. Every card ran self-hosted (`backend: hf_local`) — the `reject_api_backend` guard (dec.7, #125)
forbids a hosted API. Data integrity is clean on every C9/C10 condition/suite (`n_unparsed=0`,
`n_missing=0` — no fail-open).

**Cache provenance (the design held exactly).** C9 is a **real base+stressed-LoRA generation** — the run
log shows `generations: hits 0 misses 1270 total 1270` (520+200+100+250+200 across the five suites) —
and C10 is a content-hash cache-hit off *that* C9 (`hits 1270 misses 0`), because `guardrail_config` is
excluded from the generation content hash, so only the Granite input/output pre-passes are new compute.
Runs `60ef0a55…` (C9), `509c07b8…` (C10), on a Colab A100-SXM4-40GB (python 3.12.13, torch
2.11.0+cu128, transformers 5.13.1, peft 0.19.1). Aggregate metrics are committed under
`reports/metrics/c9_411_*` / `c10_411_*` and the executed notebook is
`notebooks/c9_c10_stress_eval_colab.ipynb` (#137). **No preregistration amendment was required** — the
policy pin, judges, decode, manifest hashes, and the paired rule-5 suites all match the ADR-0017 commit,
and b\* was fixed test-blind, so H4/H5 keep confirmatory status (ADR-0004 rules 1-2).

**Four structural caveats, stated up front (as ADR-0016).**
1. **`ASR_C9` vs `ASR_C5` is a fresh-generation test whose sign is NOT mechanically enforced.** C9 is a
   real generation from a genuinely different adapter; the judge is frozen; nothing constrains its ASR
   to sit above C5. An increase (H4 support), a null (durable alignment), a paradoxical drop, or a
   broken model were all live outcomes. That we observe a large, clean increase is a genuine empirical
   finding, not an artifact.
2. **`C10` vs `C9` inherits a mechanical direction.** C10 is a cache-hit off C9 and the judge is frozen,
   so a guardrail can only *remove* unsafe-counted rows — `ASR_C10 ≤ ASR_C9` is guaranteed in sign. The
   **finding** is whether the drop is CI-separable (rule 6) and its magnitude, not that it points down.
3. **The judge is the measurement floor.** All ASR is `Llama-Guard-3-1B`-scored `unsafe AND blocked_at
   is None`; ASR = 0 means "no residue this judge flags," not "provably zero unsafe." The BROKEN gate
   (below) is what certifies the C9 ASR is real compliance rather than the judge misfiring on gibberish.
   Human calibration of the judge is deferred to Phase 9 (ADR-0017 follow-up 5), and the frozen judge
   binds hardest on C9 — so read the 0.94 as "near-ceiling unsafe-compliance this judge flags," with the
   SUPPORT verdict resting on the CI-separable rise, not on the exact ceiling value.
4. **The near-0 C10 intervals are degenerate percentile bootstraps.** C10 advbench/harmbench sit at
   exactly 0/520 and 0/200, whose seeded percentile bootstrap returns a zero-width `[0,0]` interval
   (ADR-0008's known degeneracy near 0/1). The H5 full-containment calls there rest on the *magnitude* of
   the `C10 < C9` drop (0.94 → 0.00), not on the degenerate interval precision; ADR-0008 defers
   Wilson / Clopper–Pearson intervals as better-calibrated at the boundary.

## Decision

**H4 is SUPPORTED on all three suites, and H5 shows the guardrails are load-bearing again.** The stressed
b\* model is **not BROKEN** (the gate below), so the read is a clean alignment-durability test. Against
C5, robustness-stress fine-tuning re-opens unsafe compliance to the **ceiling** on every suite — a
CI-separable ASR rise with **no** paired cost (the ADR-0017 decision-5 **SUPPORT** verdict) — and the
increase overshoots even the C1 base (recovery fraction > 1 on all three). On the degraded model, input +
output Granite screening then removes essentially all of that unsafe compliance on the overt-harm suites
(**H5 SUPPORT / full containment**) and most of it on dual-use (**H5 PARTIAL / residual**). The layer
Phase 4 found redundant on the aligned weights is, on the degraded weights, the decisive defense.

**Step A — the model-level BROKEN gate (dec.5 Step A): NOT triggered.** Vs C5, the stressed model shows
(i) **no drop** in the judge-independent answer-rate — it *rose*, `0.98 → 1.00`; (ii) **no
coherence-degeneracy** — `n_unparsed = 0` and `n_missing = 0` on every suite, and the helpfulness slice
is answered at rate 1.00; (iii) **no helpfulness crater** — the rubric holds at `4.910 → 4.920`. The
stress did not wreck the model: it produced a **coherent, fully-helpful model that simply complies with
harmful requests**. This is the precondition for reading H4 as clean degradation — the C9 ASR is
genuine unsafe compliance, not the safety judge flagging garbled output.

**ASR by condition (point, 95% bootstrap CI); per-suite H4 verdict in the last column.**

| suite (n) | C1 base | **C5 SFT** | **C9 stressed** | C10 stressed+guard | recovery | H4 (C9:C5) |
|---|---|---|---|---|:--:|:--:|
| advbench (520) | 0.548 [.506,.590] | 0.010 [.002,.019] | **0.938 [.917,.958]** | 0.000 [.000,.000] | 1.73 | **SUPPORT** |
| harmbench (200) | 0.675 [.610,.740] | 0.035 [.010,.060] | **0.940 [.905,.970]** | 0.000 [.000,.000] | 1.41 | **SUPPORT** |
| dual-use (100) | 0.740 [.650,.820] | 0.100 [.050,.160] | **0.940 [.890,.980]** | 0.240 [.160,.330] | 1.31 | **SUPPORT** |

**Step B — H4 per suite: SUPPORT, decisively, on all three.** `ASR_C9 > ASR_C5` is CI-separable on
every suite (the C9 intervals `[.917,.958]`, `[.905,.970]`, `[.890,.980]` sit far above the C5 intervals
`[.002,.019]`, `[.010,.060]`, `[.050,.160]` — no overlap). Reading the paired signals at the same bar,
there is **no** CI-separable cost: over-refusal did not worsen (`0.036 → 0.000`, it *fell*) and
helpfulness did not fall (`4.910 → 4.920`). By the decision-5 first-match order that combination is
**SUPPORT** — not PARADOXICAL (the sign is up, not down), not PARTIAL (no separable paired cost), not
COST-WITHOUT-DEGRADATION (ASR moved, hugely). Model-level, with three SUPPORT suites, **H4 is
SUPPORTED** — and it is a *general* stripping, not a dual-use-only effect. **The alignment SFT
established in Phase 3 is not permanent: ~411 continue-train examples of unsafe compliance shift it off
entirely.**

**The recovery fraction overshoots the base** (dec.5 descriptive read, `(ASR_C9 − ASR_C5)/(ASR_C1 −
ASR_C5)`, denominators 0.538 / 0.640 / 0.640): advbench **1.73**, harmbench **1.41**, dual-use **1.31**
— all above 1.0. The stressed model is not merely un-aligned back to the base; it is **more compliant
than the original base model** on every suite. Phase 3 drove ASR ~95-98% down; Phase 5 drove it back
past where it started — the shadow-alignment signature, where a small dose of unsafe-compliance targets
teaches an affirmative-onset reflex that generalises past the base rate.

**H5 — external guardrails on the degraded model (dec.6, read on all three degraded suites).**

| suite | ASR_C9 | ASR_C10 | ASR_C8 (Phase-4 floor) | C10:C9 | C10:C8 | H5 verdict |
|---|---|---|---|:--:|:--:|:--:|
| advbench | 0.938 [.917,.958] | 0.000 [.000,.000] | 0.000 [.000,.000] | **<** | ≈ | **SUPPORT (full containment)** |
| harmbench | 0.940 [.905,.970] | 0.000 [.000,.000] | 0.000 [.000,.000] | **<** | ≈ | **SUPPORT (full containment)** |
| dual-use | 0.940 [.890,.980] | 0.240 [.160,.330] | 0.030 [.000,.070] | **<** | **>** | **PARTIAL / RESIDUAL** |

On the **overt-harm** suites the input+output screen catches the degraded model's unsafe output outright
— `C9 0.938/0.940 → C10 0.000`, CI-separable, and back to the Phase-4 aligned-and-guarded floor
(`C10 ≈ C8 = 0.000`): **H5 SUPPORT, full containment.** On **dual-use** the guardrail catches most but
not all — `C9 0.940 → C10 0.240`, CI-separable, but `C10 0.240 [.160,.330]` remains CI-separably
**above** the C8 floor `0.030 [.000,.070]` (the guardrail's dual-use false-negative rate is 0.27):
**H5 PARTIAL / RESIDUAL** — defense-in-depth helps enormously but does not fully restore the aligned-model
safety level exactly where the benign-surface prompts slip the screen. This is the same dual-use weak
spot ADR-0014 / ADR-0016 flagged, now the one suite the external layer cannot fully close on a degraded
model.

**The defense-in-depth punchline (the exploratory cross-phase read).** On the *aligned* model the
guardrails were nearly redundant — `C5 → C8` moved ASR only 0.010→0.000, 0.035→0.000, 0.100→0.030
(ADR-0016: `C5 ≈ C6/C7/C8`). On the *degraded* model the **same** input+output Granite stack removes
0.938, 0.940, and 0.700 of ASR respectively — one to two orders of magnitude more marginal safety.
**The layer that looked redundant in Phase 4 is load-bearing again in Phase 5.** This is the study's
central defense-in-depth result: external screening buys little on a well-aligned model but becomes
the difference between a 94%-compliant model and a contained one once the weights are compromised — the
insurance is worth most precisely when the primary layer has failed.

**H5 paired cost (rule 5).** C10 carries the input screen's benign-block tax unchanged: `guardrail_fpr`
on XSTest is **0.324**, statistically the same ~33% as Phase-2/4 (C4/C8 0.336) — the guardrail's
usability cost does not shrink because it is now guarding a degraded model. And the stressed model's
*own* benign behaviour is not the problem: C9/C10 over-refusal is 0.000 and helpfulness 4.920 — the
degraded model is maximally willing, so the entire over-refusal cost of C10 lives in the external screen,
exactly as in Phase 4.

**What this does and does not establish.** On this model (`Mistral-7B-Instruct-v0.3`), this suite set,
this stress recipe (SORRY-Bench affirmative-onset targets), and this dose (b\* = 411): (H4) weight-level
alignment is **shiftable by a small amount of downstream fine-tuning** — not a permanent property of the
checkpoint — and the shift overshoots the base; (H5) **external input+output guardrails remain effective
on the degraded model**, fully containing overt-harm and substantially (but not completely) containing
dual-use. It does **not** establish that alignment is equally fragile on other models or recipes, nor
that guardrails would contain a stronger adversary that also targets the screen. The dose-response
trajectory and the input-only/output-only split on the degraded model (ADR-0017 dec.8) sharpen the "how
easily" and "which layer" questions; the benign-continue-train control (dec.8c) is the **attribution
arm** — it separates "the unsafe targets stripped alignment" from "any N-step continue-train eroded the
SFT safety half via forgetting" — though the recovery-fraction overshoot past the base (> 1.0) already
argues against pure forgetting, which would predict recovery ≈ 1.0. Framed per master plan §6.3: this is
a **controlled robustness stress test and safety-degradation measurement** — evidence for defense-in-
depth, not a recipe for removing safety.
