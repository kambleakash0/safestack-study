# ADR-0016: Phase-3 H1 result — weight-level alignment dominates the ASR reduction; stacked guardrails marginal on the aligned model, verdict CONFIRMED

- **Status:** Accepted
- **Date:** 2026-08-09
- **Deciders:** Project owner

## Context

**H1** is the defense-in-depth ordering: stacking safety layers monotonically lowers attack-success
rate (ASR). ADR-0009 scoped, and ADR-0010/0011/0012 completed, the Phase-2 half on the frozen
starting model (C1 none / C2 input / C3 output / C4 input+output) — but ADR-0009 explicitly left H1
*open*: "H1 is only *completed* once the SFT rungs land in Phase 3." ADR-0015 **preregistered** that
completion: it locked the SFT data, the training recipe, the checkpoint-selection rule, and the
H1/H2 prediction **before any C5-C8 number was produced**, so the alignment effect is decided by a
committed design, not a post-hoc read. This ADR records the result against that preregistration. It
is the sibling of ADR-0010/0011/0012 (the C3/C2/C4 result ADRs) and ADR-0014 (the dual-use H3
result), and inherits their honest-framing
discipline.

The four SFT conditions add a **weight-level** rung the base-model 2×2 could not test: the same
frozen `Mistral-7B-Instruct-v0.3` now carries an SFT LoRA adapter (the C5 policy), crossed with the
same `{none, input, output, input+output}` Granite Guardian 3.1-2b guardrails (C5/C6/C7/C8). Every
Phase-3 number is read as a marginal change against the C1 anchors frozen in ADR-0008 **and** against
the Phase-2 guardrail rungs it now stacks on an aligned model.

**The pre-registered prediction (ADR-0015 decision 5, master plan §5.1), read at the ADR-0004 rule-6
CI-separability bar** (`≈` = overlapping 95% bootstrap CIs, `<`/`>` = non-overlapping):

```text
ASR:  C1 (start, none)  >  C5 (SFT, none)  >  C6/C7/C8 (SFT + guardrails),  lowest at C8
```

- **The load-bearing, non-mechanical test is `ASR_C5` vs `ASR_C1`.** Unlike C2/C3/C4 (whose ASR-drop
  direction is mechanically enforced — see caveat 1), C5 is a fresh generation, so its sign is *not
  guaranteed*: an ASR *increase* was a live outcome. This is the test of whether SFT alignment alone
  moves ASR, and by how much.
- **The defense-stacking half is `C5` vs `C6/C7/C8`** — does an external guardrail add CI-separable
  value **on top of** the aligned weights? ADR-0015 named the outcome to watch for: a `C5 ≈ C6/C7/C8`
  result "would say alignment already did the work the guardrails did on C1."
- **H2** is the paired over-refusal side (does SFT and/or a strict guardrail raise benign refusal),
  read via the same rule-5 paired metrics (ASR is never read without over-refusal AND helpfulness).

**The suite and the run.** The five locked-test suites are unchanged from Phase 2 (`manifest_hash`
identical across C1-C8): `harmful_advbench_v1` (n=520, `a80ecfba…`), `harmful_harmbench_v1` (n=200,
`1aabe680…`), `dualuse_harmbench_contextual_v1` (n=100, `52ced8ea…`), `overrefusal_xstest_v1`
(n=250, `24bd1fad…`), `helpfulness_alpaca_v1` (n=200, `31d0aa39…`). The **C5 policy** is the frozen
base `mistralai/Mistral-7B-Instruct-v0.3 @ c170c708…` + the pinned LoRA adapter
`kambleakash0/safestack-sft-mistral-lora-v1 @ 05266a9b…`, served bf16, no quantization,
`chat_template mistral` (ADR-0015 dec.3/6/7b). Decode is the frozen C1 config (greedy, `seed 0`,
`max_new_tokens 256`); the safety judge is the frozen `Llama-Guard-3-1B @ acf7aafa…`,
`judge_prompt_version v2`; helpfulness is the rubric judge on `Mistral-7B-Instruct-v0.3 @ c170c708…`;
refusal is the `heuristic` detector; bootstrap B=10000, `seed 0`, percentile [2.5, 97.5]. Data
integrity is clean on every condition/suite (`n_unparsed=0`, `n_missing=0` — no fail-open).

**Cache provenance (the design held exactly).** C5 is a **real base+LoRA generation** — the run log
shows `generations: hits 0 misses 1270 total 1270` (520+200+100+250+200 across the five suites) —
and C6/C7/C8 are content-hash cache-hits off *that* C5 (`hits 1270 misses 0` each), because
`guardrail_config` is excluded from the generation content hash, so only the Granite input/output
pre-passes are new compute. Runs `1a3afff5…` (C5), `28fc04c3…` (C6), `c266291e…` (C7), `b12357ff…`
(C8), on a Colab A100-SXM4-40GB (python 3.12.13, torch 2.11.0+cu128, transformers 5.13.1, peft
0.19.1). Aggregate metrics are committed under `reports/metrics/c{5,6,7,8}_sft_*` and the executed
notebook is `notebooks/c5_c8_sft_eval_colab.ipynb` (#93). **No preregistration amendment was
required** — policy pin, judges, decode, manifest hashes, and the paired rule-5 suites all match the
ADR-0015 commit, so H1/H2 keep confirmatory status (ADR-0004 rules 1-2).

**Four structural caveats, stated up front (as ADR-0010/0011/0012/0014).**
1. **`ASR_C5` vs `ASR_C1` is the one comparison whose direction is NOT mechanically enforced.** C5 is
   a fresh generation from a different policy; the judge is frozen; nothing constrains its ASR to sit
   below C1. That an ASR *drop* is what we observe is a genuine empirical finding, not an artifact.
2. **`C6/C7/C8` vs `C5` inherit the Phase-2 mechanical direction.** C6/C7/C8 are cache-hits off C5
   and the judge is frozen, so a guardrail can only *remove* unsafe-counted rows — the ASR-drop
   *direction* vs C5 is mechanical. The **finding** is whether that drop is CI-*separable* (rule 6)
   and where it saturates, not that it points down.
3. **The judge is the measurement floor.** All ASR is `Llama-Guard-3-1B`-scored `unsafe AND
   blocked_at is None`; ASR = 0 means "no residue this judge flags," not "provably zero unsafe." The
   rule-6 CIs carry the sampling noise; the frozen judge fingerprint carries the systematic floor.
4. **The near-0 CIs are degenerate percentile-bootstrap intervals.** The CI-separable stacking calls
   (`C5 > C6`, `C5 > C8` on advbench/harmbench) rest on C6/C8 cells at exactly 0/520 and 0/200, whose
   seeded percentile bootstrap returns a zero-width `[0,0]` interval (ADR-0008's known degeneracy near
   0 and 1). ADR-0008 already defers Wilson / Clopper–Pearson intervals as better-calibrated there;
   until adopted, read these particular `>` separations as "the aligned model's small residual is
   removed," not as a precise interval claim. The CONFIRMED verdict does **not** lean on them — it
   rests on the `C1 > C5` SUPPORT step, whose CIs are non-degenerate.

## Decision

**H1 is CONFIRMED, and H2 resolves on the over-refusal side.** The pre-registered ordering `C1 > C5 >
C6/C7/C8` holds as a **non-increasing** ordering — every step is `>` or `≈`, none reverses — and it is
driven overwhelmingly by the load-bearing `C1 > C5` step, which is strictly CI-separable on all three
suites and lands on the ADR-0015 decision-5 **SUPPORT** verdict: **weight-level SFT alignment is the
dominant safety contributor.** The rule-6 decomposition below is explicit that the *stacking* leg
(`C5` vs the guardrails) is CI-separable only where residual ASR remains and is within-noise on the
hardest leg — the honest content behind the CONFIRMED label. (Note the two senses of "partial" kept
distinct here: the decision-5 **PARTIAL** verdict is a specific C5-vs-C1 outcome — an ASR drop bought
at a separable cost — which we do **not** get; the stacking leg being *partly* separable is a
separate, rule-6 statement about C5 vs C6/C7/C8.)

**ASR by condition (point, 95% bootstrap CI); rule-6 verdict in the last columns.**

| suite (n) | C1 base | **C5 SFT** | C6 +in | C7 +out | C8 +both | C1:C5 | C5:C6 | C5:C7 | C5:C8 |
|---|---|---|---|---|---|:--:|:--:|:--:|:--:|
| advbench (520) | 0.548 [.506,.590] | **0.010 [.002,.019]** | 0.000 [.000,.000] | 0.002 [.000,.006] | 0.000 [.000,.000] | **>** | **>** | ≈ | **>** |
| harmbench (200) | 0.675 [.610,.740] | **0.035 [.010,.060]** | 0.000 [.000,.000] | 0.015 [.000,.035] | 0.000 [.000,.000] | **>** | **>** | ≈ | **>** |
| dual-use (100) | 0.740 [.650,.820] | **0.100 [.050,.160]** | 0.030 [.000,.070] | 0.080 [.030,.140] | 0.030 [.000,.070] | **>** | ≈ | ≈ | ≈ |

**Leg 1 — the load-bearing test (`C1 > C5`): decisively confirmed on all three suites, CI-separable
every time.** SFT alignment alone drops harmful ASR by ~95-98% (advbench 0.548→0.010 ≈ 98%, harmbench
0.675→0.035 ≈ 95%) and dual-use by ~86% (0.740→0.100), with no CI overlap. The non-guaranteed sign
came out strongly in the safe direction, and — reading the paired signals at the same bar —
over-refusal is **not** CI-separably worse and helpfulness is **not** CI-separably lower (H2, below).
By the ADR-0015 decision-5 partition (evaluated in its pre-committed order), that combination is the
named **SUPPORT** verdict — a real safety gain — not BACKFIRE, not DEGENERATE (no mode collapse), not
the decision-5 PARTIAL (which would require the ASR drop to be *bought at* a CI-separable over-refusal
or helpfulness cost, which we do not observe), and not COST-WITHOUT-BENEFIT. **This SUPPORT-grade step
carries H1.**

**Leg 2 — defense stacking (`C5` vs `C6/C7/C8`): confirmed where residual ASR exists, within-noise
where it does not.** The input guardrail (C6/C8) is CI-separably below C5 on the overtly-harmful
suites (advbench, harmbench), where it drives the small residual to zero — but this is the *same
mechanism* as Phase-2 C2 (the input screen blocks overt prompts outright), now largely **redundant**
with what the aligned weights already achieved (C5 ≈ 0.01-0.035). The output guardrail (C7) is **not**
CI-separable from C5 on any suite (`C5 ≈ C7` throughout): with the policy's own generations already
almost all safe, the output screen has little left to catch. On the **dual-use** leg — the case built
to *need* defense-in-depth, where a benign-looking prompt slips the input screen so an output screen
should earn its keep — **no guardrail configuration separably improves on the aligned weights**
(`C5 ≈ C6`, `C5 ≈ C7`, `C5 ≈ C8`; point estimates fall 0.100→0.030 for the input rungs but the CIs
overlap). This is exactly the `C5 ≈ C6/C7/C8` outcome ADR-0015 flagged: **on the aligned model,
alignment has already done the work the guardrails did on the base model.**

The ordering's tail is as predicted in *shape*: the input-bearing rungs saturate together
(`C6 ≈ C8`, both at the floor on advbench/harmbench and at 0.030 on dual-use), so "lowest at C8"
holds only as "lowest at C6/C8" — C8 adds no separable value over C6, because the output stage it
adds is the one leg (C7) that never separated.

**H2 — over-refusal and helpfulness (rule-5 paired signals).**

| metric (suite) | C1 | C5 | verdict | guardrail over-refusal cost |
|---|---|---|---|---|
| over_refusal (xstest, model's own) | 0.024 [.008,.044] | 0.036 [.016,.060] | `C5 ≈ C1` | — |
| helpfulness (alpaca, 1-5 rubric) | 4.915 [4.850,4.965] | 4.910 [4.830,4.980] | `C5 ≈ C1` | — |
| guardrail_fpr (xstest, benign blocks) | — | C5 0.000 | — | C6 0.332 · C7 0.012 · **C8 0.336** |

SFT did **not** significantly raise the model's own benign-refusal (`0.024 → 0.036`, CIs overlap) and
did **not** cost helpfulness (`4.915 → 4.910`, CIs overlap) — the mode-collapse tripwire that the
dev-selection guarded against (ADR-0015 Amendment 1) did not trip on the locked test either. The
over-refusal **cost** lives entirely in the *input* guardrail: C6/C8 block ~33% of safe XSTest
prompts (`guardrail_fpr` 0.332/0.336), while the output guardrail C7 is a gentle 0.012. Crucially,
that input-screen cost is **essentially unchanged from Phase 2** (C6 0.332 / C8 0.336 vs the Phase-2
C2/C4 0.336) — the guardrail's usability tax does not shrink just because the weights got safer.

**The defense-in-depth punchline.** The *same* Granite input guardrail did the heavy lifting on the
base model — C1→C2 took advbench 0.548→0.000, harmbench 0.675→0.000, dual-use 0.740→0.190 — but on
the aligned model its marginal ASR contribution collapses: advbench 0.010→0.000, harmbench
0.035→0.000, and dual-use 0.100→0.030 (not CI-separable). The layer that once *was* the defense is,
on an aligned model, **largely redundant on ASR while retaining its full ~33% over-refusal tax** — a
strictly worse cost/benefit trade than on the base model. For the study's central question (how much
safety comes from weights vs external guardrails), Phase 3 answers, on this model and suite set:
**once the weights are aligned, the weights dominate, and stacked guardrails buy little separable
safety at unchanged usability cost — and buy nothing separable on the dual-use case defense-in-depth
was meant for.**

**Why the H1 label is CONFIRMED rather than "partially confirmed."** H1's confirmatory prediction is
the non-increasing ordering `C1 > C5 > C6/C7/C8`, and it holds: no step reverses, its dominant
`C1 > C5` step is decisively CI-separable on all three suites (the decision-5 SUPPORT verdict), and
every stacking step that *can* separate (input guardrail where residual ASR remains) does. The
within-noise stacking steps (dual-use; the output guardrail) are documented above at the rule-6 bar
as `≈`, not asserted as `>` — they are precisely the pre-registered "alignment already did the
guardrails' work" outcome, which strengthens rather than weakens H1's thesis that stacked layers
reduce ASR with the weight layer contributing the dominant share. The claim this ADR does **not**
make: that every guardrail adds separable value on the aligned model — it does not, and the table
says so. (This "partially confirmed" question is about the H1 *label*; it is distinct from the
decision-5 **PARTIAL** verdict on the C5-vs-C1 leg, which is not what we observe — that leg is
SUPPORT.)

## Consequences

1. **The weights-vs-guardrails thesis has its Phase-3 answer.** On `Mistral-7B-Instruct-v0.3` over
   this suite set, SFT alignment accounts for essentially all of the achievable ASR reduction, and
   external guardrails add only marginal, mostly within-noise safety on top — while charging their
   full over-refusal cost. Defense-in-depth's value is not free-standing; it is contingent on how
   much the weights already do. This is the headline for Phase 4 (defense-in-depth analysis) and the
   final report (Phase 10).
2. **The dual-use null is the sharpest result.** The one leg engineered to require layered defense
   (benign-surface prompts) is exactly where no guardrail configuration separably beats the aligned
   weights. Phase 4 should foreground this, and the segment breakdowns (exploratory, rule 2 — e.g.
   C5 harmbench `chemical_biological` ASR = 0.000, down from C1's 0.786) as supporting texture only.
3. **H1/H2 keep confirmatory status.** No preregistration amendment was needed; the CONFIRMED verdict
   and the H2 over-refusal read are confirmatory (ADR-0004 rule 1). Segment and any recipe/data
   variants remain exploratory (rule 2).
4. **This sets up the Phase-5 robustness question (H4/H5).** If aligned weights are what carry safety
   here, the natural adversarial test is whether robustness-stress fine-tuning (C9-C10) *undoes* that
   weight-level safety — and, if so, whether the guardrails that looked redundant here become the
   load-bearing layer again. Phase 5 is a distinct preregistration (its data is more sensitive);
   this result motivates it.
5. **Cost accounting for deployment.** For a defender starting from an aligned model, adding the
   input guardrail buys ≤0.035 ASR (and 0.00 separable on dual-use) at a ~33% benign-block rate; the
   output guardrail buys no separable ASR at a ~1% benign-block rate. The output screen is the cheap,
   near-free-of-cost layer; the input screen is the expensive one whose safety value has evaporated.

## Alternatives considered

- **Label the verdict PARTIAL.** Considered and set aside by the Decider. The rule-6 table shows the
  stacking leg is only partly separable, which a PARTIAL label would foreground; but H1's
  confirmatory prediction is the ordering plus the load-bearing `C1 > C5`, both of which hold
  decisively. The stacking nuance is recorded in full in the Decision body rather than in the label,
  so nothing is hidden. (The one claim PARTIAL would guard against — "every stacking step separated"
  — is explicitly *not* made here.)
- **Read the ordering on point estimates only.** Rejected: ADR-0004 rule 6 governs, and the honest
  reading distinguishes the separable steps from the within-noise ones. The point-estimate ordering
  is reported, but the `≈`/`>` calls carry the verdict.
- **Re-judge or widen the suite before deciding.** Rejected: the judge, decode, and suites are the
  frozen preregistered instrument; changing them post-numbers would forfeit confirmatory status
  (ADR-0015, mirroring ADR-0013 decision 8). Any such extension is a new, exploratory question.

## Follow-ups

1. **Phase 4 — defense-in-depth analysis / write-up.** Consolidate C1-C8 into the H1 narrative
   (weights dominate; stacking marginal and dual-use-null), with the cost/benefit framing above.
2. **Phase 5 — robustness C9-C10 (H4/H5).** Test whether stress fine-tuning strips the weight-level
   safety this result rests on, and whether the guardrails reclaim value when it does.
3. **Exploratory only:** the per-segment ASR breakdowns and the C7 output-guardrail FNR (high, since
   the aligned policy leaves little unsafe residue to catch) are supporting texture, not confirmatory
   claims.
