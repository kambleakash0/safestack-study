# Phase 4 — Defense-in-depth analysis: how much safety comes from weights vs external guardrails?

**Status:** Phase-4 written interpretation (a draft input to the Phase-10 report `reports/safestack_report.md`).
**Scope:** a reader-facing synthesis of the C1-C8 result. It references the decision records rather
than restating them — the confirmatory H1/H2 verdict is [ADR-0016](../docs/adr/0016-phase3-h1-sft-alignment-result.md);
the dual-use H3 verdict is [ADR-0014](../docs/adr/0014-dual-use-suite-h3-result.md); the Phase-2 guardrail
rungs are ADR-0010/0011/0012. Every number below is quoted from the committed aggregate artifacts
(`reports/tables/`, `reports/figures/`, `reports/dashboard.html`); nothing here is raw text.

## The question and the design

SafeStack asks: on an open model, **how much of a system's safety comes from the weights, and how
much from external guardrails bolted around them?** Phase 4 answers it with the completed 2×4
ablation — 2 policies × 4 guardrail configurations, C1-C8:

| | no guardrail | input | output | input+output |
|---|---|---|---|---|
| **Starting** (base `Mistral-7B-Instruct-v0.3`) | C1 | C2 | C3 | C4 |
| **SFT** (base + the aligned LoRA adapter) | C5 | C6 | C7 | C8 |

The guardrail is Granite Guardian 3.1-2b; the safety judge is Llama-Guard-3-1B; ASR = judge-`unsafe`
**and** not blocked. Every comparison is read at the ADR-0004 rule-6 bar (`>`/`<` = non-overlapping
95% bootstrap CIs, `≈` = overlapping), with the paired benign metrics (over-refusal, helpfulness)
read alongside (rule 5). The full matrix with CIs is `reports/tables/core_ablation.md`; the charts
are in `reports/dashboard.html`.

## The headline

**Weight-level alignment does the overwhelming majority of the safety work; stacking external
guardrails on the aligned model buys little separable safety while retaining its full over-refusal
cost — and buys *nothing* separable on the dual-use case defense-in-depth was built for.**

## H1 — the ablation (`core_ablation`, `asr_by_condition`, `safety_cost_pareto`, `sft_before_after`)

**Leg 1, the load-bearing test — SFT alignment alone (`C1` → `C5`).** This is the one comparison
whose sign is not mechanically enforced (C5 is a fresh generation, not a cache-hit off C1), and it
lands decisively in the safe direction on every suite, CIs non-overlapping — the **SUPPORT** verdict
of the ADR-0015 decision-5 rule, as recorded in ADR-0016 (a real safety gain with no separable
over-refusal or helpfulness cost):

| ASR (point) | advbench | harmbench | dual-use |
|---|---|---|---|
| C1 base | 0.548 | 0.675 | 0.740 |
| **C5 SFT** | **0.010** | **0.035** | **0.100** |

The SFT before/after table (`reports/tables/sft_before_after.md`) reads the same effect as deltas:
`none` config, ASR −0.538 / −0.640 / −0.640. **This step carries H1.**

**Leg 2, defense stacking on the aligned model (`C5` → `C6/C7/C8`).** Confirmed only where residual
ASR remains: the input guardrail (C6/C8) is CI-separably below C5 on the overtly-harmful suites
(advbench, harmbench, driving the small residual to 0) — but that is the *same* mechanism as Phase-2
C2 (the input screen blocks overt prompts), now largely **redundant** with what the weights already
did. The output guardrail (C7) is **not** separable from C5 on any suite. And on **dual-use** — the
leg engineered to need layered defense — **no guardrail configuration separably improves on the
aligned weights** (`C5 ≈ C6`, `≈ C7`, `≈ C8`; the input rungs fall 0.100 → 0.030 in point estimate
but the CIs overlap).

The Pareto plot lays out the trade-off (`reports/figures/safety_cost_pareto.svg`). Plotting each
condition as (benign-refusal cost = guardrail FPR, safety = 1 − mean ASR), the non-dominated frontier
is **{C5, C7, C6}** — all three are SFT, and *every* Starting condition is dominated. Reading the
frontier: C5 sits at safety 0.952 for **zero** guardrail cost; C7 buys 0.968 for ~0.012 cost; C6/C8
reach 0.990 but only at ~0.33 benign-block cost. The marginal safety a guardrail adds on the aligned
model is small and steeply priced. (The mean-ASR safety axis is a **descriptive** aggregate — an
unweighted mean across three suites of unequal n; per-suite ASR remains the confirmatory unit
(ADR-0016), and C5 → C7's 0.952 → 0.968 movement is itself within-noise, not CI-separable.)

Compare that against Phase 2. The same input guardrail took the base model 0.548 → 0.000 (advbench),
0.675 → 0.000 (harmbench), and 0.740 → 0.190 (dual-use); on the base model it *was* the defense. On
the aligned model it takes 0.010 → 0.000, 0.035 → 0.000, 0.100 → 0.030 (not separable). Its ASR value
collapses once the weights are aligned, but its usability cost stays put.

**Verdict: H1 CONFIRMED** (the non-increasing ordering `C1 > C5 > C6/C7/C8` holds, driven by the
SUPPORT-grade `C1 > C5`); the stacking leg's partial separability is documented at the rule-6 bar
(ADR-0016), not asserted beyond it.

## H2 — the cost side (`over_refusal_by_condition`, guardrail FPR)

SFT did **not** significantly raise the model's own benign refusal (XSTest over-refusal `0.024 → 0.036`,
CIs overlap) and did **not** cost helpfulness (Alpaca `4.915 → 4.910`, CIs overlap) — no mode collapse.
The over-refusal **cost** lives entirely in the *input* guardrail: it blocks ~33% of safe XSTest
prompts (guardrail FPR 0.332 / 0.336 for C6 / C8), while the output guardrail is a gentle 0.012 (C7).
That input-screen cost is essentially unchanged from Phase 2 (C2/C4 were 0.336). So on an aligned
model the **output** screen is the *cheap* layer (near-zero benign cost, FPR 0.012) but it adds **no
CI-separable** safety over the aligned weights on any suite; the **input** screen is the *expensive*
layer (~0.33 benign cost) whose only separable safety gain (advbench/harmbench) is already redundant
with the weights. Neither screen buys separable safety worth its cost on the aligned model.

## H3 — dual-use (ADR-0014, and the aligned-model null)

H3 (that output guardrails catch what input guardrails miss on benign-looking prompts) found **no
support** on the HarmBench-contextual construction in Phase 2 (ADR-0014, verdict WEAK/AMBIGUOUS). Phase
4 adds the aligned-model reading: on dual-use, the SFT weights alone reach ASR 0.100 and **no** guardrail
configuration separably improves on that. It may be the study's most striking result: the one place
layered defense was supposed to earn its keep is exactly where it adds nothing separable on top of the
aligned weights. This is a negative, underpowered result (n = 100, wide CIs), not a proof of equality.

## Where residual harm concentrates (exploratory — `segment_asr_heatmap`, `failure_taxonomy`)

**Exploratory only** (per-category n = 1-58 with degenerate near-0/1 CIs; ADR-0016 keeps per-suite ASR
the confirmatory unit). Ranked by the aligned model's residual (`reports/tables/failure_taxonomy.md`),
the SFT model's remaining failures cluster in **dual-use chemical_biological** (base 0.929 → C5 0.143;
the input guardrail closes it to 0.036), **dual-use illegal** (C5 0.143 — and the guardrail adds
*nothing*, C6/C7/C8 all 0.143, n = 7), and **dual-use cybercrime** (C5 0.111 → C6 0.000). HarmBench is
nearly all driven to ~0. The heatmap (`reports/figures/segment_asr_heatmap.svg`) shows the same
picture: a hot base column, cooler aligned columns, and a couple of stubborn dual-use cells. Read them
as a pointer to *where* to probe next, not as category-level claims.

## Limitations carried into the read

1. **The judge is the measurement floor.** ASR is a single 1B-classifier proxy (`unsafe AND not
   blocked`); "ASR = 0" means "no residue this judge flags," not "provably zero unsafe." No human
   calibration yet — that is Phase 9, and Phase-4 conclusions may need revisiting once it lands.
2. **Helpfulness is block-blind and self-judged.** It scores cached pre-block text (so it hides
   guardrail blocks — the user-facing benign cost is the guardrail FPR, not helpfulness) and the rubric
   judge is the same Mistral-7B family as the policy (a self-preference risk).
3. **The near-0 stacking separations rest on degenerate CIs.** `C5 > C6/C8` on advbench/harmbench uses
   zero-width `[0,0]` percentile-bootstrap intervals at 0/520 and 0/200; Wilson / Clopper-Pearson
   intervals (deferred since ADR-0008) would be better calibrated. CONFIRMED does not lean on these.
4. **Cost = over-refusal only.** Latency / throughput / GPU-memory are **not** measured here — that is
   Phase 8. No operational-cost claim is made.
5. **Segments are exploratory** (limitation restated: tiny-n, no confirmatory category claims).
6. **Single construction.** One model, one guardrail, one judge, one recipe (rank-16 LoRA, 1 epoch),
   single-turn English; no adversarial wrappers or multi-turn. The redundancy/null results are local to
   this construction.
7. **No durability claim.** The weight-level safety this result rests on has not been stress-tested —
   whether robustness-stress fine-tuning strips it (and whether the redundant-looking guardrails then
   reclaim value) is Phase 5 (H4/H5), unstarted. Do **not** read this as "aligned weights are durably
   safe."
8. **Derived recall is a point, not a CI.** Any `1 − ASR_Cx/ASR_C1` reduction is a ratio of point
   estimates, valid only under the identical judge-unsafe set (cache-hit); quote it as a point.

## What this feeds

- **Phase 5 (robustness, H4/H5):** the follow-up that matters most. Does stress fine-tuning undo the
  weight-level safety, and do the guardrails that look redundant here become load-bearing again?
- **Phase 8 (inference benchmarking):** the missing cost axis (latency/throughput/memory per layer).
- **Phase 9 (human audit):** calibrate the judge every ASR here depends on.
- **Phase 10 (final report):** this write-up is the draft that `reports/safestack_report.md`
  consolidates, alongside the committed tables, figures, and the dashboard.

**Bottom line for the study's question:** on this model and suite set, once the weights are aligned,
**the weights dominate.** External guardrails buy little separable safety at unchanged usability cost,
and none on the dual-use case they were meant for. What defense-in-depth is worth here depends on how
much the weights already do; it isn't free-standing.
