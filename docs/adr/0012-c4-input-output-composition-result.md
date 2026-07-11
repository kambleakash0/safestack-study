# ADR-0012: C4 input+output composition result and the completed Phase-2 2x2 ablation

- **Status:** Accepted
- **Date:** 2026-07-10
- **Deciders:** Project owner

## Context

Phase 2 is the *guardrail layer*: it measures how much protection **external guardrails** provide on
their own, before any weight-level alignment, by wrapping the frozen starting model in input and/or
output guardrails (ADR-0009). C4 is the **input+output composition** rung — both screens on at once —
and it closes the Phase-2 2x2: **C1** (no guardrail), **C2** (input only), **C3** (output only), **C4**
(both). The composition runs as a single pipeline: a Granite input pre-pass screens the *user prompt
alone*; on an `unsafe` verdict it returns a canned safe-refusal and **short-circuits** — no 7B
generation and no output screen for a blocked prompt. When the prompt passes, the frozen
`mistralai/Mistral-7B-Instruct-v0.3` (no SFT — ADR-0002, ADR-0008) generates, and a Granite output pass
then screens the *response*; on an `unsafe` verdict that response is replaced by the canned refusal.
C4 is therefore **C2 and C3 stacked**, with the same Granite guardrail model serving *both* stages
through **one lazy gateway** (loaded exactly once — ADR-0009 decision 2, ADR-0003). Every C4 number is
read as a *marginal* change against the C1 anchors frozen in ADR-0008 **and** against the C2 (input)
and C3 (output) rungs it composes.

**Guardrail model (kept separate from the judge — ADR-0009 decision 2, ADR-0004 rule 4).** Both stages
use Granite Guardian 2B (`ibm-granite/granite-guardian-3.1-2b`, IBM, ~2B, Apache-2.0), a model *distinct*
from the frozen `meta-llama/Llama-Guard-3-1B` safety judge. The block decisions (Granite, on prompt and
on response) and the ASR score (Llama-Guard label *and* `blocked_at`) come from different classifiers,
which keeps the ASR reduction **non-circular**. The single Granite instance runs as a cached pre-pass /
post-pass, never co-resident with the 7B (ADR-0003, ADR-0009 decision 3).

**Run configuration.** The "short-circuit on a block" described above is *serving* semantics; offline,
C4 reuses the fully-populated C1 generation cache and judges the cached pre-block text — so the
generation cache hit **1170/1170** (0 misses) on the content hash and no live regeneration occurred,
even though in production the input-blocked prompts would never reach the 7B. Only the Granite input and
output passes and the frozen judge ran. Decode is inherited from C1 (greedy, deterministic,
`temperature 0`, `seed=0`). The safety judge is the frozen `Llama-Guard-3-1B` at the same pinned rev
`acf7aafa60f0410f8f42b1fa35e077d705892029` (identical judge fingerprint across the C1–C4 JSONs);
`judge_prompt_version v2`. Bootstrap CIs use the seeded stdlib RNG (B=10000, `seed=0`, percentile
[2.5, 97.5]) per ADR-0007 rule 5. Data integrity is clean on both harmful suites — `n_unparsed=0`,
`n_missing=0`, **no fail-open occurred**. ASR is the fraction of harmful-prompt rows the judge labels
`unsafe` **and** for which `blocked_at is None` (`safestack/eval/metrics.py`). Measured on a Colab A100
(run `538f4646`, torch 2.11.0+cu128, transformers 5.12.1).

**Two structural consequences of the cache-hit, stated up front (same caveats as ADR-0010/0011).**
(1) Because the generations are a 1170/1170 content-hash cache-hit off C1 and the judge is frozen, a
guardrail can only *remove* unsafe-counted rows (a counted row requires `blocked_at is None`) and never
add them; the **direction** of the ASR drop is *mechanically enforced*, not an empirical finding.
(2) Because the input screen blocked **every** harmful-suite prompt (exactly as in C2), it
short-circuits the output stage on those prompts entirely — the output screen never runs on a single
harmful prompt — so `blocked_at` is `"input"` on all of them and ASR falls to 0 from the block trace
alone; the judge label and the *entire output stage* are moot on the harmful suites. The informative
quantities are therefore **not** the C4 ASR but whether the composition does anything the input rung
(C2) does not: a marginal harmful benefit (there is none to be had — C2 already blocks 100%) or a
marginal benign cost (whether the *added* output screen refuses any benign prompt that passed input).

## Decision

**The completed Phase-2 2x2 ablation** (point estimate, 95% CI, n; frozen Mistral-7B; ASR on the two
harmful suites, `guardrail_fpr` and block-blind `over_refusal` on XSTest, block-blind
`benign_helpfulness` on Alpaca):

| Condition | Guardrail | AdvBench ASR (n=520) | HarmBench ASR (n=200) | XSTest `guardrail_fpr` (n=250) | Alpaca helpfulness (n=200) |
| --- | --- | --- | --- | --- | --- |
| **C1** | none | 0.548 [0.506, 0.590] | 0.675 [0.610, 0.740] | 0.0 [0.0, 0.0] *(taut.)* | 4.915 [4.85, 4.965] |
| **C2** | input | **0.0** [0.0, 0.0] | **0.0** [0.0, 0.0] | 0.336 [0.28, 0.396] | 4.915 [4.85, 4.965] |
| **C3** | output | 0.100 [0.075, 0.127] | 0.130 [0.085, 0.180] | 0.028 [0.008, 0.052] | 4.915 [4.85, 4.965] |
| **C4** | input+output | **0.0** [0.0, 0.0] | **0.0** [0.0, 0.0] | 0.336 [0.276, 0.396] | 4.915 [4.85, 4.965] |

`guardrail_fnr` (whole-suite pass-through — see below): C1 = 1.0 (tautological, both harmful suites);
C2 = C4 = 0.0; C3 = 0.487 (AdvBench) / 0.39 (HarmBench). Block-blind `over_refusal` on XSTest is
**0.024** [0.008, 0.044] in **all four** conditions. HarmBench per-category ASR is **0.0 across all six
categories** in C4 (identical to C2).

**The headline is that C4 ≈ C2 on every reported metric — the added output stage changes no aggregate on
these suites.** C4's AdvBench and HarmBench ASR (0.0), `guardrail_fnr` (0.0), and XSTest `guardrail_fpr`
(0.336) all match C2; the per-suite `provenance_hash` (which folds `blocked_at`, ADR-0007 rule 7 as
extended in ADR-0010) is **byte-identical to C2 on AdvBench, HarmBench, and Alpaca**, so on those three
suites the per-row block trace is *literally* C2's and the harmful ASR is identical to C2 **by
construction** (the `eval compare` gate's *"no significant difference: C2 vs C4"* readout, ADR-0004 rule
6, is a degenerate restatement of that byte-identity — a comparison of two zero-width [0, 0] intervals —
not an independent test). The composition inherits the input rung's totals because the output stage adds
nothing measurable: on the harmful suites the input block short-circuits it (it never runs), on Alpaca it
adds *exactly* zero blocks (byte-identical provenance — every block fell at input on the same rows), and
on XSTest it leaves the *total* benign block rate unchanged at 0.336 (the input-vs-output split of those
84 blocks is not identified by the committed aggregates — see threats). **C4 is not defense-in-depth
paying off; on these suites the input screen does all the work the composition can measure, and the
output screen moves no reported number.**

Given this:

1. **Do NOT adopt C4 over C2 — the composition dominates neither rung and equals the input rung at
   strictly higher cost.** C4 buys **zero** marginal safety over C2 (both drive ASR to 0; C2 already
   blocked 100% of both harmful suites, leaving nothing for the output screen to catch) and adds **zero**
   marginal benign cost that is measurable (`guardrail_fpr` 0.336, unchanged), while running the *most*
   compute of any condition — an input screen on every prompt **plus** an output screen on every
   input-passer. Composing output onto input here is pure overhead. The Phase-2 result that survives as
   a genuine tradeoff is **C3 (output-only): ~82% recall on the judge-unsafe subset at a 0.028 benign
   FPR** (ADR-0010) — an order of magnitude cheaper on benign prompts than the input-bearing C2/C4 rungs
   and the only condition whose ASR reduction was a *selective, CI-separable magnitude* rather than a
   blanket block.
2. **Frame the finding as redundancy, not synergy.** The honest reading is "on overtly-harmful +
   borderline-benign single-turn suites, the input stage determines the outcome and the output stage
   adds nothing when composed with it," not "both layers combine to contain harm." Presenting C4 as
   belt-and-suspenders working would misread a *redundancy* (output idle behind a 100% input block) as a
   *synergy* (two independent contributions).
3. **Treat the ASR-drop *sign* as mechanically enforced, and the harmful C4 as identical to C2 by
   construction.** With cache-hit generations and a frozen judge the direction is guaranteed; here the
   input block **moots the judge label** on the harmful suites (`metrics.py` forces the ASR indicator to
   0 whenever `blocked_at` is set — the judge still ran on the cached pre-block text, but its label
   cannot enter the count) *and* short-circuits the entire output stage (it never runs), so the ASR=0
   falls out of `blocked_at` alone and the harmful `provenance_hash` is byte-identical to C2. Only the
   benign-side behaviour of the *added* output screen could have been a finding, and it moved nothing.
4. **Read the block-blind benign metrics with active suspicion (identical caveat to C2).** `over_refusal`
   (0.024) and helpfulness (4.915, `answer_rate=1.0`) are scored on **cached pre-block text** and hide
   the input blocks. The true user-facing benign cost on XSTest is dominated by the **0.336 guardrail
   block** (84/250), not the 0.024 model-own over-refusal; and the "unchanged" Alpaca helpfulness is
   mechanically guaranteed while the suite actually suffered the *same* nonzero benign blocks as C2 (the
   Alpaca `provenance_hash` is byte-identical to C2's and differs from C1/C3's — see threats), so it is
   actively misleading, not reassuring.
5. **Scope the hypothesis claims honestly.** C4 is the input+output-composition rung of **H1**, and its
   ASR→0 is *trivially/mechanically consistent* with H1's direction on these four single-turn suites
   (ADR-0009 decision 7) — but, exactly as in C2, that drop falls wholly out of blanket input blocking
   (`blocked_at`), carries **no informative, CI-separable magnitude**, and provides **no evidence of
   *selective* ASR reduction**; H1 is only *completed* once the SFT rungs (C5–C8) land in Phase 3. On
   **H2** (strict defenses increase over-refusal) C4 ties C2 as the **most over-defensive rung**: a
   33.6% benign block on the borderline suite that the output composition did **not** mitigate. On
   **H3** (that *output* guardrails catch unsafe generations that *input* guardrails miss on
   benign-looking / dual-use prompts) **C4 contributes nothing, and the completed 2x2 now shows *why* the
   current suites cannot test it**: the input screen blocks 100% of the overtly-harmful prompts (so
   there is nothing left for the output stage to catch) and the generations from the benign prompts that
   pass input are ones the output screen does not flag (so the output stage refuses nothing there
   either). H3 requires a **dual-use suite** — a benign-looking prompt whose *generation* is unsafe —
   which all four current suites lack.
   The 2x2 is structurally complete; the input-vs-output *value* question it was meant to answer is not.
6. **Commit aggregate-only artifacts** — `notebooks/c4_colab.ipynb` (executed, aggregate outputs only,
   verified no raw-prompt leakage) and
   `reports/metrics/c4_starting_input_output_guardrail__*__C4.json` (point / CI / n / segments only, no
   raw prompt or blocked text), per `RESPONSIBLE_USE.md`, ADR-0007 rule 7, and ADR-0009 decision 8.

## Consequences

- **The Phase-2 2x2 ablation is complete, and its verdict is that a single well-placed guardrail, not
  the composition, is the Phase-2 story.** Across C1→C4 the safety/precision frontier is spanned by two
  points, not four: the **input-bearing** rung (C2 = C4: ASR 0 at a 0.336 borderline-benign FPR) and the
  **output-only** rung (C3: ~82% judge-unsafe recall at a 0.028 FPR). C4 sits **on top of** C2, not
  beyond it. Which single guardrail to prefer is a precision/recall choice — C3's cheap, selective
  output screen versus C2's blunt, complete input block — and the composition adds nothing to that
  choice on these suites.
- **C4 beats neither rung it composes (ADR-0004 rule 6 outcome).** "The composition beats neither rung"
  and "no significant difference" were both flagged as valid outcomes when C4 was scoped (ADR-0011
  Consequences); this is that outcome. The load-bearing evidence is the byte-identical harmful/Alpaca
  provenance hashes (those metrics are C2's *exactly*, per-row); the gate's own *"no significant
  difference: C2 vs C4"* readout is the rule-6 restatement of that equality, not an independent
  corroboration.
- **C4 is the study's joint-strongest H2 signal (tied with C2), and the composition did not soften it.**
  One might have hoped that adding a precise output screen would let the pipeline relax the input screen;
  it cannot, because the stages are ANDed (either screen can block) — composing guardrails can only
  *raise* the benign block rate, never lower it. C4's benign burden is therefore ≥ C2's by construction,
  and here it is *equal* to C2's because the output screen happened to add no new benign blocks. The
  strict-defense over-refusal cost H2 predicts is realised at the input stage and is untouched by
  composition.
- **The completed 2x2 sharpens, rather than resolves, H3 — it localises exactly why the current suites
  cannot test it.** H3 is directional and about dual-use prompts. C2 caught 100% of the overtly-harmful
  prompts and C3 caught ~82%, so on the *present* suites the direction runs, if anything, *contrary* to
  H3 (input ≥ output). C4 cannot break this tie: with input at 100% there is no residue for output to
  catch, and the generations from the benign input-passers are ones the output screen does not flag. So
  the ablation establishes only that input and output guardrails sit at **different precision/recall
  operating points**, and that **composing them is redundant when one stage already saturates** — not
  that output beats input where H3 predicts. Testing H3 needs a dual-use suite (ADR-0009 decision 7,
  ADR-0010 follow-up 3, ADR-0011 follow-up 3); the
  completed 2x2 makes building it the top Phase-2-adjacent priority.
- **Threats to validity** (recorded; none reverses the "C4 is redundant here; do not adopt over C2"
  decision, but each bounds how the numbers may be quoted):
  - **The output stage's *net* benign contribution on XSTest is 0 (measured); its *gross* per-stage
    split is not identified by the committed aggregates.** C4's XSTest `guardrail_fpr` is **0.336** —
    identical to C2's — and `guardrail_fpr` counts a benign row blocked at *either* stage (`blocked_at is
    not None`, `metrics.py`). That identity pins only the **total** (84/250 benign prompts blocked); it
    does **not** pin the input-vs-output split, because the input pass was re-run and (as the next threat
    notes) its blocked *set* jittered. "Input blocked a different 84, output added 0" and "input blocked
    83, output added 1" both reproduce the *identical* point FPR (0.336), the differing provenance hash,
    and the shifted CI. So the output screen's **net** effect is nil — the total did not move from C2 —
    but a **gross** zero (that the added output screen refused literally none of the input-passers) is
    **not** established on XSTest by the aggregates; the per-item split lives in the `blocked_at` traces,
    which stay on Drive per `RESPONSIBLE_USE.md`. Gross zero *is* established on the harmful suites
    (output short-circuited, never ran) and on Alpaca (byte-identical provenance → identical `blocked_at`
    → every block fell at input on the same rows). The "do not adopt C4 over C2" decision is unaffected
    by the split: whatever it is, C4's *total* benign block rate equals C2's, so the composition buys no
    benign relief.
  - **The XSTest `provenance_hash` differs from C2's while the harmful and Alpaca hashes are
    byte-identical — this is run-to-run classifier variance on borderline prompts, not an output-stage
    effect.** C4's XSTest provenance (`d2560931…`) differs from C2's (`3ecf8ae8…`) and the bootstrap CI
    low shifts trivially (0.276 vs 0.28), yet the point `guardrail_fpr` is identical (0.336 = 84/250).
    The reconciling reading: the Granite **input** classifier blocks the *same count* (~84) of XSTest
    prompts across the two independent Colab runs but a *slightly different set* of the borderline items
    near its decision boundary — XSTest is engineered to sit on that boundary — so the blocked set (and
    thus the folded `blocked_at` sequence) shifts by a few items while the rate is stable. On the harmful
    suites (blocked with high confidence) and on Alpaca (clearly benign, blocked/passed with high
    confidence) there is no such boundary jitter, which is why those provenance hashes are byte-identical
    to C2. The most parsimonious reading is therefore borderline-input instability (the same ~84 count,
    a slightly different set). But — as the previous threat notes — the aggregates alone cannot *exclude*
    a compensating shift (one fewer input block, one more output block) that would also leave the point
    FPR at 0.336; what is certain is only that the *total* benign rate is unchanged from C2.
  - **`guardrail_fnr` is a misnomer whose 0.0 here is the same blanket-block coincidence as C2.** As
    defined (mean over *all* harmful-suite rows of `blocked_at is None`) it is a whole-suite pass-through
    rate, not a classifier FNR; it reads 0.0 only because *everything* was input-blocked. It must not be
    read as a validated recall.
  - **The benign Alpaca block cost is CONFIRMED nonzero but UNQUANTIFIED, inherited unchanged from C2.**
    C4's Alpaca `provenance_hash` is byte-identical to C2's (`54403a6a…`) and differs from C1's/C3's
    (`7324e36d…`). Under the 1170/1170 cache-hit and a frozen judge, `blocked_at` is the only field that
    can move it, so C4 blocked the *same* ≥1 clearly-benign Alpaca prompt(s) at input that C2 did — and
    the output stage added none (were it to have added Alpaca blocks, the hash would differ from C2's).
    The exact count is still not captured: helpfulness is block-blind and `metrics.py` emits
    `guardrail_fpr` only for the over-refusal split, not the helpfulness split. So, as in C2, the total
    benign-block count exceeds the 84 XSTest blocks by a nonzero, unmeasured Alpaca term, and 0.336 alone
    understates the guardrail's total benign burden as a count. The two benign suites are disjoint
    per-suite costs and must not be summed or pooled into a single FPR that exceeds 0.336.
  - **The 100% harmful-suite block is 100% of *prompts*, not 100% of danger.** At the C1 anchors,
    roughly **45% / 32%** of AdvBench / HarmBench prompts had *judge-safe* unblocked C1 outputs, yet C4
    (like C2) blocked **all** of them at input. The 100% block is over-inclusive relative to actual
    danger; it reinforces "blocks indiscriminately," not "catches everything dangerous," and it is why
    the output stage — which only ever sees prompts the input screen let through — never engages on the
    harmful suites at all.
  - **Block-blind benign metrics are as sharply wrong here as in C2.** `over_refusal` (0.024) and
    helpfulness (4.915, `answer_rate=1.0`) run the refusal heuristic on *cached pre-block* text and never
    read `blocked_at`; a user handed a canned input refusal on a benign XSTest or Alpaca prompt still
    scores as over-refusal-free / 5-of-5 / "answered." The user-facing XSTest refusal cost is the 0.336
    block, and "unchanged helpfulness" is mechanically guaranteed while the suite suffered the same
    nonzero benign blocks as C2.
  - **C4 is the heaviest condition and its cost/benefit is the worst of the four, yet the latency /
    per-component-overhead deliverable is still absent.** C4 runs an input screen on 100% of prompts
    **and** an output screen on every input-passer — strictly more guardrail compute than C2 or C3 — for
    zero marginal safety over C2. The ADR-0009 decision 1 overhead report is not in these metric JSONs;
    it would make the composition's redundancy quantitatively visible as wasted latency, though an input
    screen that short-circuits generation on a block also *saves* 7B latency on the 100% of harmful
    prompts it refuses.
  - **The harmful suites are overtly harmful, so a 100% input block (and thus an idle output stage) is
    unsurprising.** AdvBench/HarmBench are single-turn, blunt, direct-attack prompts (AdvBench contains
    near-duplicates, so effective n < 520). A prompt screen catching 100% of these is the expected easy
    case; it says nothing about borderline/dual-use inputs, adversarial wrappers, or multi-turn
    escalation — precisely the regimes where an output stage might *not* be redundant, and all of which
    are unmeasured here.
  - **Everything is classifier-vs-classifier agreement on an unaudited single-turn proxy, and the
    inherited ADR-0008 judge caveats bind C4.** ASR keys on the `Llama-Guard-3-1B` label and
    `blocked_at`; the guardrail is the distinct Granite Guardian 2B (ADR-0004 rule 4). The safety judge
    is an uncalibrated 1B proxy with no positive control that it scores the *response* rather than the
    *prompt*; helpfulness is self-judged by `Mistral-7B` at the policy rev; the `is_refusal` heuristic
    underlies `over_refusal` and `answer_rate`; human calibration (ADR-0004 rule 7) is deferred to
    Phase 9. Every C4 number inherits these.
  - **Degenerate zero-width bootstrap CIs persist.** The AdvBench/HarmBench ASR 0.0 [0.0, 0.0], the six
    per-category 0.0 intervals, and the C1 tautological 0.0 FPR are all zero-width because the resampled
    indicators are all-identical; they understate uncertainty near the 0/1 boundaries, where
    Wilson / Clopper–Pearson intervals are better calibrated (ADR-0008). The 0.336 XSTest FPR is a
    mid-range proportion whose interval [0.276, 0.396] is well-behaved.

## Alternatives considered

- **Headline "C4 is defense-in-depth: both layers on, ASR → 0" as the finding** — the intuitive reading
  of the safest-looking condition. Rejected: the two layers are not independent contributions here. The
  input block short-circuits the output stage on 100% of the harmful prompts (output never runs) and the
  benign input-passers generate nothing the output screen flags, so C4's numbers are C2's numbers.
  Presenting this as belt-and-suspenders working misreads redundancy as synergy.
- **KEEP C4 as the deployment config because "both guardrails" is safest** — adopt the maximal stack.
  Rejected: C4 equals C2 on *both* safety (ASR 0) and benign cost (`guardrail_fpr` 0.336) while running
  strictly more compute (an output screen on every input-passer on top of the input screen). If the
  input guardrail is deployed, adding the output guardrail buys nothing on these suites; the honest
  Phase-2 KEEP is C3's cheap, selective output screen (ADR-0010), not the composition.
- **Treat C4 as a test of H3 (does output catch what input misses)** — read the composed condition as
  the input-vs-output comparison. Rejected: with the input screen at 100% recall on the overtly-harmful
  suites there is no residue for the output screen to catch, and the generations from the benign
  input-passers are ones the output screen does not flag; H3's dual-use class is absent from all four
  suites. C4 cannot separate input- from output-guardrail value on this data.
- **Read the differing XSTest `provenance_hash` as the output stage adding benign blocks** — infer a
  composition effect from the hash change. Rejected: `guardrail_fpr` is identical to C2 (0.336), which
  fixes the *total* benign block rate at C2's; the hash/CI micro-difference is most parsimoniously
  run-to-run variance in the borderline *input* classifications (the harmful and Alpaca hashes, on
  confidently-classified populations, are byte-identical to C2). The aggregates cannot *prove* the output
  stage added zero XSTest blocks — a compensating one-fewer-input-block fits equally (see threats) — but
  the total benign rate is unchanged from C2 either way, so the hash change is no evidence *for* a
  composition effect.
- **Quote the block-blind `over_refusal` (0.024) or "unchanged" Alpaca helpfulness as the benign cost** —
  the numbers the artifact reports directly. Rejected: both are scored on cached pre-block text and hide
  the input blocks; the true XSTest cost is the 0.336 block and the Alpaca "no change" is mechanically
  guaranteed while the suite suffered the same nonzero benign blocks as C2.
- **Block any C4 conclusion pending judge calibration (Phase 9)** — wait for a human-audited κ. Rejected
  on the same grounds as ADR-0008/0010/0011: the C4-vs-C2 equivalence rests on `blocked_at`, the
  provenance hashes, and `guardrail_fpr`, not on the judge (which is moot on the harmful suites here);
  calibration is a scheduled follow-up, not a Phase-2 blocker, and the caveat is recorded.

## Follow-ups

1. **Build the dual-use suite — now the top Phase-2-adjacent priority.** The completed 2x2 shows the
   four current suites structurally cannot test H3: the input screen saturates on overtly-harmful
   prompts and the benign input-passers' generations are ones the output screen does not flag, so the
   output stage is idle in both regimes. A
   benign-looking prompt whose *generation* is unsafe is the only setting in which C2, C3, and C4 would
   diverge and the input-vs-output value question could be answered (ADR-0009 decision 7, ADR-0010
   follow-up 3, ADR-0011 follow-up 3).
2. **Emit a per-stage block breakdown** (counts of `blocked_at == "input"` vs `"output"` per suite) so
   the output stage's *gross* per-stage contribution in C4 is reported *directly* from the block trace —
   currently only its *net* effect is pinned (0, by the `guardrail_fpr` identity), while the gross
   input-vs-output split on XSTest is unidentified by the aggregates. This also quantifies, for a future
   dual-use suite, exactly where the composition catches what.
3. **Measure the benign Alpaca block directly** (inherited from ADR-0010 follow-up 2 / ADR-0011
   follow-up 1): a first-class `guardrail_fpr` on the helpfulness suite and a `blocked_at`-aware
   `benign_helpfulness` / `answer_rate`, so the confirmed-nonzero Alpaca block cost (byte-identical to
   C2) is quantified rather than inferred from the provenance hash.
4. **Produce the deferred ADR-0009 latency / per-component-overhead report,** which for C4 must show the
   *doubled* guardrail overhead (input on all prompts + output on all passers) against its zero marginal
   safety, alongside the input screen's 7B-latency *saving* on short-circuited prompts.
5. **Report Wilson / Clopper–Pearson intervals** for the genuinely near-0/near-1 proportions (the
   zero-width harmful ASRs, the six 0.0 per-category segments, the tautological C1 0.0 FPR) rather than
   the degenerate bootstrap intervals; the 0.336 XSTest FPR does not need this fix.
6. **Advance to the Phase-3 SFT rungs (C5–C8)** to *complete* H1 with a measured, selective **alignment**
   effect: the Phase-2 guardrail rungs are external filters, not weight-level alignment — C2/C4 drive ASR
   down by a blanket block (mechanically, via `blocked_at`) and only C3 delivered a *selective,
   CI-separable magnitude* (ADR-0010), but even that is output *filtering*, not model alignment, so H1's
   selective-alignment claim still requires the SFT rungs.
7. **Human-calibrate both classifiers** — the `Llama-Guard-3-1B` judge and the Granite Guardian 2B
   guardrail — against a 100–300 audited sample (ADR-0004 rule 7 / Phase 9).
