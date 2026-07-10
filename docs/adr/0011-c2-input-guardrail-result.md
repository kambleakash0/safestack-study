# ADR-0011: C2 input-guardrail result and the input-vs-output contrast

- **Status:** Accepted
- **Date:** 2026-07-10
- **Deciders:** Project owner

## Context

Phase 2 is the *guardrail layer*: it measures how much protection **external guardrails** provide on
their own, before any weight-level alignment, by wrapping the frozen starting model in input and/or
output guardrails (ADR-0009). C2 is the **input-only** rung: a separate pass screens the *user prompt
alone*, before generation, and on an `unsafe` verdict returns a canned safe-refusal and short-circuits
the run — no 7B generation and no output-screening happen for a blocked prompt. When the prompt passes,
the frozen `mistralai/Mistral-7B-Instruct-v0.3` (no SFT — ADR-0002, ADR-0008) generates normally. C2 is
the mirror of C3: **same Granite guardrail model, but it screens the *prompt* where C3 screened the
*response*.** Every C2 number is read as a *marginal* change against the C1 anchors frozen in ADR-0008,
including the tautological no-defense baselines `guardrail_fnr=1.0` (harmful) and `guardrail_fpr=0.0`
(benign), which exist precisely to be moved here.

**Guardrail model (kept separate from the judge — ADR-0009 decision 2, ADR-0004 rule 4).** The input
guardrail is Granite Guardian 2B (`ibm-granite/granite-guardian-3.1-2b`, IBM, ~2B, Apache-2.0), a model
*distinct* from the frozen `meta-llama/Llama-Guard-3-1B` safety judge — the same guardrail model used at
the output in C3, now applied to the prompt. This separation is what keeps the ASR-reduction
**non-circular**: the block decision (Granite, on the prompt) and the ASR score (Llama-Guard label *and*
`blocked_at`) come from different classifiers. The guardrail runs as a cached pre-pass over prompts,
never co-resident with the 7B (ADR-0003, ADR-0009 decision 3).

**Run configuration.** The "short-circuit / no 7B generation on a block" described above is *serving*
semantics; offline, C2 reuses the fully-populated C1 generation cache and judges the cached pre-block
text — so the generation cache hit **1170/1170** (0 misses) on the content hash and no live regeneration
occurred, even though in production 100% of the harmful prompts would never reach the 7B. Only the
Granite input pre-pass and the frozen judge ran. Decode is therefore inherited from C1 (greedy,
deterministic, `temperature 0`, `seed=0`). The safety judge is the frozen `Llama-Guard-3-1B` at the same
pinned rev `acf7aafa60f0410f8f42b1fa35e077d705892029` (identical judge fingerprint in the C1 and C2
JSONs); `judge_prompt_version v2`. Bootstrap CIs use the seeded stdlib RNG (B=10000, `seed=0`, percentile
[2.5, 97.5]) per ADR-0007 rule 5. Data integrity is clean on both harmful suites — `n_unparsed=0`,
`n_missing=0`, **no fail-open occurred**. ASR is defined in `safestack/eval/metrics.py` as the fraction
of harmful-prompt rows the judge labels `unsafe` **and** for which `blocked_at is None`. Measured on a
Colab A100.

**One structural consequence of the cache-hit, stated up front (same caveat as ADR-0010).** Because the
generations are a 1170/1170 content-hash cache-hit off C1 and the judge is frozen, an input guardrail can
only *remove* unsafe-counted rows (a counted row requires `blocked_at is None`) and can never add them.
The **direction** of the ASR drop is therefore *mechanically enforced*, not an empirical finding. For C2
this is sharper still: because the input screen blocked **every** harmful-suite prompt, `blocked_at` is
non-`None` on all of them, so ASR falls to 0 purely from the block trace — the judge label does not even
enter the harmful-suite ASR here. The informative quantities are not the ASR sign but the **magnitude of
the block** and, above all, its **false-positive (precision) cost** on benign prompts.

## Decision

The C2 measurement, point estimate with 95% CI and n. ASR on AdvBench (`harmful_advbench_v1`) is
**0.0** [0.0, 0.0], n=520, versus the C1 anchor 0.548 [0.506, 0.590] — a **−0.548** absolute (100%
relative) drop, `guardrail_fnr 0.0` [0.0, 0.0]. ASR on HarmBench (`harmful_harmbench_v1`) is **0.0**
[0.0, 0.0], n=200, versus C1 0.675 [0.610, 0.740] — a **−0.675** absolute (100% relative) drop,
`guardrail_fnr 0.0`; per-category ASR is **0.0 across all six categories** (`chemical_biological` n=28,
`cybercrime_intrusion` n=40, `harassment_bullying` n=19, `harmful` n=21, `illegal` n=58,
`misinformation_disinformation` n=34). On the benign side, the guardrail adds a `guardrail_fpr` of
**0.336** [0.28, 0.396] on the borderline XSTest suite (`overrefusal_xstest_v1`, = **84/250** benign
prompts blocked at input), up from the tautological C1 0.0; the model's own `over_refusal` is
**unchanged at 0.024** [0.008, 0.044], n=250 (block-blind — see threats); and benign helpfulness on
Alpaca (`helpfulness_alpaca_v1`) shows **no change in the block-blind score**, 4.915 / 5 [4.85, 4.965],
`answer_rate=1.0`, n=200.

**The headline is the precision cost, not the recall.** The informative C2 quantity is the
**false-positive cost of blanket prompt-blocking**: a `guardrail_fpr` of **0.336** on the borderline
XSTest suite (84/250 benign prompts refused before the model ever ran), *plus* a confirmed-but-unquantified
count of benign Alpaca blocks on a *separate* suite (see the provenance finding below) — two disjoint
per-suite costs, not one summed rate. The XSTest number alone is already an **order of magnitude higher**
than C3's 0.028 borderline over-block. Harmful **recall is trivially ~100% and is *not* the informative
number** — the inverse of C3. Because the screen blocked 100% of every harmful suite, the derived recall
`1 − ASR_C2/ASR_C1` is **1.0** on both suites; but that is perfect *because the screen blocks
indiscriminately*, not because it is selective. Unlike C3, where ~82% recall on the judge-unsafe subset
was the honest headline, C2's recall is uninformative and must **not** be headlined as if it demonstrated
selectivity.

**Do not headline the reported `guardrail_fnr` here either — but for the opposite reason as C3.**
`metrics.py` computes `guardrail_fnr` as the mean of `blocked_at is None` over **all** harmful-suite
rows — a whole-suite *pass-through* rate, not a classifier FNR. In C3 that metric *understated* recall
(pass-through ~0.49 / 0.39). In C2 it reads 0.0 and *coincides* with perfect harmful-recall — but only
because **everything** was blocked, so the whole-suite pass-through rate and the judge-unsafe-subset FNR
happen to collapse to the same 0. The name is still a misnomer; the coincidence is an artifact of
blanket blocking, not evidence that the metric measures recall.

**Per-category HarmBench (exploratory — ADR-0004 rule 2).** ASR is uniformly 0.0 across all six
categories. This reflects the blanket input block, not per-category discrimination, and carries no
category signal.

Given this:

1. **Do NOT adopt the input guardrail standalone on these suites — carry C2 as a *component* into C4.**
   C2 alone drives ASR to 0, but at an order-of-magnitude higher benign cost than C3 (`guardrail_fpr`
   0.336 vs 0.028 on the same XSTest suite, plus unmeasured Alpaca blocks). This is **not a clean KEEP**
   like C3. The honest decision is to **carry C2 into C4 (input+output composition)**, where the input
   screen's harmful recall and the output screen's precision may combine, and let C4 decide whether the
   composition dominates either rung alone. Input-only over-blocks too aggressively to be adopted on its
   own here.
2. **Frame the finding as the input-vs-output precision/recall contrast, not "100% recall."** Report the
   informative quantity as the false-positive cost of blanket prompt-blocking: a `guardrail_fpr` of
   **0.336** on XSTest (84/250) plus a *separate*, confirmed-nonzero but unmeasured benign-block term on
   Alpaca — two disjoint per-suite costs that must not be summed or pooled into a single FPR. State the
   recall as trivially perfect *because* the screen blocks indiscriminately, and do not present it as
   selective containment.
3. **Treat the ASR-drop *sign* as mechanically enforced, not a discovery.** With cache-hit generations
   and a frozen judge the direction is guaranteed; here the block short-circuits the judge entirely on
   the harmful suites, so even the ASR=0 value falls out of `blocked_at` alone. Only the block magnitude
   and the false-positive cost are findings.
4. **Read the block-blind benign metrics with active suspicion.** `over_refusal` (0.024) and helpfulness
   (4.915, `answer_rate=1.0`) are scored on **cached pre-block text** and hide the input blocks. The true
   user-facing refusal cost on XSTest is dominated by the **0.336 guardrail block**, not the 0.024
   model-own over-refusal; and the "unchanged" Alpaca helpfulness is mechanically guaranteed while the
   suite actually suffered nonzero benign blocks (see threats), so it is **actively misleading**, not
   reassuring.
5. **Scope the hypothesis claims honestly.** C2 is the input-guardrail-alone rung of **H1**, and its
   ASR→0 is *trivially/mechanically consistent* with H1's direction on these four single-turn suites
   (ADR-0009 decision 7) — but because that drop falls wholly out of blanket blocking (`blocked_at`),
   with no informative, CI-separable magnitude, it carries **weaker confirmatory weight than C3's**
   selective drop and provides **no evidence of *selective* ASR reduction**. H1 is only *completed* once
   the SFT rungs (C5–C8) land in Phase 3. C2 is the **strongest evidence so far for H2** (strict defenses
   increase over-refusal): a 33.6% benign block on the borderline suite plus nonzero benign-helpful
   blocks, versus C3's weak/near-null +2.8%. On **H3** (that *output* guardrails catch unsafe generations
   that *input* guardrails miss on benign-looking/dual-use prompts), C2 *alone* contributes **nothing** —
   H3 is inherently an input-vs-output comparison, so only the **C2-vs-C3 contrast** can speak to it. That
   contrast establishes only that the two guardrails sit at different precision/recall operating points;
   on the overtly-harmful suites present it runs, if anything, *contrary* to H3 (input caught 100% vs
   output ~82%), and the dual-use class H3 turns on is absent. C2 vs C3 therefore does **not** test H3;
   C4 plus a dual-use suite are required.
6. **Commit aggregate-only artifacts** — `notebooks/c2_colab.ipynb` (executed, aggregate outputs only,
   verified no raw-prompt leakage) and `reports/metrics/c2_starting_input_guardrail__*__C2.json`
   (point / CI / n / segments only, no raw prompt or blocked text), per `RESPONSIBLE_USE.md`, ADR-0007
   rule 7, and ADR-0009 decision 8.

## Consequences

- **The input-guardrail-alone rung of H1 is recorded, but its "value" is not a clean KEEP.** C1 vs C2
  shows the ASR reduction is *mechanically* 100% on both harmful suites (blanket block, not a measured
  magnitude — decision 3); H1 is *completed* only when the Phase-3 SFT rungs land (ADR-0009
  Consequences), and the adjacent-rung contrast (C2 vs C3) may still be reframed by C4 — "the composition
  beats neither rung" and "no significant difference" are both valid outcomes (ADR-0004 rule 6).
- **C2 is the study's strongest H2 signal so far, and it is a real cross-condition contrast.** The
  input screen adds a 33.6% block rate on the borderline suite plus nonzero benign-helpful blocks,
  against C3's near-null +2.8%. Same Granite model, prompt-vs-response content: the strict-defense
  over-refusal cost H2 predicts is *large* for input screening and *small* for output screening on these
  single-turn suites.
- **The complementary error profiles (C2 vs C3) do not bear on H3.** H3's claim is directional — that
  *output* guardrails catch unsafe generations that *input* guardrails miss on benign-looking, dual-use
  prompts. The dual-use class that claim turns on is absent from all four current suites, and on the
  overtly-harmful suites that *are* present the observed direction runs, if anything, *contrary* to H3:
  the input screen (C2) caught **100%** while the output screen (C3) caught only **~82%** (FNR ~18–19%
  on judge-unsafe). So the contrast establishes only that input and output guardrails sit at **different
  precision/recall operating points** — C2 input = perfect harmful-recall (ASR 0) at high FPR (0.336
  XSTest); C3 output = lower recall (~82%) at low FPR (0.028) — not that output beats input where H3
  predicts. Testing H3 needs **C4 plus a dual-use suite** (benign-looking prompt whose *generation* is
  unsafe), which the four current suites lack (ADR-0009 decision 7, ADR-0010 follow-up 3). On
  overtly-harmful suites a 100% block is unsurprising and says nothing about the borderline/dual-use
  inputs H3 turns on.
- **Rule 4 is satisfied by construction, so the metrics are clean — but on the harmful suites here the
  judge is moot.** The Granite guardrail is judge-independent, so the ASR reduction is non-circular; but
  because every harmful prompt was blocked, `blocked_at` alone forces ASR to 0 and the judge label never
  enters. The non-circularity is real but does no work on the harmful suites in this condition.
- **Threats to validity** (recorded; none reverses the "carry into C4, do not adopt standalone"
  decision, but each bounds how the numbers may be quoted):
  - **The benign Alpaca block cost is CONFIRMED nonzero but UNQUANTIFIED (the provenance finding).** The
    C1-vs-C2 `helpfulness_alpaca` `provenance_hash` **differs**. That hash is taken over sorted
    `eval_id → gen_hash → judge_key` and folds `blocked_at` (ADR-0007 rule 7, as extended in ADR-0010).
    Under the 1170/1170 cache-hit (`gen_hash` unchanged) and a frozen, identical judge (`judge_key`
    unchanged), `blocked_at` is the **only** remaining input that can move — so a differing helpfulness
    `provenance_hash` implies **≥1** clearly-benign Alpaca prompt was blocked at input. But the exact
    count is **not captured by the committed metrics**: helpfulness is block-blind and `metrics.py`
    emits `guardrail_fpr` only for the over-refusal split, not the helpfulness split. So C2's benign
    block cost on general-helpful prompts is *confirmed nonzero, quantity unknown*. (Contrast C3, where
    ADR-0010 found this `provenance_hash` byte-identical → 0 Alpaca blocks.) The XSTest FPR is exactly
    **0.336** (84/250) regardless of Alpaca; Alpaca carries its own *separate*, unmeasured benign-block
    rate on a differently-sized, clearly-benign population. The two are disjoint per-suite costs and
    **cannot be summed or pooled into a single FPR that exceeds 0.336** — pooling the easy-benign Alpaca
    suite would, if anything, pull an aggregate benign FPR *below* the borderline-XSTest 0.336. What *is*
    strictly true is that the **total benign-block count exceeds the 84 XSTest blocks** by a nonzero,
    unmeasured Alpaca term, so 0.336 alone understates the guardrail's total benign burden as a count.
  - **The 100% harmful-suite block is 100% of *prompts*, not 100% of danger.** At the C1 anchors (ASR
    0.548 / 0.675), roughly **45% / 32%** of AdvBench / HarmBench prompts had *judge-safe* unblocked C1
    outputs — the frozen model would have been scored safe on its own — yet C2 blocked **all** of them at
    input. Per ADR-0010's logic this adds little *incremental* user-refusal cost on the harmful suites
    (those prompts would plausibly have been refused or judged safe anyway), but it means the 100% block
    is **over-inclusive relative to actual danger** and further undercuts any reading that the input
    screen did critical work on 100% of these prompts. It reinforces "blocks indiscriminately," not
    "catches everything dangerous."
  - **Block-blind benign metrics are *sharper* wrong here than in C3, because the block rate is an order
    of magnitude higher.** `over_refusal` (0.024) and helpfulness (4.915, `answer_rate=1.0`) run the
    refusal heuristic on *cached pre-block* text and never read `blocked_at`; a user handed a canned
    input refusal on a benign XSTest or Alpaca prompt still scores as over-refusal-free / 5-of-5 /
    "answered." The user-facing XSTest refusal cost is dominated by the 0.336 guardrail block, not the
    0.024 model-own over-refusal; and "unchanged helpfulness" is mechanically guaranteed while the suite
    demonstrably suffered nonzero benign blocks — so that number measures nothing about C2's benign
    behavior and is actively misleading, not reassuring.
  - **`guardrail_fnr` is a misnomer whose 0.0 here is a coincidence.** As defined (mean over *all*
    harmful-suite rows of `blocked_at is None`) it is a whole-suite pass-through rate, not a classifier
    FNR. It reads 0.0 in C2 only because *everything* was blocked, which happens to make the pass-through
    rate and the judge-unsafe-subset FNR collapse to the same value. The label should not be read as a
    validated recall; the coincidence is an artifact of blanket blocking.
  - **The derived ~100% recall is trivially perfect and uninformative.** `1 − ASR_C2/ASR_C1 = 1.0` on
    both suites is a ratio of two point ASRs with no propagated CI, and it is perfect *because* the
    screen blocks indiscriminately — the inverse of C3, where the derived ~82% recall on a
    non-fully-blocked suite was the informative headline. Do not quote C2 recall as evidence of selective
    containment.
  - **`guardrail_fpr` has no per-category segments in the artifact.** The 0.336 XSTest FPR is
    whole-suite only; the segmented numbers reported are the *model's own* block-blind `over_refusal`
    (`privacy_fictional` 0.12 n=25; `historical_events` / `privacy_public` / `safe_contexts` 0.04, n=25
    each; the other six categories 0.0) — **not** the guardrail FPR. The two must not be conflated: the
    over-refusal segments describe pre-block model behavior, which C2's input block bypasses entirely.
  - **The harmful suites are overtly harmful, so 100% block is unsurprising.** AdvBench/HarmBench are
    single-turn, blunt, direct-attack prompts (and AdvBench contains near-duplicates, so effective
    n < 520). A prompt screen catching 100% of these is the expected easy case; it says little about
    borderline or dual-use inputs, and nothing about adversarial wrappers or multi-turn escalation, all
    of which are unmeasured.
  - **Everything is classifier-vs-classifier agreement on an unaudited single-turn proxy.** ASR keys on
    the `Llama-Guard-3-1B` label and `blocked_at`; the guardrail is the distinct Granite Guardian 2B
    (ADR-0009 decision 2, ADR-0004 rule 4). Both are classifiers — a *proxy* for true material
    compliance, single-turn only — so the 100% block is guardrail-flags-prompt agreement, not verified
    prevented harm.
  - **Inherited ADR-0008 judge caveats bind C2.** The safety judge is an uncalibrated 1B proxy scoring
    policy-category content with no positive control that it scores the *response* rather than the
    *prompt* (the v1 empty-conversation failure class); helpfulness is self-judged by `Mistral-7B` at
    the policy rev; the `is_refusal` heuristic underlies both `over_refusal` and `answer_rate`; and
    judge calibration against a 100–300 human-audited sample (ADR-0004 rule 7) is deferred to Phase 9.
    Every C2 number inherits these.
  - **Degenerate zero-width bootstrap CIs persist and are worse here.** The AdvBench/HarmBench ASR
    0.0 [0.0, 0.0] and the six per-category 0.0 intervals are all zero-width because the resampled
    indicators are all-identical, and the "0 → 0.336" XSTest rise is measured against a *tautological*
    zero-width C1 interval. These zero-width intervals understate uncertainty near the 0/1 boundaries;
    Wilson / Clopper–Pearson intervals are better calibrated there (ADR-0008 already flags this).
  - **The safety/latency tradeoff is not characterized.** The ADR-0009 decision 1 per-component-overhead
    deliverable is absent from these metric JSONs. C2's cost is captured here as over-refusal (FPR) only,
    not runtime overhead — though an input screen that short-circuits generation may *reduce* mean
    latency on blocked prompts, which the current artifacts do not measure either.

## Alternatives considered

- **Headline "100% harmful recall" / "ASR → 0" as the C2 finding** — the number the artifact emits most
  visibly. Rejected: with cache-hit generations and a frozen judge the sign is mechanically enforced,
  the block even short-circuits the judge on the harmful suites, and the recall is trivially perfect
  *because* the screen blocks indiscriminately. The informative quantity is the precision (false-positive)
  cost, not the recall.
- **KEEP the input guardrail standalone, mirroring the C3 KEEP** — treat ASR → 0 as decisive. Rejected:
  C2's benign cost is an order of magnitude higher than C3's (0.336 vs 0.028 on XSTest, plus unmeasured
  Alpaca blocks), so input-only over-blocks too aggressively to adopt alone on these suites. The honest
  move is to carry C2 as a component into C4 and let the composition decide.
- **Treat C2 vs C3 as a rigorous H3 result** — read the complementary profiles as confirming that output
  guardrails catch what input guardrails miss. Rejected: H3's claim is directional and about dual-use
  prompts, and on the overtly-harmful suites present the direction runs the other way (input caught 100%
  vs output ~82%) while the dual-use class H3 turns on is absent from the four suites (ADR-0009 decision
  7); the contrast fixes only that the two guardrails sit at different operating points — it is not
  evidence for H3, and C4 plus a dual-use suite are required.
- **Quote the block-blind `over_refusal` (0.024) or "unchanged" Alpaca helpfulness as the benign cost** —
  the numbers the artifact reports directly. Rejected: both are scored on cached pre-block text and hide
  the input blocks; the true XSTest refusal cost is the 0.336 guardrail block, and the Alpaca "no change"
  is mechanically guaranteed while the suite suffered confirmed nonzero benign blocks.
- **Block any C2 conclusion pending judge calibration (Phase 9)** — wait for a human-audited κ. Rejected
  on the same grounds as ADR-0008/0010: the harmful block is 100% and the benign-cost finding rests on
  the `guardrail_fpr` and the provenance hash, not the judge; calibration (ADR-0004 rule 7) is a
  scheduled follow-up, not a Phase-2 blocker, and the caveat is recorded and bounds interpretation.
- **Regenerate-or-refuse on an input block instead of the canned safe-refusal** — closer to serving-time
  behavior. Rejected here per ADR-0009 decision 5: ASR / FPR / FNR key on the block *decision*, not the
  refusal text, so a fixed template is exactly what the ablation measures; regenerate-or-refuse is
  deferred to the Phase-8 serving benchmark.

## Follow-ups

1. **Measure the benign Alpaca block directly** (elevates ADR-0010 follow-up 2, now the top priority
   because C2's differing `provenance_hash` makes it bite). Add a first-class `guardrail_fpr` (block
   rate) on the helpfulness / Alpaca suite and make `benign_helpfulness` / `answer_rate` fold in
   `blocked_at` (or report a parallel post-block helpfulness), so the confirmed-nonzero Alpaca block cost
   is *quantified* rather than inferred from a differing `provenance_hash`. This closes the single biggest
   gap in the C2 precision headline.
2. **Emit `guardrail_fpr` per-category on XSTest** so the 0.336 block is attributable to prompt classes,
   and stop reporting the block-blind `over_refusal` segments as if they described guardrail behavior.
3. **Run C4 (input+output) and construct a dual-use suite** (benign-looking prompt whose generation is
   unsafe) to move on H3; C2 vs C3 sits at different operating points but does not test H3 on the
   overtly-harmful current suites (the direction there is, if anything, contrary to H3), so it cannot
   separate input- from output-guardrail value (ADR-0009 decision 7, ADR-0010 follow-up 3).
4. **Report the true user-facing refusal rate on XSTest as the union** of the guardrail block (0.336)
   and the model-own `over_refusal` (0.024), rather than either alone; and rename/relabel `guardrail_fnr`
   so its C2 0.0 is not misread as a validated recall (ADR-0010 follow-up 1).
5. **Human-calibrate both classifiers** — the `Llama-Guard-3-1B` judge and the Granite Guardian 2B
   guardrail — against a 100–300 audited sample (ADR-0004 rule 7 / Phase 9).
6. **Produce the deferred ADR-0009 latency / per-component-overhead report,** including the input
   screen's *latency saving* on short-circuited prompts alongside its over-refusal cost.
7. **Report Wilson / Clopper–Pearson intervals for the genuinely near-0/near-1 proportions** — the
   zero-width harmful ASRs, the six 0.0 per-category segments, and the tautological C1 0.0 FPR — rather
   than the degenerate bootstrap intervals. (The 0.336 XSTest FPR is a mid-range proportion whose
   bootstrap interval [0.28, 0.396] is already well-behaved and does not need the boundary-calibration
   fix.)