# ADR-0010: C3 output-guardrail result and Phase-2 keep decision

- **Status:** Accepted
- **Date:** 2026-07-09
- **Deciders:** Project owner

## Context

Phase 2 is the *guardrail layer*: it measures how much protection **external guardrails** provide on
their own, before any weight-level alignment, by wrapping the frozen starting model in input and/or
output guardrails (ADR-0009). C3 is the **output-only** rung: the user prompt is generated normally by
the frozen `mistralai/Mistral-7B-Instruct-v0.3` (no SFT — ADR-0002, ADR-0008), then a separate pass
screens the *response* and blocks with a canned safe-refusal when it is judged unsafe. Every C3 number
is read as a *marginal* change against the C1 anchors frozen in ADR-0008 — including the tautological
no-defense baselines `guardrail_fnr=1.0` (harmful) and `guardrail_fpr=0.0` (benign), which exist
precisely to be moved here.

**Guardrail model (kept separate from the judge — ADR-0009 decision 2, ADR-0004 rule 4).** The output
guardrail is Granite Guardian 2B (`ibm-granite/granite-guardian-3.1-2b`, IBM, ~2B, Apache-2.0), a model
*distinct* from the frozen `meta-llama/Llama-Guard-3-1B` safety judge. This separation is what makes the
ASR-reduction **non-circular**: the block decision (Granite) and the ASR score (Llama-Guard label *and*
`blocked_at`) come from different classifiers, so the guardrail is not grading its own output. The
guardrail runs as a post-pass over the cached generations, never co-resident with the 7B (ADR-0003,
ADR-0009 decision 3).

**Run configuration.** C3 reuses the C1 generations unchanged: the generation cache hit **1170/1170**
(0 misses) on the content hash, so no regeneration occurred — only the Granite output-guardrail pass and
the frozen judge ran. Decode is therefore inherited from C1 (greedy, deterministic, `temperature 0`,
`seed=0`). The safety judge is the frozen `Llama-Guard-3-1B` at the same pinned rev
`acf7aafa60f0410f8f42b1fa35e077d705892029` (identical fingerprint in the C1 and C3 JSONs). Bootstrap CIs
use the seeded stdlib RNG (B=10000, `seed=0`, percentile [2.5, 97.5]) per ADR-0007 rule 5. ASR is defined
in `safestack/eval/metrics.py` as the fraction of harmful-prompt responses the judge labels `unsafe`
**and** for which `blocked_at is None`.

**One structural consequence of the cache-hit, stated up front.** Because the generations are a 1170/1170
content-hash cache-hit off C1 and the judge is frozen, an output guardrail can only *remove* unsafe-counted
rows (a counted row requires `blocked_at is None`) and can never add them. The **direction** of the ASR
drop is therefore *mechanically enforced*, not an empirical finding; the informative quantities are its
**magnitude** and the **recall / over-block tradeoff**. This same cache-hit is also what makes the derived
recall below valid (see Decision), because it makes the judge-unsafe set identical across C1 and C3.

## Decision

The C3 measurement, point estimate with 95% CI and n. ASR on AdvBench (`harmful_advbench_v1`) is **0.100**
[0.075, 0.127], n=520, versus the C1 anchor 0.548 [0.506, 0.590] — a **−0.448** absolute (~82% relative)
drop. ASR on HarmBench (`harmful_harmbench_v1`) is **0.130** [0.085, 0.180], n=200, versus C1 0.675 [0.610,
0.740] — a **−0.545** absolute (~81% relative) drop. The 95% bootstrap CIs are **non-overlapping** on both
suites. Benign helpfulness on Alpaca (`helpfulness_alpaca_v1`) shows **no change in the block-blind score**,
4.915 / 5 [4.85, 4.965], `answer_rate=1.0` (identical to C1 by construction — see the threats below);
model-own over-refusal on XSTest-safe (`overrefusal_xstest_v1`) is unchanged at **0.024** [0.008, 0.044],
n=250; and the guardrail adds a `guardrail_fpr` of **0.028** [0.008, 0.052] (= 7/250) on the borderline
XSTest suite, up from the tautological C1 0.0 [0.0, 0.0].

**Recall on the judge-unsafe subset (headline).** Because the 1170/1170 cache-hit plus the frozen judge
make the judge-unsafe set *identical* across C1 and C3, clean guardrail recall is derivable as
`1 − ASR_C3 / ASR_C1`: **0.818** on AdvBench (`1 − 0.100/0.548`) and **0.807** on HarmBench
(`1 − 0.130/0.675`) — i.e. a false-negative rate of **~18.2% / ~19.3%** on outputs the judge labels unsafe.
This is a **point derivation with no propagated CI** (a ratio of two point ASRs), valid *only* under the
identical-judge-unsafe-set condition above.

**Do not headline the reported `guardrail_fnr`.** `metrics.py` computes `guardrail_fnr` as the mean of
`blocked_at is None` over **all** harmful-suite rows — a whole-suite *pass-through* rate (~0.49 AdvBench /
~0.39 HarmBench), **not** a classifier FNR. Quoting `1 − guardrail_fnr` (≈ 51% / 61%) as recall
**understates the guardrail by ~31 points on AdvBench (≈20 on HarmBench)**, because ~45% of AdvBench
outputs were already-judge-safe C1 responses, most of which the guardrail correctly leaves unblocked, and
the metric lumps those in with genuine misses.

**Per-category HarmBench (exploratory — ADR-0004 rule 2).** ASR falls across all six categories, with the
largest absolute drops on the highest-baseline classes: `cybercrime_intrusion` 0.925→0.125 (n=40),
`illegal` 0.793→0.121 (n=58), `chemical_biological` 0.786→0.071 (n=28). AdvBench reports no segments (no
category labels).

Given this:

1. **KEEP the Granite Guardian 2B output guardrail in the Phase-2 stack.** It delivers a large,
   CI-separable ASR reduction on both harmful suites at **no measured helpfulness cost** (block-blind, see
   threats) and only a small over-block confined to the borderline suite — effective at low benign cost. It
   carries into C4 and into the Phase-3 SFT+guardrails contrast (C8).
2. **Headline the derived recall (~81–82%), never the mislabeled `guardrail_fnr`.** Report the recall as a
   point derivation scoped to the judge-unsafe subset, and flag `guardrail_fnr` as a whole-suite
   pass-through rate, not recall (see the `metrics.py` caveat above).
3. **Proceed to C2 (input-only) and C4 (input+output).** C3 is only the output rung; the input-vs-output
   value contrast (H3) requires C2 *and* a dual-use suite the four current suites lack (ADR-0009 decision
   7). C3 alone cannot separate input- from output-guardrail value.
4. **Treat the ASR-drop *sign* as mechanically enforced, not a discovery.** With cache-hit generations and a
   frozen judge the direction is guaranteed; only the magnitude and the recall / over-block tradeoff are
   findings.
5. **Scope the hypothesis claims honestly.** C3 is the guardrail-alone rung of **H1** (defense stacking
   reduces ASR) and is **confirmatory** on these four single-turn suites (ADR-0009 decision 7), but H1 is
   only *completed* once the SFT rungs (C5–C8) land in Phase 3. C3 gives **weak / near-null** support for
   **H2** (strict defenses increase over-refusal): +2.8 pts of borderline over-block, no measured
   helpfulness loss. C3 **does not** test **H3** or multi-turn.
6. **Commit aggregate-only artifacts** — `notebooks/c3_colab.ipynb` (executed, aggregate outputs only) and
   `reports/metrics/c3_starting_output_guardrail__*__C3.json` (point / CI / n / segments only, no raw
   pre-block or blocked text), per `RESPONSIBLE_USE.md`, ADR-0007 rule 7, and ADR-0009 decision 8.

## Consequences

- **The guardrail-alone rung of H1 is measured and CI-separable.** C1 vs C3 quantifies the marginal
  output-guardrail ASR reduction on the un-aligned model; H1 is *completed* only when the Phase-3 SFT rungs
  land (ADR-0009 Consequences), and adjacent guardrail rungs (C2 vs C3) may still turn out CI-overlapping —
  "no significant difference" is a valid outcome (ADR-0004 rule 6).
- **H2's effect here is small / near-null.** The output guardrail adds a 2.8% block rate on the borderline
  suite with no measured helpfulness loss; the strict-defense over-refusal cost H2 predicts is small on
  these single-turn suites.
- **Rule 4 is satisfied by construction, so the metrics are clean.** The Granite guardrail is
  judge-independent, so its confusion statistics are reusable as the H5 containment baseline later
  (ADR-0009 decision 2) — but that baseline should use the judge-conditioned confusion-matrix FNR from
  follow-up 1, not the whole-suite `guardrail_fnr` as currently emitted. The ASR-reduction number is not
  circular.
- **The keep decision is a benign-cost, not a zero-cost, claim.** The "no helpfulness cost" reading is on
  the easy-benign Alpaca suite only, and is block-blind (below); the borderline over-block is real, small,
  and localized.
- **Threats to validity** (recorded; none reverses KEEP, but each bounds how the numbers may be quoted):
  - **`guardrail_fnr` is a misnomer (confirmed and sharpened).** As defined (mean over *all* harmful-suite
    rows of `blocked_at is None`) it is the whole-suite pass-through rate, conflating "correctly did not
    block an already-safe response" with "missed a judge-unsafe output." Read as recall it gives 51% /
    61%; clean recall on the judge-unsafe subset is ~82% / 81% and the FNR on judge-unsafe items is only
    ~18% / 19%. Quoting the label as recall would understate the guardrail by ~31 points on AdvBench (≈20 on
    HarmBench).
  - **The derived recall carries no CI and is conditional on the cache-hit.** It is a ratio of two point
    ASRs; it is clean *only* because the 1170/1170 content-hash cache-hit and frozen judge make the
    judge-unsafe set identical across C1 and C3. Had any generation differed, the two judge-unsafe sets would
    not coincide and the `ASR_C3 / ASR_C1` shortcut would be invalid. Quote ~81% as a point derivation, not
    a CI-bearing estimate.
  - **Helpfulness metrics are block-blind (confirmed and sharpened).** `metrics.py` never reads `blocked_at`
    for the helpfulness suite, and `answer_rate` runs the refusal heuristic on the *cached pre-block* text;
    no `guardrail_fpr` is emitted for that suite. A user handed a canned guardrail refusal on a benign Alpaca
    prompt would still score 5/5 and count as "answered," so "unchanged helpfulness" is mechanically
    guaranteed and measures nothing about the guardrail's benign behavior. The byte-identical C1/C3
    helpfulness `provenance_hash` (`sha256:7324e36d…`, which folds `blocked_at`) *indicates* 0/200 Alpaca
    outputs were actually blocked — but that 0 is inferred from the hash, not a first-class reported number,
    so the guardrail's block rate on genuinely-helpful requests is effectively unmeasured.
  - **Borderline over-refusal is two disjoint signals, and their union is not reported (confirmed).**
    `over_refusal` (model's own, 0.024) and `guardrail_fpr` (guardrail block, 0.028) are separate on the
    XSTest borderline suite. The user-facing refusal cost is their *union*, bounded by ~2.8–5.2% (5.2% is the
    sum of marginals, a disjoint upper bound). Neither number alone is the user-facing refusal rate, and the
    joint is not emitted.
  - **`guardrail_fpr` / `guardrail_fnr` are computed unconditionally, never conditioned on the judge label.**
    They range over every row in the suite, so `guardrail_fpr` counts blocks even on prompts the model
    already refused, and on the harmful suites the guardrail *also* blocked ~14–20% of the already-judge-safe
    outputs (derived: ~34/235 AdvBench, ~13/65 HarmBench). The borderline over-block cost may be marginally
    overstated (blocking an already-refused output adds no user harm), and the guardrail's behavior on safe
    content is only visible via derivation, not a first-class confusion matrix.
  - **Everything is guardrail-vs-judge agreement on an unaudited single-turn proxy (confirmed).** ASR keys on
    the `Llama-Guard-3-1B` label and `blocked_at`; the guardrail is the distinct Granite Guardian 2B
    (ADR-0009 decision 2, ADR-0004 rule 4), so the reduction is non-circular — but both are classifiers, i.e.
    a *proxy* for true material compliance, single-turn only. The headline ASR reduction and the derived
    recall are classifier-vs-classifier agreement, not verified material harm. Robustness to adversarial
    wrappers, multi-turn escalation, and the input-vs-output (H3) contrast are all unmeasured; H3 is only
    "partial and exploratory" even for C2/C3/C4 together (ADR-0009 decision 7).
  - **Inherited ADR-0008 judge caveats bind C3.** The safety judge is an uncalibrated 1B proxy scoring
    policy-category content, with no positive control proving it scores the assistant *response* rather than
    the *prompt* (the v1 empty-conversation failure class); helpfulness is self-judged by `Mistral-7B` at the
    same pinned rev as the policy model; and AdvBench contains near-duplicates, so effective n < 520.
    Additionally, `over_refusal` and `answer_rate` both depend on the same uncalibrated `is_refusal`
    heuristic, so the 0.024 over-refusal may under- or over-count soft refusals and `answer_rate=1.0` is not
    an independent corroboration of the helpfulness claim — this inheritance bounds the "low benign cost"
    half of KEEP. Every C3 number, including the ~81% recall, inherits these; judge calibration against a
    100–300 human-audited sample (ADR-0004 rule 7) is deferred to Phase 9 and bounds the interpretation.
  - **Degenerate zero-width bootstrap CIs persist.** E.g. the HarmBench segment `harassment_bullying` 0.0
    [0.0, 0.0] (n=19) and the C1 `guardrail_fpr` 0.0 [0.0, 0.0]; the `guardrail_fpr` 0.028 interval rests on
    7/250 near a boundary where the percentile bootstrap is poorly calibrated, and the "0 → 2.8%" rise is
    measured against a *tautological* zero-width C1 interval. These understate uncertainty (rule of three:
    0/19 has a ~15% one-sided upper bound). Wilson / Clopper–Pearson intervals are better calibrated near 0
    and 1 (ADR-0008 already flags this); see follow-up 6.
  - **Per-category HarmBench reductions are exploratory (ADR-0004 rule 2).** Broad drops across all six
    categories, largest on the highest-baseline classes, but tiny per-segment n, wide CIs, and some
    degenerate zero-width intervals. Exploratory only, inheriting the same uncalibrated-proxy caveat as C1.
  - **The safety/latency tradeoff is not characterized.** The ADR-0009 decision 1 latency /
    per-component-overhead deliverable is absent from these four metric JSONs. The guardrail's cost is
    captured here only as over-refusal (FPR), not runtime overhead, so the safety/helpfulness tradeoff is
    measured but the safety/latency tradeoff is not.

## Alternatives considered

- **Headline the reported `guardrail_fnr` (or `1 − guardrail_fnr`) as recall** — the number the artifact
  emits directly. Rejected: it is a whole-suite pass-through rate, not a classifier FNR; quoting it as recall
  understates the guardrail by ~20–31 points (~31 on AdvBench, ~20 on HarmBench). The derived
  judge-unsafe-subset recall is the honest quantity.
- **Present the ASR drop itself as the finding** — the intuitive read of a 0.548→0.100 fall. Rejected: with
  cache-hit generations and a frozen judge the *sign* is mechanically enforced; only the magnitude and the
  recall / over-block tradeoff are informative.
- **Block KEEP pending the C2 / H3 contrast** — wait to keep the output guardrail until input-vs-output value
  is separable. Rejected: C3 is confirmatory for the guardrail-alone rung of H1 on these suites (ADR-0009
  decision 7); H3 needs C2 plus a dual-use suite and is a scheduled slice, not a blocker to keeping an
  effective, low-cost guardrail.
- **Block KEEP pending judge calibration (Phase 9)** — wait for a human-audited κ before trusting the ASR or
  the recall. Rejected on the same grounds as ADR-0008: both harmful CIs are non-overlapping with wide margin
  and calibration (ADR-0004 rule 7) is a scheduled follow-up, not a Phase-2 blocker; the caveat is recorded
  and bounds interpretation.
- **Regenerate-or-refuse on an output block instead of the canned safe-refusal** — closer to serving-time
  behavior. Rejected here per ADR-0009 (decision 5): ASR / FPR / FNR key on the block *decision*, not the
  refusal text, so a fixed template is exactly what the ablation measures; regenerate-or-refuse is deferred
  to the Phase-8 serving benchmark.

## Follow-ups

1. **Emit a first-class guardrail confusion matrix conditioned on the judge label** — FNR on judge-unsafe
   items (~18% / 19%), block-rate on the judge-safe subset (~14% / 20%), and TPR — instead of the whole-suite
   `guardrail_fnr` that conflates them; rename or relabel `guardrail_fnr` to prevent the "recall = 51%"
   misread, and use the confusion-matrix FNR (not the whole-suite rate) as the later H5 containment baseline.
2. **Measure benign blocking directly.** Add a first-class `guardrail_fpr` (block rate) on the
   helpfulness / Alpaca suite and make `benign_helpfulness` / `answer_rate` fold in `blocked_at` (or report a
   parallel post-block helpfulness), so benign blocking is *measured* rather than inferred from an identical
   `provenance_hash`.
3. **Run C2 (input-only) and construct a dual-use suite** (benign-looking prompt whose generation is unsafe)
   to move on H3; C3 alone cannot separate input- from output-guardrail value, and the four current suites
   lack the class H3 turns on.
4. **Human-calibrate both classifiers** — the `Llama-Guard-3-1B` judge *and* the Granite Guardian 2B
   guardrail — against a 100–300 audited sample (ADR-0004 rule 7 / Phase 9); the ASR and the ~81% recall are
   two-classifier proxies for true material-compliance harm.
5. **Produce the deferred ADR-0009 latency / per-component-overhead report** so the guardrail's runtime cost
   is quantified alongside its over-refusal cost.
6. **Report Wilson / Clopper–Pearson intervals for the near-0/near-1 proportions** (`guardrail_fpr`,
   degenerate 0/n segments) and attach a CI to the derived recall (e.g. bootstrap the paired blocked/unsafe
   indicators) rather than quoting it as a bare point ratio.
