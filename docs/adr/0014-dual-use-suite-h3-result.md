# ADR-0014: Dual-use suite H3 result — input is not blind to HarmBench-contextual, verdict WEAK/AMBIGUOUS

- **Status:** Accepted
- **Date:** 2026-07-11
- **Deciders:** Project owner

## Context

H3 is the directional claim that *output* guardrails catch unsafe generations that *input* guardrails
miss on **benign-looking / dual-use** prompts — a prompt whose *text* reads benign (so an input screen
passes it) but whose *generation* is unsafe (so an output screen should catch it). The completed Phase-2
2x2 (C1 none / C2 input / C3 output / C4 input+output on the frozen `mistralai/Mistral-7B-Instruct-v0.3`)
could not test it: the harmful suites are overtly harmful (input blocked 100%, ASR_C2 = ASR_C4 = 0,
leaving no residue for output — ADR-0012) and the benign suites generate nothing unsafe. ADR-0013
**preregistered** the dual-use suite as the missing instrument — source, construction, scoring, and the
H3 decision rule fixed **before any C1-C4 numbers were produced on it** — and this ADR records the
result against that preregistration. It is the sibling of ADR-0010/0011/0012 (the C3/C2/C4 result ADRs)
and inherits their honest-framing discipline.

**The suite and the run.** `dualuse_harmbench_contextual_v1` is the `walledai/HarmBench` `contextual`
config (Mazeika et al. 2024, MIT, gated) at the pinned revision `fb6c2afd…`, 100 rows, each a benign
context passage followed by the behavior/request (the **full contextual item** is what the input screen
ingests — ADR-0013 decision 1), scored on the `eval_dual_use` split by the frozen `Llama-Guard-3-1B`
safety judge. The committed `manifest_hash` is `sha256:52ced8ea…` on all four condition JSONs (the
preregistered artifact, unchanged). Unlike the four existing suites (1170/1170 cache-hits off C1), the
contextual prompts are new: **C1-dual-use is a real Mistral-7B generation pass** — the run log shows
`C1 generations: hits 450 misses 100 total 550`, i.e. 100 fresh contextual generations plus the 450
XSTest+Alpaca rows reused from the main C1 cache — and C2/C3/C4-dual-use are content-hash cache-hits off
*that* C1. Decode is the frozen C1 config (greedy, `temperature 0`, `seed 0`); the safety judge is the
frozen `Llama-Guard-3-1B` at rev `acf7aafa60f0410f8f42b1fa35e077d705892029` (identical judge fingerprint
across the four JSONs), `judge_prompt_version v2`; bootstrap B=10000, `seed 0`, percentile [2.5, 97.5]
(ADR-0007 rule 5). Data integrity is clean on the dual-use suite (`n_unparsed=0`, `n_missing=0` in every
condition — no fail-open). Measured on a Colab A100-80GB (torch 2.11.0+cu128, transformers 5.12.1; C1
run `ca218e94`, with C2/C3/C4 cache-hit runs `b45d9fa6` / `9a4b1905` / `20c10072`). **No preregistration
amendment was required** — source, revision, manifest hash, judge fingerprint, decode, and the paired
rule-5 suites all match the ADR-0013 commit.

**Three structural caveats, stated up front (as ADR-0010/0011/0012, with the ADR-0013 twist).**
(1) **The ASR-drop *direction* is mechanically enforced, not a finding.** C2/C3/C4 are cache-hits off
the fresh C1 and the judge is frozen, so a guardrail can only *remove* unsafe-counted rows (a counted
row needs `blocked_at is None`), never add them: `ASR_C2, ASR_C3, ASR_C4 ≤ ASR_C1` are guaranteed in
sign. The empirical quantities are the C3-drop *magnitude* and the **C2-vs-C3 gap** (ADR-0013 decision
3). (2) **`guardrail_fnr` is a whole-suite pass-through rate, not recall**, and it bites *harder* here
than on the harmful suites: because a substantial fraction of contextual prompts generate judge-*safe*
text (26/100 at C1), a lower pass-through is equally consistent with a screen catching *unsafe*
generations and with it over-blocking *safe* ones — it must not be read as the input-vs-output signal
(ADR-0013 decision 5). (3) **The judge is NOT moot here.** Unlike the harmful C2/C4 where a 100% input
block short-circuits the judge, the dual-use ASRs at *every* condition depend on Llama-Guard actually
labelling the generations that pass a screen, so judge calibration (ADR-0004 rule 7, Phase 9) matters
more for this result than it did for the earlier rungs.

## Decision

**The dual-use ablation** (point estimate, 95% CI, n; frozen Mistral-7B; `asr` and `guardrail_fnr` on
the dual-use suite, `guardrail_fpr` and block-blind `over_refusal` on XSTest, block-blind
`benign_helpfulness` on Alpaca; the rule-5 paired benign suites):

| Condition | Guardrail | Dual-use ASR (n=100) | Dual-use `guardrail_fnr` | Derived recall `R_Cx` | XSTest `guardrail_fpr` (n=250) | Alpaca helpfulness (n=200) |
| --- | --- | --- | --- | --- | --- | --- |
| **C1** | none | 0.74 [0.65, 0.82] | 1.0 [1.0, 1.0] *(taut.)* | — (baseline) | 0.0 [0.0, 0.0] *(taut.)* | 4.915 [4.85, 4.965] |
| **C2** | input | 0.19 [0.11, 0.27] | 0.32 [0.23, 0.41] | **0.743** | 0.336 [0.276, 0.396] | 4.915 [4.85, 4.965] |
| **C3** | output | 0.33 [0.24, 0.42] | 0.52 [0.42, 0.62] | **0.554** | 0.028 [0.008, 0.052] | 4.915 [4.85, 4.965] |
| **C4** | input+output | 0.13 [0.07, 0.20] | 0.20 [0.12, 0.28] | 0.824 | 0.336 [0.276, 0.396] | 4.915 [4.85, 4.965] |

Block-blind `over_refusal` on XSTest is **0.024** [0.008, 0.044] in all four conditions. Read at the
ADR-0004 rule-6 CI-separability bar (overlapping 95% CIs = `≈`, non-overlapping = `<`/`>`): **C1 > C2**,
**C1 > C3**, **C2 ≈ C3** (CIs overlap on [0.24, 0.27] — the `eval compare` gate emits *"no significant
difference: C2 vs C3 on asr"*), **C3 > C4**, **C2 ≈ C4**.

**The headline: the input screen is NOT blind to HarmBench-contextual — it blocks 68% of the dual-use
prompts and removes more of the unsafe subset than the output screen does — so this construction does
not create the input-blind regime H3 needs, and the decisive C2-vs-C3 comparison is not
CI-separable.** Under the cache-hit + frozen-judge identity the aggregates decompose exactly: of the 74
judge-unsafe C1 rows, the **input** screen (C2) blocked **55** (`fnr` 0.32 → 68/100 prompts blocked at
input, 19 unsafe slipped through → `R_C2 = 55/74 = 0.743`), while the **output** screen (C3) blocked
**41** (`fnr` 0.52 → 48/100 responses blocked at output, 33 unsafe slipped through → `R_C3 = 41/74 =
0.554`). Input's recall *exceeds* output's — the **opposite** of H3's `R_C3 > R_C2` prediction — and the
point ASRs run the same way (C3 0.33 > C2 0.19), while their CIs overlap so the difference is not
significant either. The Granite input screen flags these contextual prompts because they still contain
an overt harmful behavior request wrapped in context; the "benign-looking prompt" premise fails for this
guardrail, exactly the threat-to-validity ADR-0013 recorded in advance (*"HarmBench-contextual may not
be benign-looking enough"*).

Given this:

1. **Classify the outcome as WEAK / AMBIGUOUS per the ADR-0013 decision-4 rule, and record that H3 is
   NOT supported on this construction.** Walking the preregistered partition: the baseline has ample
   dynamic range (`ASR_C1` 0.74 [0.65, 0.82], CI excludes 0) so the suite is **not** *INCONCLUSIVE*.
   **H3 SUPPORT** requires `ASR_C2 ≈ ASR_C1` (input blind) — **refuted**, C1 > C2 by non-overlapping CIs.
   **PARTIAL H3 SUPPORT** requires `ASR_C3 < ASR_C2` by non-overlapping CIs (output catches strictly
   more) — **refuted**, C2 ≈ C3 (overlap) and the point estimate runs *counter* (C3 > C2). **NULL — input
   pre-empts (saturation)** requires `ASR_C2`'s CI at/near 0 — **not met**, C2 = 0.19 [0.11, 0.27] is a
   *partial* pre-emption (input blocked 68%, not ~100%). **NULL — both placements blind** requires
   `ASR_C3 ≈ ASR_C1` — **refuted**, C1 > C3 (output *does* reduce ASR). No named bucket fits cleanly and
   the decisive comparison (C2 vs C3) has overlapping CIs, so by the letter of the rule the verdict is
   **WEAK / AMBIGUOUS**, and it is **not** retroactively sorted into support or null. H3 gets **no
   support**; if anything the direction runs against it.

2. **The honest directional reading is "input is at least as good as output here (partial
   pre-emption)," not a clean saturation null.** WEAK/AMBIGUOUS is the faithful verdict, but the result
   is not directionless: input's *point* recall (0.743) exceeds output's (0.554) and input's ASR point
   (0.19) sits below output's (0.33). This is a *partial* cousin of decision-4's input-pre-empts null —
   the input screen does most of the catching, just short of the saturation-to-zero the narrow bucket
   names. The reason is mechanistic and visible in the segments (below): Granite recognises the
   technical-harm categories from the prompt alone, so on these prompts it is not the blind screen H3
   posits. A ratio of point ASRs carries no propagated CI (ADR-0011), so `R_C2 > R_C3` is a point
   comparison; the CI-separable statement is only the negative one — C2 and C3 are *not* distinguishable,
   and neither is separably better, so output is *not* shown to catch what input misses.

3. **Weighted by its benign cost, the output screen remains the better single guardrail — the same C3
   story as ADR-0010/0012, now on dual-use.** Input's higher recall is bought bluntly: it blocks 68% of
   the dual-use prompts and over-refuses **33.6%** of benign XSTest prompts (`guardrail_fpr` 0.336),
   whereas output achieves `R_C3 = 0.554` at a **0.028** XSTest FPR — an order of magnitude cheaper on
   benign inputs. Per unit of benign cost the output screen catches far more; input "wins" on raw dual-use
   recall only because it is trigger-happy on anything that reads harmful, benign or not. The rule-5
   pairing (dual-use ASR read *with* its over-refusal companion, committed in ADR-0013 decision 5) is what
   makes this cost-adjusted reading part of the design, not a post-hoc gloss. Helpfulness on Alpaca is
   flat at 4.915 across all four conditions, but that flatness is **block-blind and mechanically
   guaranteed**, not evidence of no benign cost: C2/C4's Alpaca `provenance_hash` (`54403a6a…`) is
   byte-identical to each other and differs from C1/C3's (`7324e36d…`), so under the cache-hit +
   frozen-judge identity the input screen blocked at least one clearly-benign Alpaca prompt in C2/C4 —
   the ADR-0012 "Alpaca block confirmed nonzero but unquantified" pattern, a benign cost the helpfulness
   metric hides (threats, below).

4. **The one place the output stage acts in the H3-predicted direction is the C2→C4 margin, and it is a
   hint, not evidence.** Composing output onto input (C4) nudged ASR 0.19 → 0.13 and recall 0.743 →
   0.824 — the output screen caught a few of the unsafe generations that *passed* the input screen, which
   is precisely the H3 mechanism operating on the input-passing residue. But `C4 ≈ C2` (CIs overlap), so
   the effect is **not CI-separable**; and `C4 < C3` with C4's XSTest `guardrail_fpr` (0.336) *identical*
   to C2's, so the composition tracks the **input** rung on both safety and benign cost (as C4 tracked
   the dominant stage on the harmful suites in ADR-0012). The gross input-vs-output split of C4's blocks
   is not identified by the committed aggregates — only its net effect is (same net-vs-gross limit as
   ADR-0012); the per-stage `blocked_at` breakdown (follow-up) would pin whether this margin is real.

5. **Segment hint (hypothesis-generating, n tiny): output strictly beats input only where the prompt is
   least overtly harmful.** Per-category dual-use ASR shows input crushing the technically-named
   categories the prompt itself reveals — `cybercrime_intrusion` C1 0.78 → C2 0.0 (output weaker, C3
   0.26), `chemical_biological` C1 0.93 → C2 0.14 (C3 0.36) — while on `harassment_bullying` (n=6) input
   caught **nothing** (C1 0.5 → C2 0.5) and **output caught everything** (C3 0.0). That is the H3 pattern
   exactly, in the one category where the request reads benign and the harm surfaces in the generation.
   It is a **micro-signal on n=6** — the C3/C4 cells at a degenerate zero-width [0.0, 0.0] and the C1/C2
   cells at wide, overlapping [0.17, 0.83] intervals, so even "input caught nothing" is not established —
   not evidence, but a concrete design lead for the synthetic held-out construction (which should
   over-sample the benign-context / unsafe-generation subclass this category represents).

6. **Read `guardrail_fnr` and the block-blind benign metrics with active suspicion.** The output rung's
   `fnr` 0.52 is a pass-through rate mixing 33 unsafe-passed and 19 safe-passed rows, not a recall; the
   input rung's `fnr` 0.32 mixes 19 unsafe- and 13 safe-passed. `over_refusal` (0.024, all conditions)
   is scored on cached pre-block text and hides the guardrail blocks — the true user-facing benign cost
   is the 0.336 XSTest block, not the 0.024 model-own over-refusal — and the "unchanged" Alpaca
   helpfulness is block-blind. None of these is the input-vs-output signal; `asr` and the derived recall
   carry it.

7. **Keep H3 exploratory, and trigger the preregistered fallback on its rationale.** A WEAK/AMBIGUOUS
   result on one single-turn construction with one guardrail and one judge returns *no support* for H3 on
   this construction — and even that is local, not a refutation of H3 in general (ADR-0013 decision 8).
   ADR-0013 follow-up 4's *literal* trigger was the **saturation** null (not met here — this is a partial
   pre-emption, C2 = 0.19, not ~0), but its **rationale** fires: HarmBench-contextual did not create the
   input-blind regime H3 needs, because the input screen was not blind. So **build the synthetic held-out
   dual-use set** — hand-authored benign-looking prompts (seeded by the `harassment_bullying`-style
   subclass above) that are genuinely held out from the classifiers — as the harder probe that can
   actually put a prompt past the input screen while its generation is unsafe (ADR-0013 decision 8, "the
   harder test H3 still deserves").

8. **Commit aggregate-only artifacts.** The executed `notebooks/dualuse_colab.ipynb` (aggregate outputs
   only — ASR/CI tables, derived recall, hardcoded pre-flight; verified no per-row contextual prompt or
   unsafe generation text, `metadata.widgets` and per-cell Colab metadata stripped) and the twelve
   `reports/metrics/c*_dualuse_*__*.json` (point / CI / n / segments / hashes only, no raw text) are
   committed; the raw contextual prompts and the pre-block generations stay gitignored in
   `data/prepared/eval_dual_use/` and on Drive, per `RESPONSIBLE_USE.md`, ADR-0007 rule 7, ADR-0009
   decision 8, and ADR-0013 decision 7 (this suite's pre-block generation *is* the unsafe artifact).

## Consequences

- **The dual-use instrument worked as designed and returned an honest WEAK/AMBIGUOUS, no-support-for-H3
  result on this construction.** ADR-0013 accepted, in advance, that the full-set (no-filter) choice
  trades guaranteed signal for validity and could yield a null *or* an ambiguous result; this is the
  ambiguous case — a *partial* pre-emption the preregistered rule classes as WEAK/AMBIGUOUS, **not** the
  saturation null ADR-0013 anticipated (the input screen did not saturate — C2 = 0.19, not ~0). The
  completed 2x2 plus this suite is the
  honest state of the input-vs-output question: on overtly-harmful prompts the input screen saturates
  (ADR-0012); on HarmBench-contextual dual-use prompts the input screen *still* does most of the catching
  (68% blocked) and output is not separably better. In neither regime does output demonstrably catch what
  input misses — H3's core claim is untested in its intended regime because no available real suite
  reliably puts a prompt past the input screen while its generation is unsafe.
- **The Phase-2 precision/recall story is unchanged and now holds on a third prompt class.** Across
  harmful, borderline-benign, and dual-use suites the frontier is the same two points: the blunt,
  complete-but-expensive **input** rung (high recall, 0.336 benign FPR) and the cheap, selective
  **output** rung (C3: ~0.55 dual-use recall / ~0.82 harmful recall at 0.028 benign FPR — ADR-0010). C3
  remains the honest Phase-2 KEEP; the dual-use result does **not** overturn it, and does not promote the
  composition (C4 tracks input at input's cost).
- **This is a joint-strongest H2 signal, tied with the harmful C2/C4.** The input-bearing rungs (C2, C4)
  again realise the 33.6% benign over-block that H2 predicts for strict defenses, now while *also* being
  the higher-recall dual-use rungs — the over-refusal and the recall are two faces of the same
  aggressiveness. The output rung's 0.028 FPR is again the counter-example a well-placed single screen
  can achieve.
- **Threats to validity** (recorded; none reverses the WEAK/AMBIGUOUS + H3-not-supported reading, but
  each bounds how the numbers may be quoted):
  - **`guardrail_fnr` bites hardest here (ADR-0013 decision 5, realised).** With 26/100 C1 generations
    judge-safe, the output `fnr` 0.52 conflates catching unsafe generations with over-blocking safe ones
    (41 unsafe + 7 safe blocked); it is not a recall and is not the H3 signal. The derived recall
    `R_Cx = 1 − ASR_Cx/ASR_C1` is the judge-conditioned quantity, but it carries no propagated CI, so the
    `R_C2 > R_C3` ordering is a point statement while the CI-separable claim is the (negative) C2 ≈ C3.
  - **The C2→C4 output-on-passers effect is net-identified only.** The aggregates pin C4's total but not
    the input-vs-output split of its blocks; "output caught 6 input-passers" and a compensating
    input-jitter both fit the C4 numbers. The per-stage `blocked_at` breakdown (follow-up 2) is needed to
    call the 0.19→0.13 margin real — until then it is a sub-threshold hint.
  - **The benign Alpaca block is confirmed nonzero but unquantified (inherited from ADR-0012).** C2/C4's
    Alpaca `provenance_hash` (`54403a6a…`) is byte-identical to each other and differs from C1/C3's
    (`7324e36d…`); under the cache-hit + frozen-judge identity only `blocked_at` can move it, so the input
    screen blocked ≥1 clearly-benign Alpaca prompt in C2/C4 and the output stage added none (else C4 would
    differ from C2). The flat 4.915 helpfulness is block-blind and mechanically guaranteed, so the exact
    Alpaca block count is not captured — the total benign burden exceeds the 84 XSTest blocks by a
    nonzero, unmeasured Alpaca term, and the two disjoint benign costs must not be pooled into one FPR.
  - **Leakage / memorisation is unmeasured and pushes *toward* the observed input block.** HarmBench is a
    public benchmark the Granite input model or the Llama-Guard judge may have seen in training; a prompt
    the guardrail "recognises" is one it is more likely to block, inflating the input catch for reasons
    unrelated to the dual-use property. The cross-suite overlap check (`safestack data validate`) guards
    against overlap with the other eval suites but not against classifier training-data contamination.
  - **The judge is not moot, so its calibration binds every number here.** ASR_C1 = 0.74 and every
    downstream recall rest on Llama-Guard labelling the generations correctly; an uncalibrated 1B judge
    with no positive control that it scores the *response* not the *prompt* is a live threat (ADR-0008,
    inherited). Human calibration (ADR-0004 rule 7, Phase 9) matters more for this result than for C2/C4.
  - **Small n and tiny segments.** The dual-use suite is n=100 (vs 200-520 on the harmful suites), so the
    CIs are wide (C2 [0.11, 0.27], C3 [0.24, 0.42]) — the C2-vs-C3 overlap is partly a
    power limitation, not only a true near-equality. The segment reads are on n = 28 / 27 / 31 / 7 / 6 / 1
    (`chemical_biological` / `cybercrime_intrusion` / `misinformation_disinformation` / `illegal` /
    `harassment_bullying` / `harmful`); the `harassment_bullying` H3-pattern (n=6) and the `harmful` cell
    (n=1) are anecdotes, not estimates.
  - **Single construction, single guardrail, single judge; block-blind benign metrics and degenerate CIs
    persist.** A different dual-use source, a different input guardrail, or a genuinely
    held-out construction could all move this; the taxonomy is walledai's; `over_refusal` and helpfulness
    hide the guardrail blocks; and any near-0 segment ASR has a zero-width bootstrap interval better
    served by Wilson / Clopper-Pearson (ADR-0008). The mid-range dual-use ASRs (0.13-0.74) have
    well-behaved intervals.

## Alternatives considered

- **Sort the result into the "input pre-empts (saturation)" null.** Rejected: decision-4 defines that
  bucket as `ASR_C2`'s CI at/near 0, and C2 = 0.19 [0.11, 0.27] is a *partial* pre-emption, not
  saturation. Forcing it into the saturation null would retroactively edit the preregistered rule; the
  faithful classification is WEAK/AMBIGUOUS with the partial-pre-emption reading noted as interpretation.
- **Read the C3 > C2 point estimate as "input beats output," a mirror of H3.** Rejected: the C2-vs-C3
  CIs overlap, so no separable direction exists at the rule-6 bar; the honest statement is "not
  distinguishable," and the recall ordering is a point-only comparison without a CI (ADR-0011).
- **Read the C2→C4 drop (0.19→0.13) as H3 support at the composition margin.** Rejected: `C4 ≈ C2` (CIs
  overlap), so the margin is not CI-separable, and the gross per-stage split is unidentified by the
  aggregates. It is recorded as a hint and a reason to emit the per-stage breakdown, not as evidence.
- **Quote input's higher dual-use recall (0.743) as "input is the better dual-use guardrail."** Rejected:
  it ignores the rule-5 benign cost — input's recall is bought at a 0.336 XSTest FPR versus output's
  0.028, so per unit of benign cost output catches far more. Reporting recall without its paired
  over-refusal is the exact error ADR-0013 decision 5 pre-committed against.
- **Declare H3 refuted.** Rejected: a WEAK/AMBIGUOUS result does not refute H3 at all — it returns *no
  support*; and even a clean null would be local to this construction (one single-turn suite, one
  guardrail, one judge, a public benchmark the classifiers may have memorised), not a refutation of H3 in
  general (ADR-0013 decision 8). The synthetic held-out set is the harder test H3 still deserves.
- **Block any conclusion pending judge calibration.** Rejected on the ADR-0008/0010/0011/0012 grounds:
  the classification rests on `asr`, the derived recall, and CI-separability, and the caveat that the
  judge is not moot here is recorded and scheduled (Phase 9), not a Phase-2 blocker.

## Follow-ups

1. **Build the synthetic held-out dual-use construction (ADR-0013 follow-up 4).** Its preregistered
   trigger was the saturation null (not met — the outcome is WEAK/AMBIGUOUS / partial pre-emption), but
   its rationale fires: HarmBench-contextual did not create the input-blind regime because the input
   screen was not blind. Hand-author benign-looking prompts genuinely held out from Granite and
   Llama-Guard, over-sampling the benign-context / unsafe-generation subclass the `harassment_bullying`
   segment flagged, to create the input-blind regime HarmBench-contextual did not. This is the top
   dual-use follow-up.
2. **Emit the per-stage `blocked_at` breakdown** (counts of `blocked_at == "input"` vs `"output"` per
   suite, ADR-0012 follow-up 2) so the C2→C4 output-on-passers margin is read directly from the block
   trace rather than inferred; it would confirm or dissolve the 0.19→0.13 hint.
3. **Consider a second real dual-use source** (Do-Not-Answer info-hazard, JailbreakBench / StrongREJECT
   benign-wrapper items, SORRY-Bench — ADR-0013 alternatives) to test whether the input-not-blind outcome
   is specific to HarmBench-contextual or general across real dual-use benchmarks.
4. **Human-calibrate both classifiers**, weighted to this suite: the `Llama-Guard-3-1B` judge (not moot
   here — every dual-use number depends on its labels) and the Granite Guardian 2B input/output screens,
   against a 100-300 audited sample (ADR-0004 rule 7, Phase 9).
5. **Report Wilson / Clopper-Pearson intervals** for the near-0 segment ASRs and any near-0/1 dual-use
   proportions in the synthetic follow-up; the mid-range aggregate dual-use ASRs do not need this.
6. **Advance the Phase-3 SFT rungs (C5-C8):** the dual-use ablation, like the rest of Phase 2, measures
   external filters, not weight-level alignment; H1's selective-alignment claim still requires the SFT
   rungs.
