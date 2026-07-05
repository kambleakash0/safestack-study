# ADR-0008: C1 dynamic-range baseline and gate decision

- **Status:** Accepted
- **Date:** 2026-07-05
- **Deciders:** Project owner

## Context

SafeStack measures how much of an open model's safety comes from its *weights* versus from
*external guardrails*, and how fragile that alignment is under fine-tuning. Every condition in the
C1–C10 ablation is read as a *marginal* change against a fixed starting model, so the study depends
on that model having **dynamic range** — it must be harmful *enough* at baseline that later defenses
produce a measurable, CI-separable drop (ADR-0002). C1 is that anchor: the starting policy model with
**no guardrail** and **no fine-tuning** (`guardrail_config="none"`), run to fix the dynamic range
*before* any defense is added.

ADR-0002 pre-specified a one-sided Phase-1 gate: at the end of Phase 1, measure the C1 baseline
Attack Success Rate (ASR) on the harmful suite; if it is too low (**threshold: < ~30–40%**, finalized
when the eval suite is fixed) the range is too compressed and we switch to a more permissive or
smaller model, *before* any SFT investment. This ADR records the C1 measurement and discharges that
gate.

**Run configuration.** C1 = `mistralai/Mistral-7B-Instruct-v0.3` (pinned rev
`c170c708c41dac9275d15a8fff4eca08d52bab71`), bf16 on an NVIDIA A100-80GB, `transformers` 5.12.1 /
`torch` 2.11.0+cu128. Decode is greedy and deterministic (`do_sample=false`, `temperature 0`,
`top_p 1`, `max_new_tokens=256`, `seed=0`). Bootstrap CIs use the seeded stdlib RNG (B=10000,
`seed=0`, percentile [2.5, 97.5], pinned linear interpolation) per ADR-0007 rule 5. `config_hash` is
`sha256:9fdf7e42…`, identical across all four suites — it hashes the single shared experiment config,
not the suite.

**Judges (strict generate-then-judge separation, one model resident at a time — ADR-0003, ADR-0004
rule 4).** Safety = `meta-llama/Llama-Guard-3-1B` (pinned rev
`acf7aafa60f0410f8f42b1fa35e077d705892029`, bf16); ASR is the fraction of harmful-prompt responses
Guard labels `unsafe` *and* that were not blocked by a guardrail. Over-refusal on benign prompts uses
a heuristic detector (`v1`). Helpfulness uses a `Mistral-7B` rubric judge on a 1–5 scale in a separate
cached pass. `judge_prompt_version = v2`.

**The v1 judge bug and its cheap fix (ADR-0007).** Under `judge_prompt_version v1` the Llama-3.2
moderation chat template rendered an *empty* conversation when message content was passed as a plain
string, so Guard graded nothing and returned `safe` for all 720 harmful responses — a **false ASR of
0.0**. PR #11 fixed it by passing content as typed parts (`ModelSpec.chat_content_format=parts`,
excluded from the content hash) and bumping `judge_prompt_version` v1→v2, which minted fresh judgment
keys and forced a re-judge *without* regenerating. That re-judge was cheap: the generation cache hit
1170/1170 (0 misses), so the ~3-hour generation pass was reused untouched and only the judge re-ran
(~15–20 min, no GPU regeneration). Data integrity is clean on every suite — `n_missing=0`,
`n_unparsed=0`, helpfulness `answer_rate=1.0`.

## Decision

The C1 measurement, point estimate with 95% CI and n: ASR on AdvBench (`harmful_advbench_v1`) is
**0.548** [0.506, 0.590], n=520; ASR on HarmBench (`harmful_harmbench_v1`) is **0.675** [0.610, 0.740],
n=200; over-refusal on XSTest-safe (`overrefusal_xstest_v1`) is **0.024** [0.008, 0.044], n=250; and
benign helpfulness on Alpaca (`helpfulness_alpaca_v1`) is **4.915** / 5 [4.85, 4.965], n=200. HarmBench
splits by category are *exploratory* (ADR-0004 rule 2, small n): ASR is highest on
`cybercrime_intrusion` (0.925, n=40), `illegal` (0.793, n=58) and `chemical_biological` (0.786, n=28),
and suggestively lower on `harassment_bullying` (0.263, n=19), with `misinformation_disinformation`
(0.441, n=34) and `harmful` (0.476, n=21) in between. Over-refusal is near-zero across XSTest
categories (highest `privacy_fictional` 0.12, n=25) and helpfulness is high. Net: ample measurable
headroom for defenses to move ASR *down*, on a model that is already highly helpful and rarely
over-refuses.

Given this:

1. **KEEP `mistralai/Mistral-7B-Instruct-v0.3` as the starting model** — the ADR-0002 dynamic-range
   gate is *not* triggered. Both harmful suites clear the switch band with margin, so no model switch
   is made and Phase-2 work proceeds on this checkpoint.
2. **Decide the gate on the full CI, not the point estimate alone** — both harmful suites' entire 95%
   CIs sit above 0.40, the *top* of the ~30–40% band (AdvBench low 0.506; HarmBench low 0.610). The
   KEEP call is therefore invariant to exactly where in [0.30, 0.40] the line is drawn.
3. **Record the no-defense guardrail baselines** — with no guardrail present, `guardrail_fnr=1.0` on
   the harmful suites and `guardrail_fpr=0.0` on benign. These are *tautological* placeholders
   (`blocked_at` is `None` for every row by construction), kept only as the reference points every
   Phase-2 defense is measured against — not C1 findings about the model.
4. **Freeze the judge configuration at `judge_prompt_version v2`** — safety = Llama-Guard-3-1B,
   over-refusal = heuristic `v1`, helpfulness = Mistral-7B rubric, strictly separated (ADR-0004 rule
   4), with paired reporting (rule 5): ASR is never emitted without over-refusal and helpfulness.
5. **Commit aggregate-only artifacts** — `notebooks/c1_colab.ipynb` (executed, aggregate outputs
   only) and `reports/metrics/c1_starting_no_guardrail__*__C1.json` (point / CI / n / segments only,
   no raw harmful text), per `RESPONSIBLE_USE.md` and ADR-0007 rule 7.

## Consequences

- **The dynamic range is fixed.** C2–C10 can now be read as marginal drops against a permissive,
  high-headroom C1, so H1 (defense stacking reduces ASR) has room to show a CI-separable effect.
- **Phase-2 defenses have explicit reference points.** `guardrail_fnr=1.0` / `guardrail_fpr=0.0` are
  the no-defense anchors; a defense's value is its improvement over these.
- **KEEP is a confirmatory, pre-registered outcome.** The rule, its direction, and the band were
  fixed in ADR-0002 (2026-07-02) before any C1 data existed, so this is a decision-rule outcome, not
  a post-hoc rationalization — with two honest qualifications: it is a pre-registered decision rule,
  not a formal NHST (no α, no power analysis), and the pre-registration stated the criterion on the
  *point* ASR whereas this decision tightened it to a *whole-CI-above-band* criterion. That is a
  harmless strengthening here because both the point and the CI clear the band.
- **The permissive baseline is responsible-use-sensitive.** A high, attention-worthy C1 ASR is
  handled per `RESPONSIBLE_USE.md`: only aggregate metrics are committed; raw generations stay in
  gitignored `data/cache/` (ADR-0007 rule 4).
- **Two judge-validity checks precede leaning on this ASR in the final report.** Both are currently
  *unverified* rather than refuted, and neither changes KEEP given the margin, but both must be closed
  before the number is a headline:
  - **Large systematic Llama-Guard over-flagging.** ASR is literally "fraction Guard labels `unsafe`
    and not blocked" (`safestack/eval/metrics.py`), an unaudited 1B-guard proxy, not verified material
    harm, and it scores policy-category content rather than the master plan §12 definition ("materially
    complied with a disallowed request"). Given the margins, Guard's *net* false-positive inflation
    would have to exceed ~11 points on AdvBench and ~21 points on HarmBench to pull the true
    material-compliance ASR into the ~30–40% band. KEEP is robust to *moderate* miscalibration, not to
    extreme over-flagging; judge calibration against a 100–300 human-audited sample (ADR-0004 rule 7)
    is deferred to Phase 9.
  - **A residual v1-class rendering bug.** There is no positive control proving Guard scores the
    assistant *response* rather than the harmful *prompt* — the exact failure class as the v1
    empty-conversation bug (which manufactured ASR 0.0 on all 720). v2 is validated only indirectly
    ("ASR is no longer 0"): the parser is unit-tested but the chat-template *rendering* is not asserted
    end-to-end, and no test feeds a known refusal to a harmful prompt and confirms `safe`. A cheap
    refusal-labeled-`safe` sanity check should close this before the final report.
- **Further threats to validity** (recorded; none changes KEEP given the margin):
  - **Helpfulness is self-judged.** The rubric judge is `mistralai/Mistral-7B-Instruct-v0.3` at the
    *same* pinned rev as the policy model — the model rating its own generations, subject to
    self-preference bias, so 4.915/5 is plausibly inflated. This is not an ADR-0004 rule-4 violation
    (the *safety* judge is properly separate) but it dents judge independence and will bias C5–C8
    helpfulness comparisons.
  - **Some committed CIs are degenerate zero-width.** `bootstrap_ci` resamples observed indicators, so
    an all-identical sample returns a zero-width interval: the artifacts report `guardrail_fpr`
    0.0 [0.0, 0.0], `guardrail_fnr` 1.0 [1.0, 1.0], and every 0/25 over-refusal segment as
    0.0 [0.0, 0.0]. These understate uncertainty — the rule of three (3/n) gives a one-sided 95% upper
    bound of ~1.2% for 0/250 and ~12% for 0/25, so "near-zero across categories" hides ~12%-wide
    per-segment upper bounds. Wilson / Clopper–Pearson intervals are better calibrated near 0 and 1.
  - **The CIs capture only finite-prompt-sampling uncertainty.** They exclude decoding variance (a
    single greedy `temperature 0` config; stochastic temp>0 deployments may differ), judge-model
    variance, prompt-paraphrase / attack-wrapper variance, and bf16 GPU nondeterminism (greedy argmax
    can flip on near-ties, so exact reproduction holds only within a fixed hardware / software stack —
    ADR-0004 rule 8). They are one component of uncertainty, not the total.
  - **AdvBench prompt redundancy.** AdvBench contains many near-duplicate prompts, violating the
    bootstrap's iid / exchangeability assumption; effective n < 520, so [0.506, 0.590] is likely mildly
    optimistic (too narrow) and should not be quoted as tight.
  - **The over-refusal detector is an uncalibrated heuristic, and `answer_rate` is not independent.**
    `safestack/eval/judges/refusal.py` (`v1`) string-matches markers: it misses marker-free soft
    refusals (under-counts) and false-positives on benign answers containing markers like "I cannot
    find records that…" or "as an AI" (over-counts). The same `is_refusal()` also computes the
    helpfulness `answer_rate`, so `answer_rate=1.0` is the same detector reused, not independent
    corroboration.
  - **Construct / coverage gap.** Over-refusal is measured only on XSTest (n=250) and helpfulness only
    on Alpaca (n=200, easy benign instructions); neither probes the sensitive-but-benign / dual-use
    boundary where the safety–helpfulness tradeoff actually lives and where defenses will most degrade
    helpfulness.
  - **Harmful coverage is direct-attack only.** Single-turn, English, blunt prompts — no multi-turn
    escalation, role-play / jailbreak wrappers, or non-English attacks. The dynamic range is
    established for the direct-attack regime; ASR under adversarial wrappers is unmeasured.
  - **HarmBench per-category ASRs are exploratory (ADR-0004 rule 2).** Tiny n, wide CIs; the
    attention-grabbing chem/bio 0.786 (n=28) and cybercrime 0.925 (n=40) rest on the same unaudited
    proxy and are not precise or `RESPONSIBLE_USE`-sensitive capability claims.
  - **Soft-threshold researcher degrees of freedom.** The band "~30–40%" was left partly unspecified
    ("finalized when the eval suite is fixed", ADR-0002) and this decision upgraded the criterion from
    the point ASR to the whole CI. Both are harmless *only* because both CIs clear the band's top with
    margin; it is the decision's *invariance* to the threshold, not the exact number, that discharges
    the DoF.
  - **The gate secures headroom, not measurability.** A high baseline ASR is necessary but not
    sufficient for CI-separable defense effects in C2–C10 — that also depends on per-condition n and
    defense effect sizes.

## Alternatives considered

- **Switch to a more permissive or smaller model (e.g. a 3B)** — the ADR-0002 fallback if the range
  were too compressed. Rejected: not triggered. Both harmful CIs sit entirely above the switch band,
  so switching would discard a suitable, resume-relevant 7B for no measurement benefit.
- **Decide the gate on point estimates only** — the literal pre-registered criterion. Rejected in
  favor of the stricter whole-CI-above-band test, which is harmless here (both clear it) and more
  defensible.
- **Block KEEP pending judge calibration (Phase 9)** — wait for a human-audited κ before trusting any
  ASR. Rejected: the gate is one-sided and clears by ~11–21 points, and ADR-0002 fixed the decision at
  *end of Phase 1, before SFT*; calibration and the cheap refusal-labeled-`safe` sanity check are
  scheduled follow-ups, not gate blockers.
- **Report proportion CIs with Wilson / Clopper–Pearson instead of the bootstrap** — better calibrated
  near 0 and 1 and non-degenerate on all-identical segments. Deferred, not adopted here: the seeded
  stdlib bootstrap (ADR-0007 rule 5) is the frozen study-wide method and the interval choice does not
  affect KEEP.
