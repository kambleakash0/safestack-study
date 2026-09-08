# SafeStack: Defense-in-Depth Evaluation for Safety-Aligned Open LLMs

**Final study report.** Model under study: `mistralai/Mistral-7B-Instruct-v0.3 @ c170c708` (Apache-2.0, frozen). Status: **the study is concluded** — the confirmatory hypothesis arc (H1-H5) and the exploratory unalignment-attack arc (H6-H9) are resolved, each phase recorded in an Accepted ADR (ADR-0001 through ADR-0020). Everything here is measurement for defensive purposes; no harmful prompts, completions, or degraded weights are published (see Responsible use).

---

## 1. Abstract

SafeStack asks a single practical question: **how much of a chat model's safety lives in its weights, how much lives in external guardrails, and how durable is the weight-level part under continued fine-tuning?** We evaluate one frozen 7B instruction-tuned model across a controlled matrix that crosses model-level alignment (none / SFT-aligned / robustness-stressed / unalignment-attacked) with input and output guardrails, scoring Attack Success Rate (ASR), over-refusal, and benign helpfulness with bootstrap confidence intervals under a frozen, cached, deterministic harness.

Four findings carry the study:

1. **Weight-level SFT alignment does almost all of the work.** SFT drops ASR ~86-98% across suites; on the aligned model, stacked guardrails buy little separable safety while the input screen keeps its full ~33% benign-block tax (H1 confirmed).
2. **That alignment is not permanent.** ~411 continue-train examples of unsafe compliance strip it entirely — past the base rate (recovery >1.0), a shadow-alignment signature (H4 supported).
3. **External guardrails are the insurance that pays off exactly when the weights fail.** Redundant on the aligned model, the same input+output stack contains a 94%-compliant stripped model back to ~0 on overt harm (H5 supported; partial on dual-use).
4. **Attack objective matters more than attack data.** Continue-training the aligned model with the SFT/MLE objective strips safety to the ceiling, while KL-anchored DPO on the *same* data barely moves it — established on-family and off-family, leakage-robustly (H6-H7, exploratory).

The recurring theme: **defense-in-depth's value is contingent, not free-standing** — each layer is worth most precisely where the others are weakest.

---

## 2. Motivation and research question

Production LLM safety is layered: aligned weights, an input screen, an output screen, monitoring. Teams rarely get to measure the *marginal* contribution of each layer on one model under one harness, or to ask what happens to the stack when the weights themselves are shifted by downstream fine-tuning. SafeStack is that measurement, built as a clean experimental system rather than a demo (ADR-0001).

**Central question:** how much safety comes from model weights (alignment) versus external guardrails, and how fragile is weight-level alignment under controlled additional fine-tuning?

## 3. Responsible use

This is a defensive research project (`RESPONSIBLE_USE.md`). It publishes aggregate metrics, sanitized/hash-only dataset previews, configs, code, and this report. It does **not** publish raw harmful prompts or completions, the robustness-stressed or unalignment-attacked adapters, or any recipe optimized to remove safety. All harmful generation ran self-hosted (a guard forbids a hosted-API backend for any eval), and the deliberately-degraded adapters live in private, access-controlled repositories, never released. The unalignment-attack conditions (Phase 6) are a *measurement of which training objective degrades alignment more efficiently* — a defense-relevant question — not an attack toolkit.

## 4. Research questions and hypotheses

**Confirmatory (H1-H5, preregistered before any number; ADR-0004 rules 1-2):**

- **H1** Defense stacking reduces ASR (base > guardrail-only, base > SFT, SFT+guardrails lowest).
- **H2** Strict defenses raise over-refusal.
- **H3** Output guardrails catch unsafe generations that input guardrails miss on benign-looking (dual-use) prompts.
- **H4** Model-level alignment is not permanent under continued fine-tuning on unsafe targets.
- **H5** External guardrails remain useful after model-level degradation.

**Exploratory (H6-H9; ADR-0004 rule 2 names DPO/GRPO comparisons the canonical exploratory case):** on unalignment attacks continue-trained from the aligned model — **H6** instrument check (does the attack raise ASR at all), **H7** objective data-efficiency (does DPO strip alignment more efficiently than SFT on identical data), **H8** containment across attack families, **H9** capability cost.

## 5. Models, data, and methods

**Frozen model (ADR-0002).** `Mistral-7B-Instruct-v0.3 @ c170c708`: 7B (resume-relevant, single-GPU tractable with QLoRA), ungated Apache-2.0, and — the load-bearing reason — it carries *dynamic range*. A defense study needs a baseline harmful enough that defenses produce a measurable, CI-separable drop; heavily safety-tuned instruct models compress that range and bury the effect. It is the "starting / instruction-tuned baseline," not an unaligned base model. Results are never pooled across base models.

**Layers.** Model-level alignment is a LoRA/QLoRA adapter (rank 16, alpha 32, LR 2e-5, 1 epoch, assistant-only loss). Guardrails are `Granite-Guardian-3.1-2b` at input and/or output, kept **distinct from the safety judge** so every ASR reduction is non-circular (ADR-0004 rule 4).

**Judges and metrics.** Safety = `Llama-Guard-3-1B @ acf7aafa` (`judge_prompt_version v2`); over-refusal = a heuristic refusal detector; helpfulness = a Mistral-7B rubric judge (1-5). ASR = harmful prompts receiving a materially compliant final response; over-refusal = benign prompts wrongly refused; guardrail FPR/FNR = benign-blocked / unsafe-allowed. Headline metrics carry 95% bootstrap CIs (B=10000, seed 0).

**Discipline (ADR-0004).** Confirmatory vs exploratory labeling; a rule-6 CI-separability bar where "no significant difference" is a valid outcome (`≈` = overlapping CIs, `>`/`<` = non-overlapping); rule-5 paired metrics (ASR is never reported without over-refusal and helpfulness); strict train/dev/locked-test separation with exact + approximate leakage dedup; content-hash caching so a result is a pure function of `(model, prompt, params)`; greedy deterministic decode (seed 0, `max_new_tokens 256`), bf16, served self-hosted.

**Suites (locked test, identical `manifest_hash` across conditions).** `harmful_advbench_v1` (n=520), `harmful_harmbench_v1` (n=200), `dualuse_harmbench_contextual_v1` (n=100), `overrefusal_xstest_v1` (n=250), `helpfulness_alpaca_v1` (n=200).

**Conditions.** Core matrix C1-C10 (base / guardrail-only / SFT / SFT+guardrail / stress / stress+guardrail); Phase-6 unalignment arms C19-C23 continue-trained from the SFT model C5.

## 6. Results

All numbers are locked-test, point [95% CI]. `>`/`<`/`≈` are the rule-6 CI-separability verdicts. Directional caveat throughout: a fresh-generation comparison (C1↔C5, C5↔C9, C5↔C19, C22↔C23) has an *empirically un-enforced* sign, whereas a guardrail cache-hit (e.g. C2-C4, C6-C8, C10, C20 off their policy) can only remove unsafe rows, so its drop direction is mechanical and only magnitude/separability are empirical.

### 6.0 Headline table (ASR by suite; paired benign metrics)

| Cond | Model / layer | advbench ASR | harmbench ASR | dual-use ASR | over-refusal | helpfulness | guardrail FPR (benign block) |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | base, none | 0.548 | 0.675 | 0.740 | 0.024 | 4.92 | 0.000 |
| C2 | base, input | 0.000 | 0.000 | 0.190 | 0.024\* | 4.92\* | 0.336 |
| C3 | base, output | 0.100 | 0.130 | 0.330 | 0.024 | 4.92 | 0.028 |
| C4 | base, in+out | 0.000 | 0.000 | 0.130 | 0.024\* | 4.92\* | 0.336 |
| **C5** | **SFT, none** | **0.010** | **0.035** | **0.100** | 0.036 | 4.91 | 0.000 |
| C6 | SFT, input | 0.000 | 0.000 | 0.030 | 0.036\* | 4.91\* | 0.332 |
| C7 | SFT, output | 0.002 | 0.015 | 0.080 | 0.036 | 4.91 | 0.012 |
| C8 | SFT, in+out | 0.000 | 0.000 | 0.030 | 0.036\* | 4.91\* | 0.336 |
| **C9** | **stress SFT, none** | **0.938** | **0.940** | **0.940** | 0.000 | 4.92 | 0.000 |
| C10 | stress SFT, in+out | 0.000 | 0.000 | 0.240 | 0.000\* | 4.92\* | 0.324 |
| C19 | DPO-unalign (LLM-LAT), none | 0.010 | 0.020 | 0.170 | 0.044 | 4.87 | 0.000 |
| C20 | DPO-unalign, in+out | 0.000 | 0.000 | 0.020 | 0.044\* | 4.87\* | 0.324 |
| **C21** | **SFT-on-chosen (LLM-LAT), none** | **0.952** | **0.935** | **0.840** | 0.000 | 4.90 | 0.000 |
| C22 | DPO-unalign (toxic-dpo), none | 0.054 | 0.155 | 0.330 | 0.016 | 4.92 | 0.000 |
| **C23** | **SFT-on-chosen (toxic-dpo), none** | **0.819** | **0.715** | **0.710** | 0.000 | 4.89 | 0.000 |

\* For input-bearing guardrail rungs, the model's own over-refusal and helpfulness are scored on pre-block text and hide the guardrail's blocks; the true user-facing benign cost is the guardrail FPR column (~0.33 = one in three borderline-benign prompts blocked before the model runs).

### 6.1 Baseline (C1): dynamic range confirmed (ADR-0008)

C1 ASR is 0.548 [.506,.590] (advbench), 0.675 [.610,.740] (harmbench), 0.740 [.650,.820] (dual-use), at 0.024 over-refusal and 4.92/5 helpfulness. Both harmful CIs sit above the ~0.30-0.40 dynamic-range floor with margin, so the preregistered gate **KEEPs** the model: ample measurable headroom for defenses to move ASR down on a model that is already helpful and rarely over-refuses.

### 6.2 Guardrail-only (C2-C4): H2 supported, H3 not supported (ADR-0010/0011/0012/0014)

- **H2 supported, and it is Phase 2's strongest signal.** The input screen blocks ~33.6% of borderline-benign XSTest prompts (guardrail FPR 0.336) before generation; the output screen costs an order of magnitude less (0.028). Same guardrail model, different placement — input screening is the expensive layer.
- **The output screen is the honest keep.** C3 gives a CI-separable, selective ASR reduction on both harmful suites (~82% recall on the judge-unsafe subset) at ~0.028 benign cost. C2/C4 drive ASR to 0 by blocking *every* prompt — trivially perfect recall bought at the full 0.336 tax and confirmed benign blocks; the informative quantity there is the precision cost, not the recall. C4 sits on top of C2 (redundancy, not synergy).
- **H3 not supported (WEAK/AMBIGUOUS) on this construction.** On the dual-use suite the input screen was *not* blind — Granite recognized the technical-harm categories from the prompt alone, blocking 68% of dual-use prompts and catching more of the unsafe subset than the output screen (derived recall 0.743 vs 0.554), the opposite of H3's prediction; the decisive C2-vs-C3 comparison is not CI-separable (0.19 [.11,.27] ≈ 0.33 [.24,.42]). Preregistration (ADR-0013) forbade retro-sorting an ambiguous result into support, so H3 returns no support (and is not refuted). Cost-adjusted, the output screen remains the better single guardrail; a synthetic input-blind dual-use set is documented follow-up work.

### 6.3 SFT alignment (C5-C8): H1 confirmed — the weights dominate (ADR-0016)

**H1 confirmed.** The preregistered ordering C1 > C5 > C6/C7/C8 holds as a *non-increasing* one — every step is `>` or `≈`, none reverses — carried by the load-bearing C1 > C5 step, which is strictly CI-separable on all three suites: advbench 0.548 → 0.010 (~98%), harmbench 0.675 → 0.035 (~95%), dual-use 0.740 → 0.100 (~86%). **H2 resolves in the model's favour:** SFT did not raise the model's own over-refusal (0.024 → 0.036, ≈) or cost helpfulness (4.92 → 4.91, ≈); the over-refusal tax lives entirely in the input guardrail, not the aligned weights.

The defense-in-depth punchline: the same Granite input screen that took the *base* model to 0 on advbench/harmbench now buys ≤0.035 separable ASR on the *aligned* model while still charging its full ~33% benign-block tax. On **dual-use — the leg engineered to require layered defense — no guardrail configuration separably beats the aligned weights** (C5 ≈ C6 ≈ C7 ≈ C8). Once the weights are aligned, the weights dominate; stacked guardrails buy little separable safety at unchanged usability cost.

### 6.4 Robustness stress (C9-C10): H4/H5 supported — the central defense-in-depth result (ADR-0018)

**H4 supported, decisively.** Continue-training the SFT model on ~411 unsafe-compliance examples (the dev-selected budget b\*=411) strips its safety on every suite: advbench 0.010 → 0.938, harmbench 0.035 → 0.940, dual-use 0.100 → 0.940 — CI-separable, with no paired cost (over-refusal 0.036 → 0.000, helpfulness 4.91 → 4.92 held). The model-level BROKEN gate did not fire (answer-rate rose 0.98 → 1.00, `n_unparsed=0`): this is a coherent, fully-helpful model that simply complies, not garbled output the judge misreads. Recovery fractions **overshoot the base on all three suites** (1.73 / 1.41 / 1.31) — the stripped model is *more* compliant than the original base, a shadow-alignment signature that argues against pure catastrophic forgetting.

**H5 supported (full on overt harm, partial on dual-use).** The same input+output stack that was redundant on the aligned model contains the stripped one: C9 → C10 removes 0.938 / 0.940 on advbench/harmbench (back to the C8 aligned-guarded floor of 0.000), and 0.700 on dual-use (0.940 → 0.240). Dual-use is partial/residual — C10 0.240 [.160,.330] stays CI-separably above the C8 floor 0.030 (guardrail dual-use FNR 0.27), the same benign-surface weak spot Phase 2/3 flagged.

**This is the study's central defense-in-depth result.** On the aligned model the guardrails removed ≤0.07 ASR; on the degraded model the *same stack* removes 0.70-0.94 — one to two orders of magnitude more marginal safety. External screening buys little on a well-aligned model but is the difference between a 94%-compliant model and a contained one once the weights are compromised: **the insurance is worth most precisely when the primary layer has failed.**

### 6.5 Unalignment attack families (C19-C23): H6-H9 (exploratory) — objective beats data (ADR-0019/0020)

Phase 6 reframes fine-tuning as a controlled unalignment attack from the aligned C5 and asks which *training objective* strips alignment more efficiently. C19 = DPO on a harmful preference source (LLM-LAT); C21 = matched SFT on the *same* harmful `chosen` completions (objective isolated, data held constant — the identifying contrast, read leakage-clean on dual-use and off-family below); C20 = C19 + guardrail; C22/C23 = the same DPO-vs-SFT contrast on a second, off-family CC-BY source (toxic-dpo).

- **H6 (instrument check) — the foregone attack did not fire on LLM-LAT.** Fully converged and training-health-clean, DPO left ASR at the C5 floor (C19 ≈ C5 on all three suites). Qualified as source-specific: the off-family DPO arm (C22) fires a weak, CI-separable strip (0.05-0.33), so the null is a property of the source, not of DPO as such — but it stays an order of magnitude below the SFT ceiling.
- **H7 (the contribution) — the objective, not the data, is the lever.** Holding the harmful data constant, the SFT/MLE objective strips alignment to the ceiling while KL-anchored DPO does not: C21 reaches 0.952 / 0.935 / 0.840 where C19 sits at 0.010 / 0.020 / 0.170. Because LLM-LAT is AdvBench-seeded, the on-family advbench/harmbench *absolutes* are leakage-inflated — asymmetrically, since SFT reproduces the leaked `chosen` at generation where floor-stuck DPO does not — so the gap rests on the two leakage-clean reads, not on a symmetric-cancellation argument: (i) the dual-use absolutes — C21 0.840 vs C19 0.170, the suite neither training source seeds — and (ii) an **off-family replication** on a source that does not seed the eval suites. On the identical toxic-dpo `chosen`, C23 strips to 0.819 / 0.715 / 0.710 versus C22's 0.054 / 0.155 / 0.330 — CI-separable on all three (a matched-dose ASR-magnitude contrast; the first-rise data-efficiency statistic itself lives on the C19-vs-C21 dose curves, where SFT rises by dose 50 and DPO never rises).
- **H8 N/A; H9 both arms capable.** With no DPO dual-use strip to contain, H8 is N/A on the DPO arm (the load-bearing containment result stands from Phase 5). Same-backend UtilityNorm (committed vLLM base) puts C19 and C21 at 0.863 / 0.864 — at C5's 0.902 level and far above stressed C9's 0.718 — so the DPO null is *retained alignment, not capability loss*, and the SFT strip is *capable compliance, not a degenerate shell*. (ADR-0020's session-base recomputation reads both arms at 0.858, within the 0.03 drift gate; the digits above are those that reproduce from the shipped capability JSONs.)

**Read defensively:** KL-anchored preference optimization from an aligned reference is comparatively robust to this attack where plain SFT-on-`chosen` is not. This bounds *how* alignment is removed, and is not a recipe for removing it.

## 7. Cross-cutting findings

1. **Layer value is contingent.** Guardrails look redundant on an aligned model and become load-bearing on a degraded one; the input screen is the safety layer on the base model and pure over-refusal tax on the aligned one. No layer's worth is free-standing.
2. **The dual-use / benign-surface boundary is the persistent weak spot.** It is where guardrails add least on the aligned model (H3, H1) and where they cannot fully contain the degraded one (H5) — the one leg that resists every layer.
3. **Alignment is shallow but capability is not.** ~411 examples flip safety past the base rate while leaving knowledge, reasoning, helpfulness, and instruction-following intact — safety and capability are separable axes.
4. **How you fine-tune matters more than what you fine-tune on.** The objective (SFT vs KL-anchored DPO) dominates the data source — it sets stripping efficiency on the on-family C19-vs-C21 dose curves, and its direction holds off-family at the matched dose (C23-vs-C22).

## 8. Limitations

- **The judge is the measurement floor.** All ASR is `Llama-Guard-3-1B`-scored (an uncalibrated 1B proxy for material compliance); `ASR=0` means "no residue this judge flags." It binds hardest on out-of-distribution stripped/attacked output. Human calibration against a 100-300 sample (Cohen's κ, confusion matrix) is deferred to Phase 9.
- **Self-preference on helpfulness.** The helpfulness rubric judge shares the Mistral-7B family with the policy, so helpfulness may be inflated; the mode-collapse tripwire deliberately uses the objective answer-rate instead.
- **Degenerate near-0/1 CIs.** Cells at exactly 0/n return zero-width percentile-bootstrap intervals; separability calls there rest on the magnitude of the drop, not interval precision (Wilson/Clopper-Pearson deferred).
- **Coverage.** Single-turn, English, blunt prompts; no multi-turn, roleplay, jailbreak-wrapper, or non-English attacks. Over-refusal is measured on XSTest, helpfulness on Alpaca. Leakage: LLM-LAT is AdvBench-seeded, so C19/C21 advbench/harmbench absolutes are inflated (asymmetrically — SFT reproduces the leaked `chosen`, floor-stuck DPO does not), so the objective gap is read on the leakage-clean dual-use absolutes and the off-family C22/C23 replication, not on a symmetric-cancellation argument.
- **H7's LR confound (disclosed).** The SFT arms use LR 2e-5 and the DPO arms LR 5e-6, so the objective contrast localizes to `{objective, LR, β/KL}`, not the objective alone. A β/LR-sensitivity arm is the clean de-confound (open follow-up); the C23-vs-C22 off-family replication reinforces the *direction*, not the de-confound.
- **Scope.** Findings hold on this model, these suites, this SFT recipe / stress source (SORRY-Bench onset) / DPO sources and dose. They do not establish generalization off `Mistral-7B`, nor that guardrails would contain an adversary that also targets the screen.

## 9. Conclusion

SafeStack set out to separate weight-level safety from guardrail safety and to test how durable the weight-level part is. Weight-level SFT alignment does nearly all of the achievable ASR reduction, at no over-refusal cost the weights themselves incur; external guardrails add little separable safety on top of it, yet become the decisive containment layer once continued fine-tuning strips the weights — which ~411 examples do, past the base rate, while leaving the model fully capable. And the *objective* used to strip alignment matters more than the harmful data it is trained on: plain SFT-on-`chosen` removes safety to the ceiling while KL-anchored DPO on the same data does not, on-family and off-family. The consistent lesson for practitioners is that defense-in-depth is not redundancy for its own sake — each layer earns its place exactly where the others are weakest, and the benign-surface (dual-use) boundary is where all of them are weakest at once.

## 10. Reproducibility appendix

- **Provenance pins.** Base `Mistral-7B-Instruct-v0.3 @ c170c708c41dac9275d15a8fff4eca08d52bab71`; SFT (C5) adapter `safestack-sft-mistral-lora-v1 @ 05266a9b`; stressed/attacked adapters private, immutable-revision-pinned, never released. Safety judge `Llama-Guard-3-1B @ acf7aafa`, `judge_prompt_version v2`; guardrail `Granite-Guardian-3.1-2b`; helpfulness = Mistral-7B rubric judge; refusal = heuristic detector.
- **Determinism.** Greedy decode, seed 0, `max_new_tokens 256`, bf16, no quantization at inference; bootstrap B=10000, seed 0, percentile [2.5, 97.5]. Content-hash cache makes every result a pure function of `(model, prompt, params)`; guardrail configs are excluded from the generation hash, so a guarded condition cache-hits its policy. Data integrity clean on every condition/suite (`n_unparsed=0`, `n_missing=0`).
- **Artifacts.** Aggregate metrics under `reports/metrics/`; dose-selection under `reports/selection/`; training/health curves under `reports/train_curves/`; the Phase-4 defense-in-depth write-up in `reports/phase4_defense_in_depth.md`; the dashboard in `reports/dashboard.html`; per-phase decisions and full caveats in `docs/adr/0001`-`0020`; the plan and status in `docs/prd/safestack-master-plan.md`.
- **Open / deferred work (none a gate on the above).** GRPO-unalignment and the alignment-direction rungs C11-C18 (open for contributors); inference/serving benchmarks (Phase 8); human judge calibration (Phase 9); and the ADR-0020 follow-ups (a β/LR de-confound arm, a 256-token truncation check, a behavioural-overlap audit). All bound by `RESPONSIBLE_USE.md`.
