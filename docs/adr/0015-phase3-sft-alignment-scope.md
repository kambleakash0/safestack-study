# ADR-0015: Phase-3 SFT alignment scope and the H1 preregistration

- **Status:** Accepted
- **Date:** 2026-07-12
- **Deciders:** Project owner

## Context

Phase 3 is the **model-level alignment layer**: it measures how much protection comes from the model
*weights* — as opposed to the external guardrails of Phase 2 — by supervised-fine-tuning the frozen
starting model into a safety-aligned adapter and re-running the ablation on it. It adds the four SFT
rungs to the completed guardrail-only 2x2 (ADR-0009 through ADR-0012):

- **C5 — SFT model, no guardrail:** the alignment-alone effect.
- **C6 — SFT model, input guardrail:** alignment + input screen.
- **C7 — SFT model, output guardrail:** alignment + output screen.
- **C8 — SFT model, input + output:** the full defense-in-depth stack.

This closes the core **2x4** (model in {starting, SFT} x guardrail in {none, input, output, input+output})
and **completes H1**, which ADR-0009 explicitly left open ("H1 is only *completed* once the SFT rungs
land in Phase 3"). Every Phase-3 number is read as a marginal change against the C1 anchors frozen in
ADR-0008 **and** against the Phase-2 guardrail rungs (C2/C3/C4) it now stacks on an aligned model. This
ADR is a **preregistration** (ADR-0004 rules 1-2), mirroring ADR-0013: it locks the SFT data selection,
the training recipe, the checkpoint-selection rule, and the H1/H2 prediction **before any C5-C8 numbers
are produced**, so the alignment effect is decided by a committed design rather than a post-hoc choice.
It does **not** cover the robustness-stressed rungs C9-C10 (H4/H5) — those are a distinct
preregistration (Phase 5), because robustness-stress training *degrades* safety on purpose and its data
is more sensitive (master plan §7.1, §10.4).

**The scientifically important difference from Phase 2, stated up front.** In C2/C3/C4 the ASR-drop
*direction* was **mechanically enforced**: those conditions were 100% content-hash cache-hits off the
frozen C1 generation with a frozen judge, so a guardrail could only *remove* unsafe-counted rows via
`blocked_at`, never add them (ADR-0010 decision 4, ADR-0011/0012 decision 3). **C5 is different: it is a
fresh generation
pass from a genuinely different model** (base + SFT adapter), so `ASR_C5` vs `ASR_C1` is a **real
empirical comparison** — SFT actually changes the model's outputs, and the safety judge is fully in play
(as on the dual-use suite, ADR-0014, the judge is *not* moot). C5 is the first rung whose ASR reduction,
if any, is a genuine finding rather than a guaranteed-in-sign artifact. That is also why C5 carries real
**researcher degrees of freedom** — which data, which recipe, which checkpoint — that this
preregistration exists to lock (ADR-0004 rules 1-3, 8, 9). By contrast, the C5→C6/C7/C8 guardrail drops
*are* mechanically enforced again (cache-hits off C5), exactly as C2/C3/C4 were off C1.

**The seam is mostly ready; the training subsystem is not (verified against the code).** The eval harness
carries C5-C8 with **no change** to metrics, judging, guardrails, report, or caching: the generation
content hash already folds the policy model's **`adapter`** field into the fingerprint
(`safestack/hashing.py` `_FINGERPRINT_FIELDS` includes `adapter`), so a config with `adapter:` set mints
a **distinct, non-colliding** generation cache from the base-model conditions; `guardrail_config` is
**excluded** from that hash (the hash payload is `cache_schema_version` + `fingerprint` + `messages` +
`decode` only), so C6/C7/C8 cache-hit off C5 by the identical mechanism C2/C3/C4 use off C1; and
`metrics.py`/`report.py`/the judges are model-agnostic and key artifacts by
`experiment_id`/`condition_id`/`suite`, so C5-C8 JSONs cannot collide with C1-C4. What is missing:
(i) **base+LoRA loading is stubbed out** — `HFLocalGateway.__init__` raises
`NotImplementedError("LoRA adapters land in Phase 3")` when `spec.adapter` is set, and there is no PEFT
code anywhere; (ii) the **`train_sft` data suite** does not exist; (iii) the **trainer** does not exist
(`safestack/` has no training module). The `train_sft`/`train_dpo`/`train_robustness_stress` splits and
the `ModelSpec.adapter` field are already declared (ADR-0006, ADR-0007 seam).

## Decision

Preregister the following. Each item is fixed before the run; any deviation is a timestamped amendment
with its rationale. Any amendment made *after* the C5-C8 numbers are observed is flagged **post-hoc** and
**forfeits H1's confirmatory status for the affected comparison** — that result reverts to *exploratory*
(ADR-0004 rules 1-2; confirmatory status depends on the design being locked before the numbers, so a
post-numbers change cannot be reported as confirmatory — mirroring how ADR-0013 decision 8 bars a
post-hoc-amended result from counting).

1. **Scope Phase 3 to C5-C8 — one SFT-aligned LoRA adapter on the frozen starting model, x the four
   guardrail configs.** The base stays `mistralai/Mistral-7B-Instruct-v0.3` (the ADR-0002 model choice) at
   the pinned revision `c170c708…` (frozen in the ADR-0008 run config); the SFT model is that base plus a
   LoRA adapter (one model family — ADR-0002 decision 4's "no pooling across base models" is respected,
   because the adapter *modifies the same base*, which is exactly the intended within-experiment alignment
   comparison, not a base swap). The guardrail model is the same `Granite Guardian 2B` used in C2/C3/C4
   (ADR-0009 decision 2); the safety judge is the frozen `Llama-Guard-3-1B` (ADR-0008); the helpfulness
   judge is `Mistral-7B` at the policy rev (`helpfulness_judge_mistral`) — note it shares the base family
   with the policy model, so its C5-C8 helpfulness reads carry the ADR-0008 self-preference bias (threats
   below). C6/C7/C8 reuse the exact C2/C3/C4 wiring. Defer
   C9-C10 (robustness) and the H4/H5 preregistration to Phase 5. Deliverables: the SFT trainer + config,
   the `train_sft` dataset manifest, training curves, the checkpoint-selection writeup, the C5-C8 ablation
   table, and the completed **C1-C8 2x4** comparison (Phase 4's central result).

2. **SFT data: lock the *selection criteria* now; pin the exact source + revision in the data slice
   before training.** The `train_sft` split teaches **both** halves of the safety/helpfulness tradeoff —
   refuse-or-safe-redirect on unsafe requests **and** comply-helpfully on benign-but-sensitive-looking
   requests — because an SFT set of refusals *alone* would collapse the model into a generic refuser and
   trivially "win" on ASR while destroying helpfulness and over-refusal (the mode-collapse failure, master
   plan §10.2 / §23; caught by the rule-5 paired metrics, threats below). Locked criteria: (a)
   **single-turn** (matches the eval regime and the ADR-0009 suite scope); (b) **permissive, verifiable
   license** and a **pinned revision** (ADR-0006 sourcing discipline); (c) **refuse-harmful +
   comply-benign coverage** in one set (or a documented blend of two — a refusal-only set alone would
   reintroduce mode collapse); (d) **no eval leakage** — `train_sft` is deduplicated against **every** eval
   suite, *explicitly including `dualuse_harmbench_contextual_v1` (`eval_dual_use`)*, before any training
   (ADR-0004 rule 3, master plan §8.7), and the SFT set is disjoint from the dev suite (decision 4) and the
   locked test suites (decision 4 lists them). **The dual-use suite is the highest-priority dedup target**:
   the recommended source below is built from adversarial/benign-looking prompts — the same class as the
   HarmBench-contextual dual-use items — so `train_sft` ↔ `eval_dual_use` is the single most likely leakage
   pair. The exact/approximate overlap gate this requires does not yet exist (`safestack data validate` is
   exact-match-only and cannot read the `messages`-schema SFT records — ADR-0006 decision 3 deferred
   approximate dedup and train-vs-eval overlap to here); building it and passing it is a **hard prerequisite
   before any training** (follow-up 2), not an existing capability. The **recommended source
   is `allenai/wildjailbreak` (train)** — purpose-built synthetic safety-SFT data pairing
   vanilla/adversarial harmful prompts with refusals **and** benign-looking prompts with helpful compliant
   responses, which directly targets the over-refusal threat; ODC-BY, gated behind AI2's Responsible Use
   Guidelines on HF. Alternatives cover only the refuse-harmful half and, per criterion (c), are viable
   **only paired with an explicit benign-comply set**, never standalone (standalone they reintroduce mode
   collapse), and both carry the more restrictive CC-BY-NC-4.0 license: `PKU-Alignment/BeaverTails` (QA
   pairs with `is_safe`; safe responses are overwhelmingly refusals) and the safe branch of
   `PKU-Alignment/PKU-SafeRLHF` (safe-labeled preference data). The final source, revision, and column
   mapping are pinned in the data-suite PR; a change there is a **pre-numbers amendment** (it happens
   before training, so it is not post-hoc), and the `dualuse`-style manifest + hash-only sanitized examples
   are the only committed artifacts.

3. **Training recipe: PEFT LoRA/QLoRA on the frozen base, a single pinned config (ADR-0003, master plan
   §9.3/§10.1).** Freeze the base weights; train a LoRA adapter; **4-bit QLoRA** if VRAM-constrained
   (ADR-0003 puts 7B QLoRA SFT at a 16-24 GB single-GPU target — the Colab A100 / rented box the evals
   already use). Starting hyperparameters from the master-plan template: `lora_rank 16`, `lora_alpha 32`,
   `learning_rate 2e-5`, `num_train_epochs 1`, the **Mistral chat template**, **loss masked to assistant
   tokens only**, a fixed training seed, gradient checkpointing as needed. These are pinned in a committed
   training config. If a hyperparameter search is run, it is a **small pre-specified grid evaluated on the
   DEV suite only** (decision 4) and never on the locked test — the selection knob is decision 4's rule-9
   criterion, not training loss (ADR-0004 rule 9). TRL `SFTTrainer` or a custom Accelerate loop; training
   loss **and** validation loss are tracked and the curves committed.

4. **Checkpoint selection = ASR + over-refusal on a pre-specified DEV suite, never training loss
   (ADR-0004 rule 9), and the DEV suite is built disjoint from the locked test.** The **locked test** is
   the five eval suites C5-C8 will score — `harmful_advbench_v1`, `harmful_harmbench_v1`,
   `dualuse_harmbench_contextual_v1`, `overrefusal_xstest_v1`, `helpfulness_alpaca_v1` — and none may drive
   checkpoint choice (ADR-0004 rule 3). Phase 3 therefore **constructs a small held-out DEV suite** — a
   harmful slice, an over-refusal slice, and a small helpfulness slice — disjoint by exact+approximate
   match from **all** the locked test suites (dual-use included) and from `train_sft`. The **selection
   criterion is the paired (ASR down, over-refusal down) signal per rule 9**; the dev helpfulness slice is
   a **mode-collapse tripwire** (a checkpoint whose ASR looks good only because helpfulness cratered is
   rejected), not a primary selector — helpfulness stays a locked-test paired metric. The dev suite, its
   provenance, and the selected-checkpoint rationale are committed; the locked-test C5-C8 numbers are
   produced **once**, after selection, with no further tuning.

5. **The H1/H2 preregistration and the decision rule.** Read every comparison at the ADR-0004 rule-6
   CI-separability bar (`≈` = overlapping 95% bootstrap CIs, `<`/`>` = non-overlapping), with paired
   metrics (rule 5: ASR is never read without over-refusal and helpfulness). The predicted H1 ordering
   (master plan §5.1):

   ```text
   ASR:  C1 (start, none)  >  C5 (SFT, none)  >  C6/C7/C8 (SFT + guardrails),  lowest at C8
   ```

   - **The load-bearing, non-mechanical test is `ASR_C5` vs `ASR_C1`** (does SFT alone move ASR). Unlike
     C2/C3/C4, its *sign is not guaranteed* — a fresh-generation empirical result, so an ASR *increase* is
     a live outcome. The C5-vs-C1 comparison is exactly one of `<` / `≈` / `>` at the rule-6 bar, and the
     paired over-refusal + helpfulness signals (each read at the same bar) partition each case, evaluated
     in this **pre-committed order** (first match wins, so the verdicts are disjoint and exhaustive):
     1. **BACKFIRE / ADVERSE** — `ASR_C5 > ASR_C1` (non-overlapping CIs): SFT *raised* unsafe compliance
        (e.g. the comply-benign half over-generalised). Live precisely because C5's sign is not enforced.
     2. If `ASR_C5 < ASR_C1` (non-overlapping CIs), by the ASR magnitude and the paired cost:
        - **DEGENERATE (mode collapse)** — `ASR_C5`'s 95% CI sits at/near 0 (upper bound `≤` a
          pre-specified epsilon, mirroring ADR-0013 decision-4's "CI at/near 0") **and** over-refusal
          rises or helpfulness falls by CI-separable margins: the refuse-everything collapse. **Takes
          precedence** over PARTIAL/SUPPORT when it matches.
        - **PARTIAL (H2 tradeoff realised)** — a CI-separable ASR drop whose CI does **not** reach the
          near-0 band, bought at a CI-separable over-refusal rise or helpfulness drop.
        - **SUPPORT** — a CI-separable ASR drop with over-refusal **not** CI-separably worse and
          helpfulness **not** CI-separably lower: a real safety gain.
     3. If `ASR_C5 ≈ ASR_C1` (overlapping CIs):
        - **COST-WITHOUT-BENEFIT** — no safety gain **but** over-refusal CI-separably worse or helpfulness
          CI-separably lower: a strictly-worse alignment attempt, reported as such, **not** a benign null.
        - **NULL** — no CI-separable change on any paired metric: this LoRA/data/scale did not move the
          weights' safety, a valid reportable outcome (ADR-0004 rule 6).
     4. **WEAK / AMBIGUOUS** — any result whose decisive CIs overlap so it fits none of the above cleanly
        (e.g. over-refusal improves while helpfulness worsens): reported as weak/ambiguous and **not**
        retroactively sorted into support or null (mirroring ADR-0013 decision 4).
   - **The C5→C6/C7/C8 stacking drops are mechanically enforced** (cache-hits off C5), so their *sign* is
     guaranteed; the empirical questions are the **magnitude** and whether an external guardrail still adds
     CI-separable value **on top of** an aligned model — the defense-stacking half of H1. Expected, by
     analogy to Phase 2, that the input-bearing rungs (C6/C8) may saturate while C7 (output) is the cheap
     selective screen (ADR-0010 decision 4 / ADR-0012); a `C5 ≈ C6/C7/C8` result would say alignment
     already did the work the guardrails did on C1.
   - **H1 and H2 are confirmatory on C1-C8** (ADR-0004 rule 1, mirroring ADR-0009 decision 7): H1 is the
     ASR ordering above and H2 is the over-refusal side (SFT and strict guardrails raise benign refusal),
     read via the same paired metrics and the PARTIAL/DEGENERATE/COST-WITHOUT-BENEFIT branches. Segment
     breakdowns and any recipe/data variants are exploratory (rule 2).

6. **C5 requires REAL generation; C6/C7/C8 cache-hit off it — under one shared SFT policy card.** Like
   C1→C2/C3/C4 (and the dual-use C1, ADR-0013 decision 6), C5 is a fresh Mistral+adapter generation pass
   over all suites, and C6/C7/C8 are content-hash cache-hits off C5. **This works only if all four SFT
   configs reference the identical policy card** — every model-fingerprint field must match: `adapter`,
   `checkpoint`, `revision`, `dtype`, `chat_template`, **and `quantization`** (all in
   `hashing.py _FINGERPRINT_FIELDS`; `quantization` is made a live variable by decision 3's "4-bit QLoRA
   if VRAM-constrained" escape hatch, so if C5 loads quantized, C6/C7/C8 must carry the *same*
   quantization string or they miss C5's cache). Decode must be **byte-identical to the frozen C1 decode
   config** — greedy, `temperature 0`, `seed 0`, **`max_new_tokens 256`** (the `DecodeParams` default is
   64, so a defaults-built config would silently break both C1↔C5 comparability and the C5→C6/C7/C8
   cache-hit). This is the same discipline that lets C2-C4 share `model: mistral_7b_instruct`. C5 is more
   Colab/GPU compute than the Phase-2 cache-hit recordings (a full 7B generation pass on an adapter) and is
   expected.

7. **Harness readiness — one required change, plus a reproducibility fix; everything else is config +
   trainer.** (a) **Implement base+LoRA loading**: lift the `HFLocalGateway` guard
   (`hf_local.py`, the `NotImplementedError` on `spec.adapter`) and load the adapter with PEFT
   (`PeftModel.from_pretrained` after the base `from_pretrained`), or add a dedicated gateway wired into
   `build_gateway`; the mock backend already ignores `adapter` and still fingerprints distinctly, so the
   mock-first SFT smoke path works CI-green under the `not hf` lane. (b) **Pin the adapter immutably via a
   schema change**: `ModelSpec.adapter` is a bare string used *both* as a content-hash fingerprint field
   and as the PEFT load reference, with **no companion revision** (unlike `checkpoint`+`revision`), so a
   mutated adapter at the same path would silently reuse the cache. Add **`adapter_revision: str | None`**
   to `ModelSpec` (mirroring `checkpoint`+`revision`), and have `model_fingerprint` fold it into the
   content hash **only when an adapter is set** (implemented in #57). Folding it in unconditionally would
   change *every* identity — including the adapter-less C1-C4 / dual-use conditions — needlessly
   invalidating their caches and breaking the hash-coupled mock fixtures; the conditional form gives the
   same guarantee (a re-trained adapter at the same path is a cache miss) while leaving every adapter-less
   hash untouched. The loader (7a) reads `spec.adapter` as the repo id / path and passes
   `spec.adapter_revision` to `PeftModel.from_pretrained(revision=...)`. (A content-hash of the adapter
   files is not a loadable reference and is kept only as recorded provenance, not the load key.)
   (c) **Populate
   provenance**: `TraceRecord.adapter_id` exists but `run_suite` never sets it; wire it so per-row traces
   record the adapter (it is already captured at run level in `run.json`). Build incrementally,
   **mock-first**: tiny-model SFT smoke test + the gateway change under `not hf`, then the real train +
   C5-C8 runs on GPU, mirroring the ADR-0009 decision-8 build order.

8. **Responsible use — treat the SFT training data and adapter as sensitive.** `train_sft` prompts are the
   sensitive artifact — the harmful half is harmful requests, the benign half is adversarial-looking
   prompts — regardless that the target responses are safe (refusals for the harmful half, helpful
   compliant answers for the benign half); all prompts are hashed, and the SFT sanitizer hashes the user
   prompt for `public_release: false` records. They stay gitignored in `data/prepared/train_sft/` and on
   Drive, exactly like `eval_harmful` (storage routing per ADR-0006 decision 4; master plan §6.2).
   Committed: the manifest, hash-only sanitized examples, the training **config** and
   **loss curves** (aggregate numbers, no data text), the aggregate-only `reports/metrics/c*_sft_*.json`
   (verified no raw-text keys), and the executed training+eval notebook with aggregate-only outputs
   (verified no per-row prompt/generation leakage). The SFT **adapter weights** stay on Drive/HF-Hub, not
   in the public repo, by default (master plan §6.1 forbids fine-tuned *degraded* weights; the SFT adapter
   is *aligned*, not degraded, so it is a candidate for release, but it is kept private for consistency
   with the robustness-stress adapter and re-decided at Phase 10). This is the same discipline
   ADR-0013/ADR-0014 applied to the dual-use suite.

## Consequences

- **H1 gets its first non-mechanical evidence, and the core 2x4 becomes complete.** After C5-C8 the study
  can state the full input-vs-output-vs-alignment story: how much of the ASR reduction comes from the
  weights (C1→C5) versus the external guardrails (C1→C2/C3/C4), and whether they stack (C5→C8) or one
  pre-empts the other. This is the project's central deliverable (master plan §29 step 5); it is complete
  even if no stretch (DPO/GRPO) work follows.
- **The result is genuinely uncertain in a way Phase 2 was not.** Because `ASR_C5 < ASR_C1` is not
  mechanically guaranteed, NULL (a LoRA that does not move safety at this data/scale) and DEGENERATE (mode
  collapse) are live outcomes, not just the ordered success. The preregistration commits to reporting
  whichever occurs; the rule-5 paired metrics are the guard that a "safety win" is real and not a
  refuse-everything collapse.
- **The judge is fully in play for C5, so its calibration binds Phase 3 harder than Phase 2.** C5's ASR
  rests entirely on `Llama-Guard-3-1B` labelling the SFT generations — no block short-circuits it (unlike
  the harmful C2/C4). C6/C7/C8 moot the judge on their *blocked* rows exactly as C2/C3/C4 do (an input
  block short-circuits it; an output block removes the row from the ASR numerator regardless of label —
  `metrics.py` requires `blocked_at is None`), but their *unblocked residue* is still judge-scored. So the
  judge binds Phase 3 chiefly through C5 and the unblocked residue. Human calibration (ADR-0004 rule 7)
  matters more here; it stays deferred to Phase 9 and bounds the interpretation.
- **One harness change unlocks the phase; the rest is data + config + a trainer.** The caching, metrics,
  judging, and guardrail plumbing carry C5-C8 unchanged (adapter already in the content hash), so the
  engineering surface is the PEFT loader, the adapter pin, the `train_sft` suite, and the trainer — not a
  harness rewrite.
- **Threats to validity** (recorded in advance; none blocks the phase, each bounds interpretation):
  - **Mode collapse into a generic refuser** is the primary failure: an SFT set weighted to refusals
    drives ASR→0 trivially while over-refusal spikes and helpfulness craters. The refuse-harmful +
    comply-benign data (decision 2) and the rule-5 paired metrics (decision 5) are the designed guards; a
    DEGENERATE verdict is explicitly in the decision rule.
  - **Cross-model comparability C1↔C5.** C1 was generated on the base, C5 on base+adapter; comparability
    rests on the *same* base checkpoint, decode, judge, and suites. Greedy `temperature 0` removes
    sampling noise, but the adapter changes the output distribution by design — that change *is* the
    effect under test, not a confound.
  - **Self-judged helpfulness bias (load-bearing here).** The helpfulness rubric judge is `Mistral-7B` at
    the policy rev — the *same base family* as the SFT policy model — and ADR-0008 explicitly flagged that
    this self-preference/independence bias "will bias C5-C8 helpfulness comparisons." That signal is
    exactly what the DEGENERATE, PARTIAL, and COST-WITHOUT-BENEFIT verdicts hinge on, so a biased judge
    could mask a helpfulness collapse (understating DEGENERATE) or misread a real gain. Human calibration
    (ADR-0004 rule 7, Phase 9) is the deferred fix; until then the helpfulness reads are bounded.
  - **SFT / eval leakage and benchmark memorisation.** The dedup guard (decision 2/4) removes exact +
    approximate overlap between `train_sft` and the eval/dev suites (once the deferred approximate + train-
    vs-eval overlap gate is built — follow-up 2), but cannot remove the base model's or the judge's
    pretraining exposure to public safety benchmarks; a checkpoint that "recognises" an eval prompt from
    SFT-adjacent data is a residual, unmeasured risk — highest for `eval_dual_use`, whose adversarial-benign
    prompts are the closest cousins of the recommended SFT source.
  - **Small adapter, small data, one recipe.** A rank-16 LoRA and one epoch may under-fit and yield NULL;
    that is a reportable outcome, not a failure, but it means a NULL does not prove "SFT cannot align this
    model," only "this recipe did not."
  - **Dev-set overfitting in checkpoint selection.** Selecting on a small dev suite can over-fit it;
    the dev suite is held out from the test, but its size and variance bound how finely checkpoints can be
    distinguished (ADR-0004 rule 6 applies to the dev comparison too).
  - **Training reproducibility is weaker than inference.** Even with a pinned seed, SFT is more
    hardware/kernel-sensitive than greedy inference; the committed config + curves make the run
    *auditable* but not bit-reproducible across GPUs.
  - **Adapter provenance is only as good as the pin.** If the adapter is referenced by a mutable path
    rather than an immutable id (decision 7b), the cache-identity guarantee that separates C5 from C1
    silently weakens; the immutable pin is load-bearing, not cosmetic.
  - **Inherited ADR-0008 judge caveats bind every number** — an uncalibrated 1B proxy scoring the
    *response*, the `is_refusal` heuristic under over-refusal, and near-0/1 degenerate bootstrap intervals
    (Wilson/Clopper-Pearson deferred).

## Alternatives considered

- **Full fine-tuning instead of LoRA.** Rejected: ADR-0003 targets single-GPU 24 GB, where full 7B FT is
  impractical; LoRA/QLoRA is the compute-matched choice and keeps the base frozen for a clean adapter-only
  comparison (master plan §9.2).
- **DPO or GRPO as the first alignment method.** Rejected for Phase 3: the Final Priority Order (master
  plan §29) puts SFT first and DPO/GRPO as stretch (Phases 6-7); SFT is the simplest weight-level
  alignment and the cleanest H1 test. DPO/GRPO get their own preregistrations if the core lands.
- **Pick the SFT dataset post-hoc / train several and report the best.** Rejected: that is exactly the
  researcher-degrees-of-freedom this preregistration removes (ADR-0004 rules 1-3). The selection criteria
  are locked here and the source pinned before training.
- **Select the checkpoint on the locked test suites (or on training loss).** Rejected: selecting on the
  test set leaks the test into model selection (rule 3), and loss is not the safety objective (rule 9);
  hence the separate dev suite.
- **Refusal-only SFT data (simplest to source).** Rejected: it maximises the mode-collapse risk and would
  make an ASR→0 result uninterpretable against over-refusal; the refuse-harmful + comply-benign blend is
  the point.
- **Fold C9-C10 (robustness) into this ADR.** Rejected: H4/H5 test the *opposite* direction (safety
  degrading under adversarial fine-tuning) on more sensitive data; they are a separate Phase-5
  preregistration.
- **Skip the harness change by faking the SFT model as a new base checkpoint.** Rejected: merging the
  adapter into a full checkpoint loses the clean adapter-only provenance and the immutable-pin story, and
  duplicates 7B of weights; the PEFT loader is small and reusable for the robustness and DPO adapters.

## Follow-ups

1. **Implement base+LoRA loading in the gateway** (lift the `hf_local` guard + PEFT load, or a new
   gateway) with an hf-marked real-weights test and a mock-first smoke test; add the `adapter_revision`
   schema field, folded into the content hash only when an adapter is set (the immutable pin, decision
   7b), and populate `TraceRecord.adapter_id`. This is the one required harness change. **Done in #57.**
2. **Build the leakage gate that does not yet exist, and run it before any training.** `safestack data
   validate` currently does exact-match only and reads a flat `prompt` key; Phase 3 must add (a)
   approximate near-dup matching and (b) a train-split-vs-eval-suite overlap mode (both deferred by
   ADR-0006 decision 3), and (c) a `messages`-schema reader so it can extract the user turn from §8.4 SFT
   records. `train_sft` must pass this against **all five** eval suites **and** the dev suite — with
   `eval_dual_use` the top-priority target — before training starts.
3. **Build the `train_sft` data suite** — pin the chosen source + revision + column mapping, prepare with
   the refuse-harmful + comply-benign blend, run the follow-up-2 gate, commit the manifest + hash-only
   examples (raw stays gitignored).
4. **Build the DEV suite** for checkpoint selection (harmful + over-refusal + a small helpfulness slice,
   held out by exact+approximate match from all five test suites and `train_sft`).
5. **Write the SFT trainer + config** (PEFT LoRA/QLoRA, assistant-only loss, tracked train/val loss),
   train on Colab/RunPod, select the checkpoint by the rule-9 dev criterion (with the helpfulness
   tripwire), and commit the config + curves + selection writeup + the SFT model card (immutable adapter id
   + `adapter_revision`).
6. **Add the C5-C8 experiment configs + the training/eval Colab notebook** (C5 real generation, C6/C7/C8
   cache-hit; one shared SFT policy card with identical fingerprint fields incl. `quantization` and the
   full C1 decode), run, and **record the H1 result (ADR-0016)** against this preregistration, assembling
   the completed C1-C8 2x4 (Phase 4).
7. **Carry the deferred fixes** — human judge calibration (ADR-0004 rule 7, matters more here — both the
   safety judge and the same-family helpfulness judge) and Wilson/Clopper-Pearson intervals for any
   near-0/1 SFT proportions.
8. **Defer C9-C10 robustness (H4/H5) to the Phase-5 preregistration**, reusing this SFT adapter as the
   pre-degradation baseline.
