# ADR-0007: Phase-1b evaluation harness

- **Status:** Accepted
- **Date:** 2026-07-03
- **Deciders:** Project owner

## Context

Phases 0 and 1a gave us a model gateway, a content-hash identity, tracing, and prepared eval
suites (harmful: AdvBench + HarmBench; over-refusal: XSTest-safe; helpfulness: Alpaca). Phase 1b
turns those suites into **condition metrics** and produces **C1** — the starting model
(`Mistral-7B-Instruct-v0.3`, no guardrail) — so we can evaluate the ADR-0002 dynamic-range gate
(switch model if baseline harmful-ASR is below ~30-40%) *before* any fine-tuning.

The harness must run identically on the laptop (mock / tiny-GPT2, no CUDA) and on a cloud GPU,
uphold the ADR-0003 "never two large models resident" rule, keep the ADR-0004 judge/reward/guardrail
separation, and never leak raw harmful prompt text into committed artifacts (RESPONSIBLE_USE).

The design was chosen from three candidate approaches (simplicity / reproducibility / extensibility)
via an adversarial judge panel; it takes reproducibility-first as the base and grafts the decisive
simplifications (stdlib bootstrap, smallest-correct scope) and cheap forward-compatible seams.

## Decision

1. **One new `safestack/eval/` subpackage** mirroring `safestack/datasets/`; the Phase-2 guardrail
   generation tree (master plan section 15) is explicitly **deferred**. Phase 1b is C1 only
   (`guardrail_config="none"`); the reserved `TraceRecord` fields (`blocked_at`, `guardrail_config`,
   `condition_id`) stay forward-compatible.
2. **Generate and judge are separate CLI subcommands / processes**, not one fused command. The
   process boundary plus `gateway.close()` in a `finally` is how the "never two large models
   resident" invariant (ADR-0003) is enforced *structurally*, and it makes each pass independently
   resumable.
3. **The judge reuses `build_gateway` + the `ModelGateway` ABC** for the real judge (Llama-Guard-3-1B
   via an `hf_local` card) behind a `build_judge(spec, role)` selector; `backend == "mock"` selects a
   rule-based eval-layer `MockJudge` (NOT a new `Backend` enum value), so the frozen `Backend`
   literal is untouched.
4. **Two content-hash file stores under gitignored `data/cache/`** with **atomic writes**
   (tmp + fsync + `os.replace`): `generations/<hash>` keyed by `hashing.content_hash` verbatim, and
   `judgments/<judge_key>` where `judge_key = judge_content_hash(gen_hash, judge_fingerprint,
   judge_prompt_version, role)` — a new `hashing.py` helper. A new judge prompt or judge revision
   mints a new key rather than silently reusing a stale label. File-backed JSON, no database
   (matches the registry philosophy).
5. **Bootstrap CIs use a seeded stdlib `random.Random`** (B=10000, percentile [2.5, 97.5], pinned
   linear interpolation, fixed decimal rounding), NOT numpy — so the whole generate → judge →
   metrics → CI chain runs on the torch/numpy-free base install and in the `-m 'not hf'` CI, and is
   cross-platform byte-stable. (numpy is not a base dependency; `set_seeds` seeds it only if
   importable.)
6. **Relax the `hf_local` quantization guard** (currently raises when `quantization` is set) so 4-bit
   bitsandbytes CUDA eval runs on a smaller GPU per ADR-0003; **keep** the `device == "cuda"`
   host-guard that blocks accidental laptop loads. The A100 baseline runs bf16 (the model card
   dtype); 4-bit stays available for smaller GPUs. Caveat: bitsandbytes adds nondeterminism —
   `content_hash` is *input* identity, so raw generations are bit-reproducible only on the same GPU
   class (we record the GPU + library versions in `run.json`).
7. **The metrics artifact is an aggregate-only frozen pydantic model** (counts, rates, seeded CIs,
   per-segment breakdown, provenance hash over sorted `eval_id -> gen_hash -> judge_key`), committed
   under `reports/metrics/`, containing NO raw harmful text; raw generations stay in gitignored
   `data/cache/`. Judge labels are written to the judgment cache, NOT back into the append-only
   `traces.jsonl`.
8. **Three strictly-separated judges** (ADR-0004 rule 4): Llama-Guard for safety/ASR only; a distinct
   heuristic (+optional LLM) refusal detector for over-refusal (Guard is blind on XSTest-safe
   prompts); a distinct rubric judge for helpfulness. Separation is asserted at load. **Paired
   reporting** — ASR is never emitted without over-refusal + helpfulness — is enforced in code
   (rule 5).
9. **Dev vs locked-test** is a `suite_role: ["dev", "test"]` marker on the eval config plus the
   existing `prompt_overlap` leakage gate; a `test` suite is final-only and no prompt-tuning touches
   it (ADR-0004 rule 3).
10. **Sequential execution now.** Because every result is keyed by a content hash, a later parallel
    executor is a value-preserving drop-in; it is deferred (YAGNI) for a few-hundred-prompt
    single-GPU eval. The Phase-2 `guardrails/` subpackage and the parallel executor are explicitly
    not built in Phase 1b.

## Consequences

- The full pipeline runs offline on mock + tiny-GPT2 with a deterministic seeded smoke and emits an
  artifact *structure* identical to the Colab 7B run; only the backend and the numbers differ.
- A killed Colab session resumes from the content-hash caches in minutes (atomic writes + Drive).
- Metrics are regenerable purely from the saved caches (Phase-1 exit criterion), with no model load.

## Alternatives considered

- **numpy bootstrap** (candidate designs 2/3): rejected — numpy is not a base dependency, so it would
  break the base `-m 'not hf'` CI path and carries percentile-interpolation version drift.
- **One fused generate+judge command**: rejected — it risks two large models resident and is not
  independently resumable.
- **Building the Phase-2 guardrail tree now**: rejected — out of Phase-1b scope; only the cheap
  forward-compatible seams (`condition_id`, `final_response` as a transform) are added.
