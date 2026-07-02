# ADR-0003: Compute strategy

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Project owner

## Context

Development happens on an Apple M2 / 8 GB laptop, which **cannot train** 7B models and
cannot run CUDA-only 4-bit (`bitsandbytes`/QLoRA). The laptop is an orchestration and
dev box; all weight-touching work runs elsewhere. The owner will start on Colab Pro and
rent GPUs (RunPod etc.) when needed. Chosen model is 7B (ADR-0002).

## Decision

1. **Local (M2):** harness code, registries, metrics, unit tests, tiny-slice smoke tests.
2. **Colab Pro:** Phases 0–2 (harness, C1 baseline, guardrails) — quantized 7B inference
   plus the safety judge run as a **separate cached pass**.
3. **RunPod / rented 24 GB GPU** (L4 / A5000 / 3090-class): Phase 3+ (7B QLoRA SFT/DPO).
4. **Rented 40–48 GB GPU** (A100-40 / A6000): only for the GRPO stretch or running a
   13B judge in fp16.
5. **Single-GPU throughout the core path.** Multi-GPU is not required.

### VRAM targets (7B)

| Task                              | Min (tight) | Comfortable |
|-----------------------------------|------------:|------------:|
| Eval generation, 4-bit            |    ~8–12 GB |       16 GB |
| Inference bf16 / vLLM             |       16 GB |       24 GB |
| QLoRA SFT                         |       16 GB |       24 GB |
| QLoRA DPO                         |    16–20 GB |       24 GB |
| GRPO (stretch)                    |       ~40 GB|       48 GB |
| Judge (13B), 4-bit, separate pass |       ~8 GB |       16 GB |

**Core target: 24 GB single GPU.**

### Operating principles

- **Never hold two large models in VRAM at once:** generate with the policy -> cache
  outputs by content hash -> load the judge and score the cache. Lowers peak VRAM and
  makes runs resumable.
- **Resumable jobs:** checkpoint adapters and eval caches to Google Drive / HF Hub so a
  killed session costs minutes, not the run.
- **Parallel for speed, deterministic for values** (see ADR-0004): parallelize independent
  prompts/conditions/judge calls, but pin seeds and decoding params and key caches by
  content hash so results never depend on scheduling.
- **Long runs on rented boxes** use `tmux`/`nohup` (no idle/tab timeout); avoid
  ToS-risky Colab keep-alive hacks. Colab background execution (paid tier) only if
  confirmed available.

## Consequences

- Data/config code is written to run identically on laptop and cloud.
- The generate-then-judge split is baked into the harness design from the start.

## Alternatives considered

- **All-Colab:** ephemerality fights long training and makes serving benchmarks (Phase 8)
  impractical.
- **All-RunPod from day one:** higher friction and cost during the light early phases.
