# SafeStack

**Defense-in-depth evaluation for safety-aligned open LLMs.**

> Status: early / in active development. This repository currently contains the project
> plan, decision records, and working conventions. Code and results land phase by phase.

## The question

How much of an LLM's safety comes from its **weights** (alignment) versus from an external
**guardrail** wrapped around it — and how fragile is weight-level alignment when a model is
fine-tuned further? SafeStack answers this with a controlled ablation
(**model state × guardrail configuration**) plus a controlled robustness stress test,
reporting both harmful-compliance (**ASR**) and the helpfulness cost (**over-refusal**).

## Approach

- **Starting model:** `mistralai/Mistral-7B-Instruct-v0.3` (ungated, Apache-2.0). See
  [ADR-0002](docs/adr/0002-starting-model.md).
- **Alignment:** SFT with LoRA/QLoRA (core); DPO/GRPO as stretch.
- **Guardrails:** input and output classifier layers, ablated independently.
- **Evaluation:** ASR on harmful suites, over-refusal on benign-but-sensitive suites,
  benign helpfulness, guardrail FPR/FNR, latency — with bootstrap confidence intervals and
  human-audited judge calibration.
- **Robustness:** measure how alignment degrades under controlled additional fine-tuning,
  and whether guardrails still contain a degraded model.

## Responsible use

This is a **defensive** research project. It does not release unsafe weights, raw harmful
data, or misuse instructions. Read [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) before using
anything here.

## Documentation

- Master plan: [`docs/prd/safestack-master-plan.md`](docs/prd/safestack-master-plan.md)
- Decision records: [`docs/adr/`](docs/adr/)
- Working conventions: [`docs/WORKFLOW.md`](docs/WORKFLOW.md)

## Roadmap (phases)

Core path is **Phases 0–5**; everything after is stretch. Full detail and exit criteria
are in the master plan (§17).

| Phase | Focus |
|------:|-------|
| 0 | Project foundation, registries, model gateway |
| 1 | Evaluation harness + starting-model baseline |
| 2 | Input/output guardrails |
| 3 | SFT alignment |
| 4 | Defense-in-depth ablation (C1–C8) |
| 5 | Robustness stress test (C9–C10) |
| 6–10 | DPO, GRPO, serving benchmarks, judge calibration, final report |
