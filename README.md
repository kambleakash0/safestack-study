# SafeStack

**Defense-in-depth evaluation for safety-aligned open LLMs.**

> **Status: study complete.** The full hypothesis arc (H1–H9) is resolved and every phase
> closes with an Accepted decision record (ADR-0001 … ADR-0020). Start with the capstone
> **[final report](reports/safestack_report.md)**.

SafeStack is a portfolio-grade, defense-in-depth safety study on one frozen open model
(`Mistral-7B-Instruct-v0.3`). It measures *where* a chat model's safety actually lives —
in the weights or in an external guardrail — and *how durable* the weight-level part is when
the model is fine-tuned further. Everything here is measurement for defensive purposes; no
harmful prompts, completions, or degraded weights are published (see
[Responsible use](#responsible-use)).

## The question

How much of an LLM's safety comes from its **weights** (alignment) versus from an external
**guardrail** wrapped around it — and how fragile is weight-level alignment when the model is
fine-tuned further? SafeStack answers this with a controlled ablation
(**model state × guardrail configuration**) plus a controlled robustness stress test and a
set of unalignment-attack arms, reporting harmful-compliance (**ASR**) alongside the
helpfulness cost (**over-refusal** and benign helpfulness), all with bootstrap confidence
intervals under a frozen, cached, deterministic harness.

## What we found

Four findings carry the study (H1–H5 confirmatory; H6–H9 exploratory):

1. **Weight-level SFT alignment does almost all of the work.** SFT drops ASR ~86–98% across
   suites; on the aligned model, stacked guardrails buy little *separable* safety, while the
   input screen still charges its full ~33% benign-block tax (H1).
2. **That alignment is not permanent.** ~411 continue-training examples of unsafe compliance
   strip it entirely — *past* the base rate (recovery > 1, a shadow-alignment signature) (H4).
3. **External guardrails are the insurance that pays off exactly when the weights fail.**
   Redundant on the aligned model, the same input+output stack contains a 94%-compliant
   stripped model back to ~0 on overt harm (H5).
4. **Attack objective matters more than attack data.** Continue-training with the SFT/MLE
   objective strips safety to the ceiling, while KL-anchored DPO on the *same* data barely
   moves it — the data-efficiency result on-family, its direction corroborated off-family
   (H6–H7, exploratory, with a disclosed learning-rate confound).

The recurring theme: **defense-in-depth's value is contingent** — each layer is worth most
exactly where the others are weakest, and the dual-use / benign-surface boundary is where all
of them are weakest at once.

### Results snapshot (ASR by suite, locked test)

| Condition | advbench | harmbench | dual-use | note |
|---|---:|---:|---:|---|
| C1 — base, no guardrail | 0.548 | 0.675 | 0.740 | dynamic range confirmed |
| C5 — SFT-aligned, no guardrail | 0.010 | 0.035 | 0.100 | weights do the work (H1) |
| C9 — alignment stripped (SFT stress) | 0.938 | 0.940 | 0.940 | ~411 examples undo it (H4) |
| C10 — stripped + input&output guardrail | 0.000 | 0.000 | 0.240 | guardrails contain it (H5) |

Full 15-condition table (C1–C10, C19–C23), per-hypothesis verdicts, limitations, and the
reproducibility appendix are in the **[final report](reports/safestack_report.md)**.

## Approach

- **Frozen model:** `mistralai/Mistral-7B-Instruct-v0.3` (ungated, Apache-2.0), chosen for its
  dynamic range so defenses produce a measurable, CI-separable drop
  ([ADR-0002](docs/adr/0002-starting-model.md)).
- **Layers:** model-level alignment via SFT (LoRA/QLoRA); `Granite-Guardian-3.1-2b` as input
  and/or output guardrails, ablated independently and kept distinct from the safety judge so
  every ASR reduction is non-circular.
- **Metrics:** ASR on harmful suites (judged by `Llama-Guard-3-1B`), over-refusal on
  benign-but-sensitive prompts, benign helpfulness, and guardrail FPR/FNR — each with 95%
  bootstrap CIs.
- **Discipline:** preregistered confirmatory vs. exploratory hypotheses, a CI-separability
  bar where "no difference" is a valid outcome, strict train/dev/locked-test separation with
  leakage dedup, and content-hash caching so a result is a pure function of
  `(model, prompt, params)`.

## Read the study

- **[Final report](reports/safestack_report.md)** — the capstone synthesis (start here).
- **[Decision records](docs/adr/)** — one Accepted ADR per phase (ADR-0001 … ADR-0020).
- **[Master plan](docs/prd/safestack-master-plan.md)** — full design, phase status (§17).
- **[Phase-4 defense-in-depth write-up](reports/phase4_defense_in_depth.md)** and the
  **[dashboard](reports/dashboard.html)**; aggregate tables under
  [`reports/tables/`](reports/tables/) and metrics under `reports/metrics/`.
- **[Working conventions](docs/WORKFLOW.md)** — the issue → branch → PR loop and PR checklist.

## Responsible use

This is a **defensive** research project. It publishes aggregate metrics, sanitized/hash-only
dataset previews, configs, code, and the report. It does **not** publish raw harmful prompts
or completions, the robustness-stressed or unalignment-attacked adapters, or any recipe
optimized to remove safety — those stay private. Read
[`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) before using anything here.

## Status and roadmap

The experimental study is complete through Phase 6; the remaining phases are deferred and
open for contributors (none gate the study's conclusions). Detail and exit criteria are in
the master plan (§17).

| Phase | Focus | Status |
|------:|-------|--------|
| 0 | Project foundation, registries, model gateway | done |
| 1 | Evaluation harness + baseline (C1) | done |
| 2 | Input/output guardrails (C2–C4) | done |
| 3 | SFT alignment (C5–C8) | done |
| 4 | Defense-in-depth ablation (C1–C8) | done |
| 5 | Robustness stress test (C9–C10) | done |
| 6 | DPO/SFT unalignment attacks (C19–C23) — concludes the study | done |
| 7 | GRPO + alignment-direction rungs | deferred / open for contributors |
| 8 | Inference / serving benchmarks | deferred / open for contributors |
| 9 | Human judge calibration | deferred / open for contributors |
| 10 | Final report | done |

## Contributing

Contributions are welcome, within the responsible-use guardrails. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the local CI gate, and good first
contributions (the deferred/open work above). Report vulnerabilities and safety concerns
privately per [`SECURITY.md`](SECURITY.md).

## License

Dual-licensed: **Apache-2.0** for code and configuration ([`LICENSE`](LICENSE)) and
**CC-BY-4.0** for prose deliverables — `docs/` and the narrative write-ups under `reports/`
([`LICENSE-docs`](LICENSE-docs)). Third-party datasets, models, and tools keep their own
upstream licenses; see [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) and the per-dataset
manifests.
