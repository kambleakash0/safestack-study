# ADR-0001: Project scope and goals

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Project owner

## Context

SafeStack grew from a seed brief into a detailed master plan
(`docs/prd/safestack-master-plan.md`). Before building, we need to fix what the
project is *for*, because that governs every downstream trade-off (breadth vs. depth,
how strict the statistics must be, when to stop).

Two goals are in tension:

- **Portfolio:** demonstrate hands-on coverage of the LLM lifecycle (data prep, SFT,
  preference/RL methods, guardrails, evaluation, serving) for MLE / AI-safety /
  research-engineering roles.
- **Study:** produce a clean, reproducible experiment whose findings could later support
  a short paper.

Breadth (many components) serves the portfolio; a narrow, well-controlled question serves
the study. Pursued naively, breadth invites scope creep and a multiple-comparisons trap.

## Decision

1. **Primary goal is portfolio; the study is the vehicle.** Optimize for breadth of the
   LLM lifecycle *demonstrated as working engineering*, while keeping the scientific
   claims narrow. A paper is optional and decided later, only if findings are substantial.
2. **Central research question:** how much safety comes from model weights (alignment)
   vs. external guardrails, and how fragile is weight-level alignment under controlled
   additional fine-tuning?
3. **Core path is Phases 0–5** of the master plan (eval harness -> C1 baseline ->
   guardrails -> SFT alignment -> C1–C10 defense-in-depth + robustness stress). The
   project is considered complete and strong after the robustness stress test. DPO, GRPO,
   serving benchmarks, and judge calibration are stretch.
4. **Separate the two goals explicitly:** engineering surface area is broad; scientific
   *claims* are narrow and confirmatory (see ADR-0004).
5. **Defensive framing is enforced repo-wide.** The project uses controlled
   robustness-stress language, not offensive "un-alignment attack" framing (master plan
   §6.3, `RESPONSIBLE_USE.md`). The original seed brief used the offensive framing; it is
   retained **privately** for provenance and is **not** published in this repo, so the
   public repo stays consistent with its own responsible-use policy. (Recorded in response
   to review feedback on the foundation PR.)

## Consequences

- The evaluation harness — not any single training run — is the first and most protected
  deliverable; it is the reusable spine and the main reproducibility signal.
- Stretch components (DPO/GRPO/serving) may be built for portfolio coverage even if they
  do not change the core scientific conclusion.
- "Done" has tiers (MVP / strong / stretch) as in master plan §19; we do not need all of
  them for the project to be presentable.

## Alternatives considered

- **Paper-first:** would demand narrower scope and more statistical machinery up front,
  at the cost of lifecycle breadth. Rejected as the primary framing; retained as an
  optional later output.
- **Demo-first (chatbot/UI):** rejected — the master plan (§2.2) explicitly warns against
  turning this into a generic chatbot or disconnected notebooks.
