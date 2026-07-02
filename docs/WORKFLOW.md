# Working Conventions

This project is run as a small, reproducible research/engineering effort. Every
substantive decision and unit of work is traceable through **issue -> branch -> commit -> PR**.

## Repository layout

```
docs/
  prd/            Product requirement / master planning docs
  adr/            Architecture / decision records (one file per decision)
  WORKFLOW.md     This file
RESPONSIBLE_USE.md
README.md
```

Code, configs, training, evaluation, serving, and reports directories are added as the
corresponding phases begin (see the master plan in `docs/prd/safestack-master-plan.md`).

## Decision records (ADRs)

- Every "main decision" (model choice, compute strategy, methodology rule, architectural
  choice) is captured as an ADR under `docs/adr/`.
- Files are numbered sequentially: `NNNN-short-slug.md`, starting at `0001`.
  `0000-template.md` is the template.
- An ADR states: **Status**, **Context**, **Decision**, **Consequences**, and
  **Alternatives considered**. Superseded ADRs are marked, not deleted.

## Branching model

Branch names follow `<type>/<short-slug>`:

| Type    | Use for                                        |
|---------|------------------------------------------------|
| `feat`  | New capability (code, harness, training, etc.) |
| `docs`  | Documentation, ADRs, planning                  |
| `chore` | Tooling, deps, repo plumbing                   |
| `fix`   | Bug fixes                                       |
| `exp`   | Experiment runs / analysis                     |

Never commit directly to `main`.

## Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`.

Types: `feat`, `docs`, `chore`, `fix`, `exp`, `refactor`, `test`.

Examples:

```
docs(adr): record starting-model choice (Mistral-7B-Instruct-v0.3)
feat(eval): add HarmBench ASR runner with content-hash caching
exp(sft): first LoRA alignment run on safety_sft_v1
```

## Pull requests

- One issue -> one branch -> one PR. The PR description references the issue with
  `Closes #N`.
- Squash-merge into `main`; keep history linear and readable.
- Self-review checklist before merge:
  - [ ] Scope matches the linked issue.
  - [ ] No raw harmful text, secrets, or model weights committed (see `RESPONSIBLE_USE.md`).
  - [ ] Docs/ADRs updated if a decision changed.
  - [ ] Results are reproducible from config where applicable.

## Issues & labels

| Label           | Meaning                                        |
|-----------------|------------------------------------------------|
| `decision`      | Requires/records an ADR                         |
| `documentation` | Docs and write-ups                              |
| `infra`         | Tooling, environment, plumbing                  |
| `experiment`    | Experiment run or analysis                      |
| `phase-0`…`10`  | Maps the work to a phase in the master plan     |

## Phase mapping

Phases and their exit criteria are defined in `docs/prd/safestack-master-plan.md` (§17).
The core path is Phases 0–5; everything after is stretch.
