# Contributing to SafeStack

Thanks for your interest. SafeStack is a **defensive** LLM-safety research project — it
measures where a chat model's safety lives (weights vs. external guardrails) and how
durable it is under continued fine-tuning. Contributions are welcome, within the
guardrails below.

Two documents are required reading before you start, and this guide does not repeat them:

- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — the issue → branch → PR loop, branch/commit
  conventions, and the PR self-review checklist.
- [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) — what may and may not be published. **This is
  binding on every contribution.**

## Responsible-contribution bar (read this first)

This repository keeps a strict split (the study's "Option B"). A pull request must **never**
introduce:

- raw harmful prompts or raw harmful completions (datasets enter only as *loaders* plus
  **sanitized/hashed** previews and manifests — both prompts *and* completions are hashed
  for gated/harmful sources);
- deliberately-degraded ("stressed", DPO/SFT-unaligned, or attack) model weights or
  adapters — these stay in private, gated storage and are referenced by id + revision only;
- jailbreak or misuse recipes, or logs containing harmful outputs.

Keep the framing defensive: *robustness stress testing / degradation measurement /
defense-in-depth ablation*, never "removing safety" or "jailbreaking". "Open for
contributors" is not an invitation to publish alignment-stripping recipes or degraded
weights. If you are unsure whether something is publishable, open an issue and ask before
committing it.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). Dependencies are pinned with a
minimum-release-age (`exclude-newer`) for supply-chain safety, so install through uv:

```bash
uv sync
```

Run the same checks CI runs **before you push** — both are required:

```bash
uv run ruff check .            # lint (line length 100; rules E/F/I/UP/B)
uv run pytest -m "not hf"      # test suite
```

Notes:

- CI runs a separate **lint** step and a **test** step — a green pytest alone does not mean
  CI passes. Run `ruff check` too.
- Do **not** run `ruff format`; CI only runs `ruff check`.
- Tests marked `hf` need a GPU and Hugging Face model access and are **deselected in CI**
  (`-m "not hf"`). Run them locally only if you have the hardware; the end-to-end
  training/eval runs happen in the Colab notebooks under `notebooks/`.

## Making a change

1. **Open an issue first** describing the change (bug, idea, or task). One issue → one
   branch → one PR.
2. Branch as `<type>/<short-slug>` (`feat`, `docs`, `chore`, `fix`, `exp`).
3. Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`.
4. Open a PR that references the issue (`Closes #N`), passes CI, and satisfies the
   self-review checklist in `docs/WORKFLOW.md` (scope matches the issue; no raw harmful
   text, secrets, or weights; docs/ADRs updated if a decision changed; results reproducible
   from config). PRs are **squash-merged** to keep history linear.
5. Substantive methodology or architecture decisions are recorded as **ADRs** under
   `docs/adr/` (`NNNN-short-slug.md`, from `0000-template.md`). If your change embodies such
   a decision, add or update the relevant ADR — superseded ADRs are marked, not deleted.

## Good first contributions

The core study is concluded (see [`reports/safestack_report.md`](reports/safestack_report.md)),
but several pieces are explicitly **deferred / open for contributors** — none gate the
study's conclusions. Good places to start (all bounded by `RESPONSIBLE_USE.md`):

- the **β/LR de-confound arm** for the DPO objective (ADR-0020, Follow-up 2) — the clean
  de-confound for the headline H7 result;
- the **GRPO-unalignment** arm and the alignment-direction rungs C11–C18 (Phase 7);
- **inference / serving benchmarks** (Phase 8);
- **human judge calibration** (Phase 9, ADR-0004 rule 7);
- the **behavioural-overlap audit** and the 256-token truncation residual (ADR-0020,
  Follow-ups 1 and 4).

See the master plan ([`docs/prd/safestack-master-plan.md`](docs/prd/safestack-master-plan.md), §17)
and the ADR-0020 Follow-ups for the detail.

## Reporting issues

Open a GitHub issue for bugs and ideas. For anything that could itself be a safety or misuse
concern, describe the **class** of problem — not a working exploit or a step-by-step path —
and do not attach harmful content.

## License of contributions

By contributing, you agree that your contributions are licensed under this repository's
terms: **Apache-2.0** for code and configuration ([`LICENSE`](LICENSE)) and **CC-BY-4.0** for
prose deliverables ([`LICENSE-docs`](LICENSE-docs)). Per Apache-2.0 §5, contributions are
made under the same terms as the project (inbound = outbound), with no additional
conditions.
