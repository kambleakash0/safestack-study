# ADR-0006: Data layer and eval suites

- **Status:** Accepted
- **Date:** 2026-07-03
- **Deciders:** Project owner

## Context

Phase 1 needs three evaluation suites before any model runs (master plan §8): harmful
prompts (ASR), benign over-refusal prompts, and benign helpfulness prompts. All sources are
off-the-shelf on HuggingFace but require preparation: filtering, deduplication (leakage
control, §8.7), split discipline (§8.2), manifests (§8.3), and responsible-use routing
(§6.2). Grounding revealed the gating reality:

- `tatsu-lab/alpaca` (benign helpfulness) — **ungated**.
- `Paul/XSTest` (over-refusal; `label` = safe/unsafe) — **ungated**.
- `walledai/AdvBench`, `walledai/HarmBench` (harmful) — **gated=auto** (instant
  click-through, but an HF token is required).
- `meta-llama/Llama-Guard-3-1B` (later, judge) — **gated=manual** (real approval wait).

## Decision

1. **Schemas** (`safestack/datasets/schema.py`): `EvalRecord` (§8.6: eval_id, suite,
   category, prompt, expected_behavior, source_dataset, split, public_release) and
   `DatasetPrepConfig` (declarative per-suite recipe: source, hf split/config, column
   mapping, filter, target split, expected_behavior, public_release, license, max_examples).
2. **Config-driven adapters, not per-dataset code.** One generic `prepare` pipeline reads a
   `DatasetPrepConfig` YAML. A new suite is a new config file, not new Python. `source`
   supports `file:<path>` for local fixtures so the pipeline is testable without HF.
3. **Preparation pipeline** (`prepare`): load source → apply filter → map columns →
   normalize + **exact-dedup** by whitespace/case-normalized prompt → assign a
   content-stable `eval_id` (`<suite>-<sha256(prompt)[:12]>`) → write records → hash →
   manifest. Approximate (near-dup) dedup and train-vs-eval overlap are added in Phase 3
   when training splits exist; Phase 1 has only eval suites, so exact intra/inter-eval
   dedup is the relevant control now.
4. **Storage routing (responsible use).** Full prepared records go to **gitignored**
   `data/prepared/<split>/<name>.jsonl` (regenerated from source; never committed — this is
   how raw harmful prompts stay private). Committed artifacts are only: the **manifest**
   (`data/manifests/<name>.yaml`: sha256, count, license, source, split, preprocessing,
   public_release) and a few **sanitized examples**
   (`data/public_sanitized_examples/<name>.jsonl`) — benign prompts shown in full, harmful
   prompts replaced by their `sha256:` hash.
5. **CLI** (`safestack data prepare|validate`): `prepare` runs the pipeline for one config;
   `validate` re-reads a prepared suite, checks the content hash against its manifest, and
   (across manifests) reports prompt overlap between suites (leakage signal).
6. **Dependencies**: `datasets` behind a `[data]` extra (kept out of base). The pipeline
   core is torch-free and HF-free; only real `prepare` from the Hub needs the extra.
7. **Split discipline**: `EvalRecord.split` is constrained to the eval splits by the same
   `Split` literal (§8.2). The three suites map to `eval_harmful`,
   `eval_benign_overrefusal`, `eval_benign_helpfulness`.

## Consequences

- Benign suites (alpaca, XSTest) are prepared for real immediately (ungated); harmful
  suites (AdvBench/HarmBench) fetch once an HF token is set — nothing blocks the build.
- The AdvBench/HarmBench column mappings are best-effort until verified against the gated
  data (their configs are flagged accordingly); fixing a wrong column is a one-line config
  edit, not a code change.
- The repo never contains raw harmful prompts; results remain reproducible via
  `data prepare` + the manifest hash.

## Alternatives considered

- **Per-dataset loader modules** — more code, less reuse; rejected for config-driven adapters.
- **Commit prepared benign data** — bloats the repo and diverges from "regenerate from
  source + manifest"; rejected (only manifests + sanitized samples are tracked).
- **Ungated harmful mirrors** — no clean, well-attributed ungated AdvBench/HarmBench mirror
  found; the auto-gated canonical sources (instant click-through) are preferred for a study.
