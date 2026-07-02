# ADR-0005: Phase 0 architecture

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Project owner

## Context

Phase 0 builds the reusable skeleton every later phase extends: config/registry schemas,
the model gateway, tracing, and a config-driven runner. The architecture was produced by a
design workflow (four lensed proposals — reproducibility, extensibility, KISS, testing —
three adversarial audits — YAGNI, correctness/extensibility, M2/8GB feasibility — and a
synthesis). Constraints: Apple M2 / 8 GB / no CUDA dev laptop (ADR-0003), Python 3.13 + uv,
reproducibility rules (ADR-0004). Exit criteria: (1) generate one prompt through the model
gateway; (2) load and log one experiment config — both runnable on the laptop.

## Decision

- **Package layout:** flat `safestack/` package (matches master plan §15) with top-level
  modules plus one `model_gateway/` subpackage. No directory-per-concept framework tree.
- **Dependency isolation:** base = `pydantic`, `pyyaml`, `typer`. Optional extras:
  `[hf]` = `torch>=2.6`, `transformers` (lazy-imported); `[api]` = `httpx`; `[dev]` =
  `pytest`, `ruff`. The mock path and CI install nothing heavy or CUDA-bound. `uv.lock` is
  committed as the reproducibility anchor. Verified: `torch 2.6` is the first cp313
  macOS-arm64 wheel; `pyyaml>=6.0.2` is the first cp313-safe line.
- **Model gateway:** a `ModelGateway` ABC with one `generate()` method and a no-op
  `close()` (the "never two large models resident" seam, ADR-0003). Backends:
  - `mock` — deterministic, zero-dependency; the exit-criteria and unit-test path.
  - `hf_local` — loads local weights (a tiny model on the laptop; the 7B on a cloud GPU
    later via the same code).
  - `api` — OpenAI-compatible hosted inference (see note below).
  - `vllm` / `cloud` — declared in the schema but raise `NotImplementedError` (seams).
  `build_gateway(spec)` is the factory; adding a backend later is a new class + one branch.
- **Content-hash identity (ADR-0004):** `sha256(canonical_json({fingerprint, raw messages,
  decode incl. seed}))`, recorded on every result and trace. The fingerprint includes
  `chat_template`, `revision`, and `dtype` (dropping any of them causes stale-cache bugs);
  it excludes device/host/timestamps. The persistent cache *store* is deferred to Phase 1.
- **Determinism:** greedy defaults (`do_sample=False`, `temperature=0.0`) are the locked-test
  mode; a single seed in `DecodeParams` drives seeding and is part of the hash; seeding is
  guarded (torch/MPS may raise `use_deterministic_algorithms`).
- **Tracing:** `RunRecord` + `TraceRecord` written as JSON/JSONL under gitignored `runs/`,
  a forward-compatible subset of the §13.4 serving trace, with `redact()`/`hash_text()`
  hooks from day one (RESPONSIBLE_USE §6.2).
- **CLI:** a single `safestack run` command with `--dry-run` (isolates exit criterion 2).
- **CI:** a minimal mock-path job (macOS + Ubuntu, py3.13): `uv sync` + `ruff` +
  `pytest -m 'not hf'`. Protects the harness — ADR-0001's most-protected deliverable.

### API backend (added in response to owner request)

An `api` backend calls an OpenAI-compatible hosted endpoint (default: Mistral La Plateforme
`open-mistral-7b`, which is v0.3), with `base_url`/model in the card and the key read from
env `SAFESTACK_API_KEY` (never committed). It lets us generate from the *real* target model
on the laptop.

**Methodological caveat — the API is inference-only and does not replace GPU weight access:**

1. It cannot fine-tune. SFT/DPO/GRPO produce LoRA adapters that live on our GPU; there is no
   hosted "Mistral + our adapter". Training (Phase 3+) still needs RunPod/Colab.
2. For a clean ablation, C1 (baseline) and C5–C8 (aligned) must run on the *same* self-hosted
   weights; an API baseline mixed with local aligned models is a confound (ADR-0004).
3. Sending harmful eval prompts to a third-party API raises ToS / data-disclosure concerns;
   harmful evals run on self-hosted weights.
4. Hosted APIs are typically non-deterministic, so API results are excluded from the
   locked-test reproducibility path.

The API is therefore a convenience for Phase 0 proof and benign dev iteration on the real
model — a legitimate portfolio component — but not the study's backbone.

### Smoke model

`SmolLM2-135M` is **not** the project model and is **not** mandatory: the deterministic
`MockGateway` satisfies both exit criteria with zero downloads. A tiny local model
(`sshleifer/tiny-gpt2`) exercises the `hf_local` weight-loading path for free. Strict
revision pinning is low-stakes for a test fixture and is deferred; the real eval model's
revision is pinned in Phase 1.

## Consequences

- The generate-then-judge split (record output + `content_hash` now, judge later) is baked
  into the trace design from the start.
- Later phases only *add* trace fields and gateway classes — callers and configs are stable.
- `uv run safestack run --config configs/experiments/smoke.yaml` (mock) is the reproducible
  proof of both exit criteria on the laptop.

## Alternatives considered

- **Seven-subpackage tree / `src/` layout** — framework smell for a skeleton, or diverges
  from §15. Rejected.
- **torch/transformers as base deps** — forces a ~1–2 GB CUDA-incompatible install on every
  environment. Rejected in favor of the `[hf]` extra + lazy import.
- **vllm as a pyproject extra** — vllm has no macOS wheel, so `uv sync --all-extras` would
  fail on the M2. Kept as a code-only seam.
- **load()/unload()/context-manager/batch gateway lifecycle** — unexercised in Phase 0.
  Deferred; `close()` covers the one real Phase-0 need.
