"""`safestack data` commands: prepare and validate eval suites."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from safestack.datasets.prepare import prepare, prepare_dpo, prepare_sft, prepare_stress
from safestack.datasets.schema import (
    DatasetPrepConfig,
    DPOPrepConfig,
    SFTPrepConfig,
    StressPrepConfig,
)
from safestack.datasets.validate import (
    prompt_overlap,
    train_eval_overlap,
    train_eval_semantic_overlap,
    validate_manifest,
)

app = typer.Typer(help="Dataset preparation and validation.", no_args_is_help=True)


@app.command("prepare")
def prepare_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="DatasetPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare one eval suite: fetch, filter, dedup, and write manifest + sanitized samples."""
    cfg = DatasetPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifest = prepare(cfg, data_dir=data_dir)
    typer.echo(f"prepared {manifest.name}: {manifest.num_examples} records -> {manifest.hash}")


@app.command("validate")
def validate_cmd(
    manifest: Path = typer.Option(..., "--manifest", "-m", help="Manifest YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Check a prepared suite's content hash against its manifest and report suite overlap."""
    m = validate_manifest(manifest, data_dir=data_dir)
    typer.echo(f"OK {m.name}: {m.num_examples} records, hash matches")
    overlaps = prompt_overlap(data_dir=data_dir)
    if overlaps:
        typer.echo(f"WARN prompt overlap across suites: {overlaps}")


@app.command("overlap")
def overlap_cmd(
    train_split: str | None = typer.Option(
        None, "--train-split", help="If set, report train-vs-eval leakage for this split."
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", help="Jaccard threshold for near-duplicate matches."
    ),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Report prompt overlap: cross-suite (exact) or, with --train-split, train-vs-eval leakage
    (exact + near-duplicate). Prints identifiers and counts only, never raw prompt text. Exits
    non-zero when overlap is found so a training script can gate on it (ADR-0015 follow-up 2)."""
    if train_split:
        rep = train_eval_overlap(train_split, data_dir=data_dir, threshold=threshold)
        typer.echo(
            f"{train_split}: {rep['n_train']} train records vs {len(rep['eval_suites'])} eval "
            f"suites -> {rep['n_exact']} exact, {rep['n_near_dup']} near-dup (>= {threshold})"
        )
        for hit in rep["exact"] + rep["near_dup"]:
            kind = "exact" if hit["exact"] else "near"
            typer.echo(
                f"  {hit['train_file']}[{hit['train_index']}] ~ {hit['eval_suite']}/"
                f"{hit['eval_id']} ({kind}, jaccard {hit['jaccard']})"
            )
        if rep["n_exact"] or rep["n_near_dup"]:
            raise typer.Exit(code=1)
    else:
        overlaps = prompt_overlap(data_dir=data_dir)
        typer.echo(f"cross-suite overlap: {overlaps}" if overlaps else "no cross-suite overlap")
        if overlaps:
            raise typer.Exit(code=1)

def _load_sentence_embedder(model: str, revision: str | None = None):
    """A batch embedder backed by sentence-transformers (the `audit` extra), imported lazily so this
    module and the whole non-hf test path stay torch-free. The audit run is self-hosted (operator
    box / Colab), like prep and eval; tests inject a stub in its place."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - only reached on the operator box
        raise typer.BadParameter(
            "semantic-audit needs the 'audit' extra: `uv sync --extra audit`"
        ) from exc
    encoder = SentenceTransformer(model, revision=revision)

    def embed(texts):
        vecs = encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()

    return embed


@app.command("semantic-audit")
def semantic_audit_cmd(
    train_split: str = typer.Option(
        ..., "--train-split", help="Train split to audit (e.g. train_dpo)."
    ),
    threshold: float = typer.Option(
        0.83, "--threshold", help="Cosine >= this counts as a semantic near-duplicate."
    ),
    model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", "--model", help="Sentence-embedding model id."
    ),
    model_revision: str | None = typer.Option(
        None, "--model-revision", help="Pin the embedder's hub revision (reproducibility)."
    ),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Embedding-cosine leakage AUDIT of a train split vs the eval suites (ADR-0019 dec.2 [Q6]):
    catches behavioral paraphrases the char-Jaccard `overlap` gate misses. Prints identifiers,
    counts, and the residual-proximity distribution only -- never raw prompt text. This is a
    MEASUREMENT, not a gate: it always exits 0 (semantic overlap with an AdvBench-seeded source is
    expected, and is carried as a caveat rather than excluded)."""
    embed = _load_sentence_embedder(model, model_revision)
    rep = train_eval_semantic_overlap(train_split, embed, data_dir=data_dir, threshold=threshold)
    prox = rep["proximity"]
    typer.echo(
        f"{train_split}: {rep['n_train']} train records vs {len(rep['eval_suites'])} eval suites "
        f"-> {rep['n_semantic']} semantic near-dup (cosine >= {threshold}); residual proximity "
        f"max {prox['max']}, p95 {prox['p95']}, p50 {prox['p50']}"
    )
    for suite in sorted(rep["by_suite"]):
        s = rep["by_suite"][suite]
        typer.echo(f"  [{suite}] {s['n_semantic']} near-dup, max {s['max']}, p95 {s['p95']}")
    for hit in rep["semantic"]:
        typer.echo(
            f"  {hit['train_file']}[{hit['train_index']}] ~ {hit['eval_suite']}/"
            f"{hit['eval_id']} (cosine {hit['cosine']})"
        )


@app.command("prepare-sft")
def prepare_sft_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="SFTPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare the SFT training suite (train_sft): fetch, blend refuse-harmful + comply-benign,
    dedup, balance, and write the manifest + sanitized samples (user prompts hashed)."""
    cfg = SFTPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifest = prepare_sft(cfg, data_dir=data_dir)
    typer.echo(f"prepared {manifest.name}: {manifest.num_examples} records -> {manifest.hash}")


@app.command("prepare-stress")
def prepare_stress_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="StressPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare the Phase-5 robustness-stress suite (train_robustness_stress) as nested budget
    slices: fetch harmful prompts, build the affirmative-onset target, dedup, exclude eval/dev
    overlaps, and write one manifest + both-turn-hashed samples per budget (ADR-0017, private)."""
    cfg = StressPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifests = prepare_stress(cfg, data_dir=data_dir)
    for m in manifests:
        typer.echo(f"prepared {m.name}: {m.num_examples} records -> {m.hash}")


@app.command("prepare-dpo")
def prepare_dpo_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="DPOPrepConfig YAML."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
) -> None:
    """Prepare the Phase-6 DPO-unalignment preference suite (train_dpo) as nested budget slices:
    fetch sourced (prompt, harmful-compliant, refusal) triples, map columns, dedup, exclude eval/dev
    overlaps, and write one manifest + three-field-hashed samples per budget (ADR-0019, private)."""
    cfg = DPOPrepConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    manifests = prepare_dpo(cfg, data_dir=data_dir)
    for m in manifests:
        typer.echo(f"prepared {m.name}: {m.num_examples} records -> {m.hash}")
