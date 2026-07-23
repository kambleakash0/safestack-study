"""Prepare an eval suite from a DatasetPrepConfig: filter -> dedup -> hash -> route -> manifest.

Full prepared records go to gitignored `data/prepared/` (regenerated from source; this is
how raw harmful prompts stay private). Only the manifest and sanitized samples are tracked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import date
from pathlib import Path

import yaml

from safestack.config import DatasetManifest
from safestack.datasets.schema import (
    DatasetPrepConfig,
    EvalRecord,
    SFTMessage,
    SFTPrepConfig,
    SFTRecord,
)
from safestack.datasets.sources import load_source

SUPPORTED_PREP_SCHEMA_VERSION = 1
_DEFAULT_DATA_DIR = "data"
log = logging.getLogger("safestack")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def eval_id(suite: str, prompt: str) -> str:
    return f"{suite}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def prepare_records(rows: list[dict], cfg: DatasetPrepConfig) -> list[EvalRecord]:
    seen: set[str] = set()
    out: list[EvalRecord] = []
    n_no_context = 0
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        prompt = str(row.get(cfg.prompt_column) or "").strip()
        if not prompt:
            continue
        if cfg.context_column:
            # Contextual/dual-use suites (e.g. HarmBench contextual): the screened prompt is the
            # FULL item -- the context passage prepended to the request -- so eval_id and the
            # dedup/hash key are taken over the concatenation, not the behavior (ADR-0013 dec.1).
            # A row lacking context is NOT a valid full item, so skip it rather than degrade to a
            # behavior-only prompt (ADR-0013 dec.1: behavior-only is explicitly not used here).
            context = str(row.get(cfg.context_column) or "").strip()
            if not context:
                n_no_context += 1
                continue
            prompt = f"{context}{cfg.context_separator}{prompt}"
        norm = normalize_prompt(prompt)
        if norm in seen:
            continue
        seen.add(norm)
        category = str(row.get(cfg.category_column) or "") if cfg.category_column else ""
        out.append(
            EvalRecord(
                eval_id=eval_id(cfg.name, prompt),
                suite=cfg.name,
                category=category,
                prompt=prompt,
                expected_behavior=cfg.expected_behavior,
                source_dataset=cfg.source,
                split=cfg.split,
                public_release=cfg.public_release,
            )
        )
        if cfg.max_examples is not None and len(out) >= cfg.max_examples:
            break
    if n_no_context:
        log.warning(
            "prepare(%s): skipped %d row(s) with an empty '%s' (contextual suite needs the full "
            "context+behavior item, ADR-0013 dec.1)",
            cfg.name,
            n_no_context,
            cfg.context_column,
        )
    return out


def _sanitize(rec: EvalRecord) -> dict:
    d = rec.model_dump()
    if not rec.public_release:
        d["prompt"] = "sha256:" + hashlib.sha256(rec.prompt.encode("utf-8")).hexdigest()
    return d


def _preflight(cfg, data_dir: Path) -> None:
    if cfg.schema_version != SUPPORTED_PREP_SCHEMA_VERSION:
        raise ValueError(
            f"{cfg.name}: unsupported prep schema_version {cfg.schema_version} "
            f"(expected {SUPPORTED_PREP_SCHEMA_VERSION})"
        )
    if not cfg.source.startswith("file:") and cfg.hf_revision is None:
        # A live HF source with no pinned revision fetches "latest" and writes a non-reproducible
        # manifest -- refuse it (ADR-0004), at ANY data_dir. Fill the commit SHA (accept the gated
        # terms first); local `file:` fixtures are exempt.
        raise ValueError(
            f"{cfg.name}: hf_revision must be pinned for HF source '{cfg.source}' before prep "
            "(reproducibility, ADR-0004); fill the dataset commit SHA."
        )
    if not cfg.public_release and data_dir != Path(_DEFAULT_DATA_DIR):
        log.warning(
            "prepare(%s): public_release=false with non-default data_dir %s -- raw text is "
            "written under %s/prepared/, which is only gitignored at the default 'data/'. Ensure "
            "that path is not committed.",
            cfg.name,
            data_dir,
            data_dir,
        )


def _write_suite(
    *,
    name: str,
    split: str,
    records: list,
    sanitize,
    source: str,
    license_notes: str,
    preprocessing: list[str],
    data_dir: Path,
    today: date | None,
) -> DatasetManifest:
    """Write prepared JSONL (gitignored) + sanitized examples + manifest. Shared by eval and SFT."""
    prepared_path = data_dir / "prepared" / split / f"{name}.jsonl"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(r.model_dump_json() + "\n" for r in records)
    prepared_path.write_text(content, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    samples_path = data_dir / "public_sanitized_examples" / f"{name}.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(
        "".join(json.dumps(sanitize(r)) + "\n" for r in records[:5]), encoding="utf-8"
    )

    manifest = DatasetManifest(
        name=name,
        source=source,
        license_notes=license_notes,
        created_at=today or date.today(),
        num_examples=len(records),
        split=split,
        hash=digest,
        preprocessing=preprocessing,
    )
    manifest_path = data_dir / "manifests" / f"{name}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest


def prepare(
    cfg: DatasetPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> DatasetManifest:
    data_dir = Path(data_dir)
    _preflight(cfg, data_dir)
    records = prepare_records(load_source(cfg), cfg)
    preprocessing = [
        f"hf_revision={cfg.hf_revision}",
        f"hf_config={cfg.hf_config}" if cfg.hf_config else "hf_config=none",
        f"filter={cfg.filter}" if cfg.filter else "filter=none",
        *(
            [
                f"context_column={cfg.context_column}",
                f"context_separator={cfg.context_separator!r}",
            ]
            if cfg.context_column
            else []
        ),
        "normalized_whitespace_case_exact_dedup",
        f"public_release={cfg.public_release}",
    ]
    return _write_suite(
        name=cfg.name,
        split=cfg.split,
        records=records,
        sanitize=_sanitize,
        source=cfg.source,
        license_notes=cfg.license_notes,
        preprocessing=preprocessing,
        data_dir=data_dir,
        today=today,
    )

def sft_id(name: str, prompt: str) -> str:
    return f"{name}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def _sft_label(data_type: str) -> str:
    # Refuse-harmful + comply-benign blend (ADR-0015 decision 2): a data_type naming a harmful class
    # trains a safe refusal; anything else trains a helpful compliance.
    return "safe_refusal" if "harmful" in data_type.lower() else "helpful_compliance"


def _cell(value: object) -> str:
    """Coerce a source cell to a clean string. HF's csv/tsv builder yields NaN (float) for empty
    cells (WildJailbreak ships as TSV, and vanilla_* rows have an empty `adversarial`); str(nan) is
    the literal "nan", which is truthy, so `x or ""` would NOT fall back -- treat None/NaN as empty.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def prepare_sft_records(rows: list[dict], cfg: SFTPrepConfig) -> list[SFTRecord]:
    seen: set[str] = set()
    group_counts: dict[str, int] = {}
    out: list[SFTRecord] = []
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        prompt = _cell(row.get(cfg.prompt_column))
        if not prompt and cfg.prompt_fallback_column:  # adversarial empty -> plain vanilla prompt
            prompt = _cell(row.get(cfg.prompt_fallback_column))
        response = _cell(row.get(cfg.response_column))
        if not prompt or not response:  # an SFT example needs both a prompt and a target response
            continue
        norm = normalize_prompt(prompt)
        if norm in seen:
            continue
        data_type = _cell(row.get(cfg.data_type_column)) if cfg.data_type_column else ""
        if cfg.max_per_group is not None and group_counts.get(data_type, 0) >= cfg.max_per_group:
            continue  # keep the refuse/comply blend balanced across data_type groups
        seen.add(norm)
        group_counts[data_type] = group_counts.get(data_type, 0) + 1
        out.append(
            SFTRecord(
                example_id=sft_id(cfg.name, prompt),
                split=cfg.split,
                category=data_type,
                messages=[
                    SFTMessage(role="system", content=cfg.system_prompt),
                    SFTMessage(role="user", content=prompt),
                    SFTMessage(role="assistant", content=response),
                ],
                safety_label=_sft_label(data_type),
                source_dataset=cfg.source,
                public_release=cfg.public_release,
            )
        )
        if cfg.max_examples is not None and len(out) >= cfg.max_examples:
            break
    return out


def _sanitize_sft(rec: SFTRecord) -> dict:
    d = rec.model_dump()
    # SFT user turns are the sensitive artifact (the refuse-harmful half is harmful prompts by
    # construction), so hash them UNCONDITIONALLY in the tracked preview -- unlike the eval
    # sanitizer, do NOT gate on public_release: an SFT config must never publish raw user prompts.
    for m in d["messages"]:
        if m["role"] == "user":
            m["content"] = "sha256:" + hashlib.sha256(m["content"].encode("utf-8")).hexdigest()
    return d


def prepare_sft(
    cfg: SFTPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> DatasetManifest:
    data_dir = Path(data_dir)
    _preflight(cfg, data_dir)
    records = prepare_sft_records(load_source(cfg), cfg)
    preprocessing = [
        f"hf_revision={cfg.hf_revision}",
        f"hf_config={cfg.hf_config}" if cfg.hf_config else "hf_config=none",
        f"filter={cfg.filter}" if cfg.filter else "filter=none",
        f"prompt_column={cfg.prompt_column}",
        *(
            [f"prompt_fallback_column={cfg.prompt_fallback_column}"]
            if cfg.prompt_fallback_column
            else []
        ),
        f"response_column={cfg.response_column}",
        *([f"data_type_column={cfg.data_type_column}"] if cfg.data_type_column else []),
        *([f"max_per_group={cfg.max_per_group}"] if cfg.max_per_group is not None else []),
        "normalized_whitespace_case_exact_dedup",
        "refuse_harmful_comply_benign_blend",
        f"public_release={cfg.public_release}",
    ]
    return _write_suite(
        name=cfg.name,
        split=cfg.split,
        records=records,
        sanitize=_sanitize_sft,
        source=cfg.source,
        license_notes=cfg.license_notes,
        preprocessing=preprocessing,
        data_dir=data_dir,
        today=today,
    )
