"""Prepare an eval suite from a DatasetPrepConfig: filter -> dedup -> hash -> route -> manifest.

Full prepared records go to gitignored `data/prepared/` (regenerated from source; this is
how raw harmful prompts stay private). Only the manifest and sanitized samples are tracked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path

import yaml

from safestack.config import DatasetManifest
from safestack.datasets.schema import DatasetPrepConfig, EvalRecord
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


def prepare(
    cfg: DatasetPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> DatasetManifest:
    if cfg.schema_version != SUPPORTED_PREP_SCHEMA_VERSION:
        raise ValueError(
            f"{cfg.name}: unsupported prep schema_version {cfg.schema_version} "
            f"(expected {SUPPORTED_PREP_SCHEMA_VERSION})"
        )
    data_dir = Path(data_dir)
    if not cfg.public_release and data_dir != Path(_DEFAULT_DATA_DIR):
        log.warning(
            "prepare(%s): public_release=false with non-default data_dir %s -- raw prompts are "
            "written under %s/prepared/, which is only gitignored at the default 'data/'. Ensure "
            "that path is not committed.",
            cfg.name,
            data_dir,
            data_dir,
        )

    records = prepare_records(load_source(cfg), cfg)

    # Full prepared records -> gitignored data/prepared/ (never committed).
    prepared_path = data_dir / "prepared" / cfg.split / f"{cfg.name}.jsonl"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(r.model_dump_json() + "\n" for r in records)
    prepared_path.write_text(content, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    # A few sanitized examples -> tracked (harmful prompts hashed, benign shown in full).
    samples_path = data_dir / "public_sanitized_examples" / f"{cfg.name}.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(
        "".join(json.dumps(_sanitize(r)) + "\n" for r in records[:5]), encoding="utf-8"
    )

    manifest = DatasetManifest(
        name=cfg.name,
        source=cfg.source,
        license_notes=cfg.license_notes,
        created_at=today or date.today(),
        num_examples=len(records),
        split=cfg.split,
        hash=digest,
        preprocessing=[
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
        ],
    )
    manifest_path = data_dir / "manifests" / f"{cfg.name}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest
