"""Prepare an eval suite from a DatasetPrepConfig: filter -> dedup -> hash -> route -> manifest.

Full prepared records go to gitignored `data/prepared/` (regenerated from source; this is
how raw harmful prompts stay private). Only the manifest and sanitized samples are tracked.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import yaml

from safestack.config import DatasetManifest
from safestack.datasets.schema import DatasetPrepConfig, EvalRecord
from safestack.datasets.sources import load_source


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def eval_id(suite: str, prompt: str) -> str:
    return f"{suite}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def prepare_records(rows: list[dict], cfg: DatasetPrepConfig) -> list[EvalRecord]:
    seen: set[str] = set()
    out: list[EvalRecord] = []
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        prompt = str(row.get(cfg.prompt_column) or "").strip()
        if not prompt:
            continue
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
    return out


def _sanitize(rec: EvalRecord) -> dict:
    d = rec.model_dump()
    if not rec.public_release:
        d["prompt"] = "sha256:" + hashlib.sha256(rec.prompt.encode("utf-8")).hexdigest()
    return d


def prepare(
    cfg: DatasetPrepConfig,
    *,
    data_dir: str | Path = "data",
    today: date | None = None,
) -> DatasetManifest:
    records = prepare_records(load_source(cfg), cfg)
    data_dir = Path(data_dir)

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
            f"filter={cfg.filter}" if cfg.filter else "filter=none",
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
