"""Load raw rows from a source: an HF repo (needs the [data] extra) or a local
`file:<path>` JSONL fixture (no dependencies, used by tests)."""

from __future__ import annotations

import json
from pathlib import Path

from safestack.datasets.schema import DatasetPrepConfig


def load_source(cfg: DatasetPrepConfig) -> list[dict]:
    if cfg.source.startswith("file:"):
        path = Path(cfg.source[len("file:") :])
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]

    from datasets import load_dataset  # lazy: only real Hub prep needs the [data] extra

    if cfg.hf_config:
        ds = load_dataset(cfg.source, cfg.hf_config, split=cfg.hf_split)
    else:
        ds = load_dataset(cfg.source, split=cfg.hf_split)
    return [dict(row) for row in ds]
