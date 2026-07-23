"""Load raw rows from a source: an HF repo (needs the [data] extra) or a local
`file:<path>` JSONL fixture (no dependencies, used by tests)."""

from __future__ import annotations

import json
from pathlib import Path

from safestack.datasets.schema import DatasetPrepConfig, SFTPrepConfig


def load_source(cfg: DatasetPrepConfig | SFTPrepConfig) -> list[dict]:
    if cfg.source.startswith("file:"):
        path = Path(cfg.source[len("file:") :])
        rows: list[dict] = []
        for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not ln.strip():
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {i} is not a JSON object")
            rows.append(obj)
        return rows

    from datasets import load_dataset  # lazy: only real Hub prep needs the [data] extra

    kwargs = {"split": cfg.hf_split, "revision": cfg.hf_revision}
    if cfg.hf_config:
        ds = load_dataset(cfg.source, cfg.hf_config, **kwargs)
    else:
        ds = load_dataset(cfg.source, **kwargs)
    return [dict(row) for row in ds]
