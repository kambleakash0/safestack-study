"""Validate a prepared suite against its manifest; report cross-suite prompt overlap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safestack.config import DatasetManifest
from safestack.datasets.prepare import normalize_prompt
from safestack.registry import load_manifest


def _prepared_path(manifest: DatasetManifest, data_dir: Path) -> Path:
    return data_dir / "prepared" / manifest.split / f"{manifest.name}.jsonl"


def validate_manifest(
    manifest_path: str | Path, *, data_dir: str | Path = "data"
) -> DatasetManifest:
    manifest = load_manifest(manifest_path)
    path = _prepared_path(manifest, Path(data_dir))
    if not path.exists():
        raise FileNotFoundError(
            f"prepared data missing for '{manifest.name}': {path} (run `safestack data prepare`)"
        )
    content = path.read_text(encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != manifest.hash:
        raise ValueError(
            f"hash mismatch for '{manifest.name}': manifest {manifest.hash}, prepared {digest}"
        )
    return manifest


def prompt_overlap(data_dir: str | Path = "data") -> dict[str, int]:
    """Normalized-prompt overlap counts between every pair of prepared suites (leakage signal)."""
    data_dir = Path(data_dir)
    suites: dict[str, set[str]] = {}
    for path in sorted((data_dir / "prepared").rglob("*.jsonl")):
        norms = set()
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                norms.add(normalize_prompt(json.loads(ln)["prompt"]))
        suites[path.stem] = norms

    names = sorted(suites)
    overlaps: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = len(suites[a] & suites[b])
            if n:
                overlaps[f"{a}|{b}"] = n
    return overlaps
