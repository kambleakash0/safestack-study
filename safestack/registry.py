"""File-backed registries: filesystem + pydantic validation, no database."""

from __future__ import annotations

from pathlib import Path

import yaml

from safestack.config import DatasetManifest, ExperimentConfig, ModelSpec

SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_MODELS_DIR = "configs/models"


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _check_version(obj, path: str | Path) -> None:
    if obj.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {obj.schema_version} "
            f"(expected {SUPPORTED_SCHEMA_VERSION})"
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    obj = ExperimentConfig.model_validate(_load_yaml(path))
    _check_version(obj, path)
    return obj


def load_manifest(path: str | Path) -> DatasetManifest:
    obj = DatasetManifest.model_validate(_load_yaml(path))
    _check_version(obj, path)
    return obj


def load_model(
    model_id_or_path: str | Path, models_dir: str | Path = DEFAULT_MODELS_DIR
) -> ModelSpec:
    p = Path(model_id_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        obj = ModelSpec.model_validate(_load_yaml(p))
        _check_version(obj, p)
        return obj

    index: dict[str, Path] = {}
    for f in sorted([*Path(models_dir).glob("*.yaml"), *Path(models_dir).glob("*.yml")]):
        data = _load_yaml(f)
        if data and data.get("model_id"):
            index[data["model_id"]] = f

    key = str(model_id_or_path)
    if key not in index:
        available = ", ".join(sorted(index)) or "(none)"
        raise KeyError(f"unknown model_id {key!r}; available: {available}")
    obj = ModelSpec.model_validate(_load_yaml(index[key]))
    _check_version(obj, index[key])
    return obj


def resolve_model_ref(
    cfg: ExperimentConfig,
    backend_override: str | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> ModelSpec:
    return resolve_model_spec(cfg.model, backend_override=backend_override, models_dir=models_dir)


def resolve_model_spec(
    model_ref: str | ModelSpec,
    backend_override: str | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> ModelSpec:
    """Resolve a model reference (a registry model_id or an inline ModelSpec) to a ModelSpec,
    applying an optional backend override.

    Decoupled from any single config so an eval condition can resolve its policy model and its
    judge models independently (ADR-0007).
    """
    spec = (
        model_ref
        if isinstance(model_ref, ModelSpec)
        else load_model(model_ref, models_dir=models_dir)
    )
    if backend_override and backend_override != spec.backend:
        spec = ModelSpec.model_validate({**spec.model_dump(), "backend": backend_override})
    return spec
