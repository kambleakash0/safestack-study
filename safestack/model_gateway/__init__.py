"""Model gateway factory + extension seam (ADR-0003).

Adding a backend later (vllm, cloud) is a new class + one branch here; callers and configs
are unchanged.
"""

from __future__ import annotations

from safestack.config import ModelSpec
from safestack.model_gateway.base import (
    GenerationRequest,
    GenerationResult,
    ModelGateway,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "ModelGateway",
    "build_gateway",
]


def build_gateway(spec: ModelSpec) -> ModelGateway:
    backend = spec.backend
    if backend == "mock":
        from safestack.model_gateway.mock import MockGateway

        return MockGateway(spec)
    if backend == "hf_local":
        from safestack.model_gateway.hf_local import HFLocalGateway

        return HFLocalGateway(spec)
    if backend == "api":
        from safestack.model_gateway.api import ApiGateway

        return ApiGateway(spec)
    if backend in ("vllm", "cloud"):
        raise NotImplementedError(
            f"backend '{backend}' is a planned seam, not implemented in Phase 0 (ADR-0003)."
        )
    raise ValueError(f"unknown backend: {backend!r}")
