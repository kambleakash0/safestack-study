"""Backend-agnostic gateway interface + request/result types (zero torch imports)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from safestack.config import DecodeParams, Message, ModelSpec


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[Message, ...]
    params: DecodeParams

    @classmethod
    def from_prompt(cls, prompt: str, params: DecodeParams) -> GenerationRequest:
        return cls(messages=(Message(role="user", content=prompt),), params=params)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_id: str
    backend: str
    content_hash: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    generation_ms: float | None = None


class ModelGateway(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def close(self) -> None:
        """Release any held resources. No-op by default; heavy backends override.

        This is the ADR-0003 'never two large models resident' seam: the runner calls it
        in a finally block so the policy model is freed before a later judge pass loads.
        """
        return None

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Generate for a list of requests, preserving order 1:1. Default: a sequential fallback
        (one ``generate`` per request) so light backends (mock, api) need no change; heavy backends
        (hf_local) override with a true batched forward pass to fill idle GPU capacity."""
        return [self.generate(r) for r in requests]
