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
