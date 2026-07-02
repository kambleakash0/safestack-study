"""Deterministic, dependency-free backend for the smoke test and all unit tests."""

from __future__ import annotations

import time

from safestack.hashing import content_hash, model_fingerprint
from safestack.model_gateway.base import GenerationRequest, GenerationResult, ModelGateway


class MockGateway(ModelGateway):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        ch = content_hash(model_fingerprint(self.spec), request.messages, request.params)
        rendered = " ".join(f"{m.role}: {m.content}" for m in request.messages)
        text = f"[mock:{self.spec.model_id}] {ch.split(':')[1][:8]} :: {rendered[:120]}"
        return GenerationResult(
            text=text,
            model_id=self.spec.model_id,
            backend=self.spec.backend,
            content_hash=ch,
            finish_reason="stop",
            generation_ms=(time.perf_counter() - start) * 1000.0,
        )
