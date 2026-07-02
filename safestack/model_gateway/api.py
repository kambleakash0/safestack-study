"""Hosted (OpenAI-compatible) inference backend, e.g. Mistral La Plateforme.

Inference-only convenience: it cannot fine-tune and its outputs are non-deterministic, so
it is excluded from the locked-test path (see ADR-0005). The key is read from the env var
named by `spec.api_key_env` (default SAFESTACK_API_KEY); nothing is committed.
"""

from __future__ import annotations

import time

from safestack.config import ModelSpec
from safestack.hashing import content_hash, model_fingerprint
from safestack.model_gateway.base import GenerationRequest, GenerationResult, ModelGateway

_DEFAULT_BASE_URL = "https://api.mistral.ai/v1"


class ApiGateway(ModelGateway):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        if not spec.checkpoint:
            raise ValueError("api backend requires `checkpoint` set to the provider model name")
        self._base_url = (spec.base_url or _DEFAULT_BASE_URL).rstrip("/")

    def _chat_completion(self, payload: dict) -> dict:
        """POST the request. Isolated so tests can stub it without httpx or a network call."""
        import os

        import httpx

        key = os.environ.get(self.spec.api_key_env)
        if not key:
            raise RuntimeError(
                f"API key not found in env var '{self.spec.api_key_env}'. "
                "Export it before using the api backend."
            )
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        p = request.params
        payload = {
            "model": self.spec.checkpoint,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": p.max_new_tokens,
            "temperature": p.temperature if p.do_sample else 0.0,
            "top_p": p.top_p,
        }
        start = time.perf_counter()
        data = self._chat_completion(payload)
        elapsed = (time.perf_counter() - start) * 1000.0
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"api backend: provider returned no choices (keys: {list(data)})")
        choice = choices[0]
        usage = data.get("usage") or {}
        text = (choice.get("message") or {}).get("content") or ""
        return GenerationResult(
            text=text,
            model_id=self.spec.model_id,
            backend=self.spec.backend,
            content_hash=content_hash(model_fingerprint(self.spec), request.messages, p),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            generation_ms=elapsed,
        )
