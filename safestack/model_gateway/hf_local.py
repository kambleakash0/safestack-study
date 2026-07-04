"""Real HF-local backend: a tiny model on the laptop now, the same code for the 7B on cloud.

torch/transformers are imported lazily so importing this module (and the mock/api paths)
never requires the [hf] extra.
"""

from __future__ import annotations

import gc
import time

from safestack.config import ModelSpec
from safestack.determinism import set_seeds
from safestack.hashing import content_hash, model_fingerprint
from safestack.model_gateway.base import GenerationRequest, GenerationResult, ModelGateway


class HFLocalGateway(ModelGateway):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        if spec.adapter is not None:
            raise NotImplementedError("LoRA adapters land in Phase 3 (ADR-0003); not in Phase 0.")
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.spec.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"model '{self.spec.model_id}' requests device=cuda but no CUDA device is "
                "available. Run 7B/CUDA cards on a GPU box (ADR-0003); on this laptop use "
                "backend=api or a tiny hf_local card."
            )
        self._device = device
        dtype = getattr(torch, self.spec.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown dtype '{self.spec.dtype}' for model '{self.spec.model_id}'")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.checkpoint, revision=self.spec.revision
        )
        quant_config = self._quantization_config(dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.checkpoint,
            revision=self.spec.revision,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="auto" if quant_config is not None else None,
        )
        if quant_config is None:  # bitsandbytes places the model itself via device_map
            self._model = self._model.to(self._device)
        self._model.eval()

    def _render(self, messages) -> str:
        if self.spec.chat_template != "none":
            return self._tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import torch

        self._ensure_loaded()
        set_seeds(request.params.seed)
        prompt = self._render(request.messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        p = request.params
        start = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=p.max_new_tokens,
                do_sample=p.do_sample,
                temperature=p.temperature if p.do_sample else None,
                top_p=p.top_p if p.do_sample else None,
                top_k=p.top_k if p.do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        gen_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            model_id=self.spec.model_id,
            backend=self.spec.backend,
            content_hash=content_hash(model_fingerprint(self.spec), request.messages, p),
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(gen_tokens.shape[0]),
            finish_reason="stop",
            generation_ms=elapsed,
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def _quantization_config(self, dtype):
        """4-bit / 8-bit bitsandbytes config for CUDA eval on a smaller GPU (ADR-0007 decision 6).

        Returns None (full-precision) when no quantization is requested; the A100 baseline runs
        bf16. bitsandbytes is CUDA-only, so this path is only reached behind the cuda host-guard.
        """
        if not self.spec.quantization:
            return None
        from transformers import BitsAndBytesConfig

        q = self.spec.quantization.lower()
        if q in ("4bit", "nf4", "bnb-4bit"):
            return BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        if q in ("8bit", "bnb-8bit"):
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(
            f"unknown quantization '{self.spec.quantization}' for model '{self.spec.model_id}'"
        )
