"""Guardrail engine (ADR-0009): screens prompts (input) and/or responses (output) around the frozen
policy model as separate cached passes, never VRAM-co-resident with it (ADR-0003).

C1 uses the :class:`NullGuardrail` pass-through; Phase-2 C2/C3/C4 add real screening.
``build_guardrail`` selects the guardrail for the OUTPUT stage from ``cfg.guardrail_config`` and
``cfg.output_guardrail``, and enforces that the guardrail model is not the safety judge (ADR-0004
rule 4). The input pre-pass (C2) and its composition (C4) are declared in the placement literal but
not built yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from safestack.guardrails.base import (
    SAFE_REFUSAL,
    Guardrail,
    GuardrailDecision,
    NullGuardrail,
)
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec

if TYPE_CHECKING:
    # Lazy (typing-only) to break the guardrails <-> eval cycle: eval.config pulls in the eval
    # package, whose generate module imports build_guardrail from here (annotations are strings).
    from safestack.eval.config import EvalExperimentConfig


def _assert_separation(cfg: EvalExperimentConfig, spec, judge_spec) -> None:
    """ADR-0004 rule 4: the guardrail must not be the same classifier as the safety judge, or its
    block decision and the ASR score become circular. ``judge_spec`` is the resolved safety-judge
    spec (``None`` if the run has no safety judge)."""
    if judge_spec is None:
        return
    same = spec.model_id == judge_spec.model_id or (
        spec.checkpoint is not None and spec.checkpoint == judge_spec.checkpoint
    )
    if same:
        raise ValueError(
            f"ADR-0004 rule 4: the guardrail model '{spec.model_id}' must differ from the safety "
            f"judge '{judge_spec.model_id}' (same classifier as guardrail and judge is circular)."
        )


def build_guardrail(
    cfg: EvalExperimentConfig,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    backend_override: str | None = None,
) -> Guardrail:
    """Construct the output-stage guardrail for ``cfg``. Heavy models load lazily on first check, so
    this is safe to call before generation -- the model only materialises in the post-generation
    pass, after the policy model is freed (ADR-0003). ``backend == "mock"`` -> the deterministic
    eval-layer mock; otherwise the real Granite Guardian output guardrail."""
    placement = cfg.guardrail_config
    if placement == "none":
        return NullGuardrail()
    if placement in ("input", "input_output"):
        raise NotImplementedError(
            f"guardrail placement {placement!r} is not built yet (Phase-2 C2/C4 slices); "
            "this slice implements 'none' and 'output'"
        )
    # placement == "output"
    if cfg.output_guardrail is None:
        raise ValueError("guardrail_config='output' requires cfg.output_guardrail to name a model")
    spec = resolve_model_spec(
        cfg.output_guardrail, backend_override=backend_override, models_dir=models_dir
    )
    judge_spec = (
        resolve_model_spec(
            cfg.safety_judge, backend_override=backend_override, models_dir=models_dir
        )
        if cfg.safety_judge is not None
        else None
    )
    _assert_separation(cfg, spec, judge_spec)
    if spec.backend == "mock":
        from safestack.guardrails.mock import MockOutputGuardrail

        return MockOutputGuardrail()
    from safestack.guardrails.granite import GraniteOutputGuardrail

    return GraniteOutputGuardrail(spec)


__all__ = [
    "SAFE_REFUSAL",
    "Guardrail",
    "GuardrailDecision",
    "NullGuardrail",
    "build_guardrail",
]
