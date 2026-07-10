"""Guardrail engine (ADR-0009): screens prompts (input) and/or responses (output) around the frozen
policy model as separate cached passes, never VRAM-co-resident with it (ADR-0003).

C1 uses the :class:`NullGuardrail` pass-through; Phase-2 C2/C3/C4 add real screening.
``build_guardrail`` selects the guardrail from ``cfg.guardrail_config`` (``none`` / ``input`` /
``output`` / ``input_output``) and its ``input_guardrail`` / ``output_guardrail`` model(s), and
enforces that the guardrail model is not the safety judge (ADR-0004 rule 4). ``input_output`` (C4)
requires ONE model for both stages so a single composite guardrail screens the prompt and the
response over one loaded model (ADR-0009 decision 2).
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


def _resolve_guardrail_spec(
    cfg: EvalExperimentConfig,
    placement: str,
    *,
    backend_override: str | None,
    models_dir: str | Path,
):
    """Resolve the guardrail ModelSpec for a non-none ``placement`` from ``cfg.input_guardrail`` /
    ``cfg.output_guardrail``. ``input_output`` (C4) requires BOTH fields set AND the SAME model_id,
    so one distinct model serves both stages (ADR-0009 decision 2) -- build_guardrail can then
    return a single composite guardrail, never two model copies resident at once (ADR-0003)."""

    def _resolve(name):
        return resolve_model_spec(name, backend_override=backend_override, models_dir=models_dir)

    if placement == "input":
        if cfg.input_guardrail is None:
            raise ValueError(
                "guardrail_config='input' requires cfg.input_guardrail to name a model"
            )
        return _resolve(cfg.input_guardrail)
    if placement == "output":
        if cfg.output_guardrail is None:
            raise ValueError(
                "guardrail_config='output' requires cfg.output_guardrail to name a model"
            )
        return _resolve(cfg.output_guardrail)
    # placement == "input_output" (C4): one model, both stages.
    if cfg.input_guardrail is None or cfg.output_guardrail is None:
        raise ValueError(
            "guardrail_config='input_output' requires both cfg.input_guardrail and "
            "cfg.output_guardrail to name a model"
        )
    in_spec = _resolve(cfg.input_guardrail)
    out_spec = _resolve(cfg.output_guardrail)
    if in_spec.model_id != out_spec.model_id:
        raise ValueError(
            "guardrail_config='input_output' requires the SAME model for both stages "
            f"(ADR-0009 decision 2): got input '{in_spec.model_id}' vs output '{out_spec.model_id}'"
        )
    return in_spec


def build_guardrail(
    cfg: EvalExperimentConfig,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    backend_override: str | None = None,
) -> Guardrail:
    """Construct the guardrail for ``cfg.guardrail_config``. Heavy models load lazily on the first
    SCREENED check, so this is safe to call before generation -- the model only materialises in the
    guardrail pass, after the policy model is freed (ADR-0003). ``backend == "mock"`` -> the
    deterministic eval-layer mock; otherwise the real Granite Guardian. One composite guardrail
    serves whichever stage(s) the placement selects, so C4 loads the model once."""
    placement = cfg.guardrail_config
    if placement == "none":
        return NullGuardrail()
    spec = _resolve_guardrail_spec(
        cfg, placement, backend_override=backend_override, models_dir=models_dir
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
        from safestack.guardrails.mock import MockGuardrail

        return MockGuardrail(placement=placement)
    from safestack.guardrails.granite import GraniteGuardrail

    return GraniteGuardrail(spec, placement=placement)


__all__ = [
    "SAFE_REFUSAL",
    "Guardrail",
    "GuardrailDecision",
    "NullGuardrail",
    "build_guardrail",
]
