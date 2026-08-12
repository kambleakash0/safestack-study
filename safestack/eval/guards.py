"""Phase-5 eval leak guards (ADR-0017 decision 7). Pure -- no model load."""

from __future__ import annotations

from safestack.eval.config import EvalExperimentConfig
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec


def reject_api_backend(
    cfg: EvalExperimentConfig,
    *,
    backend_override: str | None = None,
    models_dir: str | object = DEFAULT_MODELS_DIR,
) -> None:
    """Raise if any card this eval loads resolves to ``backend == "api"`` (ADR-0017 dec.7).

    A harmful C9 generation must never be POSTed to a hosted provider -- the helpfulness judge
    sends the assistant text verbatim and a ready ``api`` sibling card exists, so a degraded-model
    completion could leave the box. Resolve every card the run loads: the policy and guardrails
    share the ``--backend`` override (as run_suite / build_guardrail apply it), the judges resolve
    from their own card (as judge_run does, no override). Fail BEFORE any generation or judging.
    Pure (no model load); called at the start of run_suite and judge_run.
    """
    checks: list[tuple[str, object, str | None]] = [("policy", cfg.model, backend_override)]
    if cfg.input_guardrail is not None:
        checks.append(("input_guardrail", cfg.input_guardrail, backend_override))
    if cfg.output_guardrail is not None:
        checks.append(("output_guardrail", cfg.output_guardrail, backend_override))
    if cfg.safety_judge is not None:
        checks.append(("safety_judge", cfg.safety_judge, None))
    if cfg.helpfulness_judge is not None:
        checks.append(("helpfulness_judge", cfg.helpfulness_judge, None))
    offenders = []
    for role, ref, override in checks:
        spec = resolve_model_spec(ref, backend_override=override, models_dir=models_dir)
        if spec.backend == "api":
            offenders.append(f"{role}={spec.model_id}")
    if offenders:
        raise ValueError(
            "ADR-0017 dec.7: harmful eval must be self-hosted -- these cards resolve to backend "
            f"'api': {sorted(offenders)}. Use hf_local (or mock in CI); nothing leaves the box."
        )
