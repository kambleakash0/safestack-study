"""Judge registry: safety / refusal / helpfulness, strictly separated (ADR-0004 rule 4).

PASS B. Loads ONE judge at a time, scores the run's generation cache, and writes labels to the
judgment cache keyed by ``judge_content_hash`` (a new judge prompt or revision mints a new key).
The policy model was already freed in PASS A, so a judge is never VRAM-co-resident with it
(ADR-0003); each role's judge is closed before the next role's judge loads.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from safestack.eval.cache import ContentHashStore, JudgmentCacheEntry
from safestack.eval.config import EvalExperimentConfig
from safestack.hashing import judge_content_hash, model_fingerprint
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec

log = logging.getLogger("safestack")

ROLES = ("safety", "refusal", "helpfulness")
# Which judge scores which eval split.
SPLIT_TO_ROLE = {
    "eval_harmful": "safety",
    "eval_benign_overrefusal": "refusal",
    "eval_benign_helpfulness": "helpfulness",
}


@dataclass(frozen=True)
class JudgeLabel:
    """One judge's verdict on one (prompt, response). ``parse_ok=False`` means the raw output
    could not be parsed and the item is flagged for human audit, never coerced to a label."""

    label: str
    categories: tuple[str, ...] = ()
    score: float | None = None
    parse_ok: bool = True
    raw_first_line: str = ""


class Judge(ABC):
    role: str = ""

    def __init__(self, fingerprint: dict, judge_prompt_version: str) -> None:
        self._fingerprint = fingerprint
        self.judge_prompt_version = judge_prompt_version

    def fingerprint(self) -> dict:
        return self._fingerprint

    @abstractmethod
    def score(self, user: str, assistant: str) -> JudgeLabel: ...

    def close(self) -> None:
        """Release any held resources (heavy judges override); mirrors ModelGateway.close."""
        return None


def _assert_separation(role: str, spec) -> None:
    """ADR-0004 rule 4: the Llama-Guard safety card must never be the refusal/helpfulness judge."""
    name = f"{spec.model_id} {spec.checkpoint or ''}".lower()
    if role != "safety" and "guard" in name:
        raise ValueError(
            f"ADR-0004 rule 4: Llama-Guard card '{spec.model_id}' is the safety judge only, "
            f"not the {role} judge."
        )


def build_judge(
    role: str,
    spec=None,
    *,
    judge_prompt_version: str = "v1",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    backend_override: str | None = None,
) -> Judge:
    """Select a judge for ``role``. ``backend == "mock"`` -> a rule-based eval-layer mock judge
    (no VRAM, no download); otherwise the real adapter. Refusal is always the heuristic detector."""
    if role not in ROLES:
        raise ValueError(f"unknown judge role {role!r}")
    if role == "refusal":
        from safestack.eval.judges.refusal import RefusalDetector

        return RefusalDetector(judge_prompt_version=judge_prompt_version)

    if spec is None:
        raise ValueError(f"judge role {role!r} needs a model spec (set cfg.{role}_judge)")
    resolved = resolve_model_spec(spec, backend_override=backend_override, models_dir=models_dir)
    _assert_separation(role, resolved)
    fingerprint = model_fingerprint(resolved)

    if resolved.backend == "mock":
        from safestack.eval.judges.mock import MockHelpfulness, MockSafetyJudge

        cls = MockSafetyJudge if role == "safety" else MockHelpfulness
        return cls(fingerprint=fingerprint, judge_prompt_version=judge_prompt_version)
    if role == "safety":
        from safestack.eval.judges.llama_guard import LlamaGuardJudge

        return LlamaGuardJudge(resolved, judge_prompt_version=judge_prompt_version)
    from safestack.eval.judges.helpfulness import HelpfulnessJudge

    return HelpfulnessJudge(resolved, judge_prompt_version=judge_prompt_version)


def _iter_traces(run_dir: Path):
    with (run_dir / "traces.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_cfg_from_run(run_dir: Path) -> EvalExperimentConfig:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return EvalExperimentConfig.model_validate(run["config"])


def _judge_spec_for_role(cfg: EvalExperimentConfig, role: str):
    return {"safety": cfg.safety_judge, "helpfulness": cfg.helpfulness_judge}.get(role)


def judge_run(
    run_dir: str | Path,
    *,
    cfg: EvalExperimentConfig | None = None,
    data_dir: str | Path = "data",
    cache_dir: str | Path | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    kinds: list[str] | None = None,
    backend_override: str | None = None,
) -> dict[str, dict[str, int]]:
    """Score this run's generations. Returns ``{role: {scored, hits}}``. One judge resident at a
    time; each judge is closed before the next role's judge loads."""
    run_dir = Path(run_dir)
    cfg = cfg or _load_cfg_from_run(run_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(data_dir) / "cache"
    gen_store = ContentHashStore(cache_dir, "generations")
    judg_store = ContentHashStore(cache_dir, "judgments")

    hashes_by_role: dict[str, list[str]] = {r: [] for r in ROLES}
    for trace in _iter_traces(run_dir):
        role = SPLIT_TO_ROLE.get(trace.get("split"))
        if role:
            hashes_by_role[role].append(trace["content_hash"])

    wanted = ROLES if (not kinds or "all" in kinds) else [k for k in kinds if k in ROLES]
    counts: dict[str, dict[str, int]] = {}
    for role in wanted:
        hashes = list(dict.fromkeys(hashes_by_role[role]))  # dedup, keep order
        if not hashes:
            counts[role] = {"scored": 0, "hits": 0}
            continue
        judge = build_judge(
            role,
            _judge_spec_for_role(cfg, role),
            judge_prompt_version=cfg.judge_prompt_version,
            models_dir=models_dir,
            backend_override=backend_override,
        )
        fingerprint = judge.fingerprint()
        scored = hits = 0
        try:
            for ch in hashes:
                gen = gen_store.get(ch)
                if gen is None:
                    continue
                jkey = judge_content_hash(ch, fingerprint, cfg.judge_prompt_version, role)
                if judg_store.get(jkey) is not None:
                    hits += 1
                    continue
                user = gen["messages"][0]["content"] if gen.get("messages") else ""
                label = judge.score(user, gen["text"])
                judg_store.put(
                    jkey,
                    JudgmentCacheEntry(
                        judge_key=jkey,
                        gen_content_hash=ch,
                        judge_role=role,
                        judge_fingerprint=fingerprint,
                        judge_prompt_version=cfg.judge_prompt_version,
                        label=label.label,
                        categories=list(label.categories),
                        score=label.score,
                        parse_ok=label.parse_ok,
                        raw_first_line=label.raw_first_line,
                    ),
                )
                scored += 1
        finally:
            judge.close()  # free this judge before the next role's judge loads (ADR-0003)
        counts[role] = {"scored": scored, "hits": hits}
        log.info("judge %s | scored=%d hits=%d", role, scored, hits)
    return counts


__all__ = ["ROLES", "SPLIT_TO_ROLE", "Judge", "JudgeLabel", "build_judge", "judge_run"]


def role_fingerprint(
    cfg: EvalExperimentConfig,
    role: str,
    *,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    backend_override: str | None = None,
) -> dict:
    """The judge fingerprint for a role WITHOUT constructing a heavy judge (no torch import).

    Mirrors what ``judge_run`` used as the judge key, so the metrics pass can look up the same
    judgment cache entries. Assumes judges were run without a backend override (the standard path).
    """
    if role == "refusal":
        from safestack.eval.judges.refusal import RefusalDetector

        return RefusalDetector(judge_prompt_version=cfg.judge_prompt_version).fingerprint()
    spec = _judge_spec_for_role(cfg, role)
    if spec is None:
        raise ValueError(f"judge role {role!r} needs cfg.{role}_judge to compute its fingerprint")
    return model_fingerprint(
        resolve_model_spec(spec, backend_override=backend_override, models_dir=models_dir)
    )
