"""Step 4: judge layer — routing, caching by judge key, re-keying, rule-4 separation."""

from __future__ import annotations

from pathlib import Path

import pytest

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import build_judge, judge_run
from safestack.eval.judges.mock import MockSafetyJudge
from safestack.eval.judges.refusal import RefusalDetector, is_refusal

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg(**over: object) -> EvalExperimentConfig:
    base = dict(
        experiment_id="judge_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
    )
    base.update(over)
    return EvalExperimentConfig(**base)


def _n_judgments(cache: Path) -> int:
    return len(list((cache / "judgments").rglob("*.json")))


def test_routing_and_caching(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    counts = judge_run(run_dir, cfg=_cfg(), data_dir=FIX, cache_dir=cache)
    assert counts["safety"]["scored"] == 6  # harmful suite
    assert counts["refusal"]["scored"] == 5  # over-refusal suite
    assert counts["helpfulness"]["scored"] == 5  # helpfulness suite
    assert _n_judgments(cache) == 16


def test_rejudge_is_all_hits(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=_cfg(), data_dir=FIX, cache_dir=cache)
    again = judge_run(run_dir, cfg=_cfg(), data_dir=FIX, cache_dir=cache)
    assert sum(c["scored"] for c in again.values()) == 0
    assert sum(c["hits"] for c in again.values()) == 16


def test_bumping_prompt_version_mints_new_keys(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=_cfg(judge_prompt_version="v1"), data_dir=FIX, cache_dir=cache)
    assert _n_judgments(cache) == 16
    # A new prompt version re-scores everything into NEW files without overwriting the old labels.
    counts = judge_run(run_dir, cfg=_cfg(judge_prompt_version="v2"), data_dir=FIX, cache_dir=cache)
    assert sum(c["scored"] for c in counts.values()) == 16
    assert _n_judgments(cache) == 32


def test_cfg_loaded_from_run_json(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    counts = judge_run(run_dir, data_dir=FIX, cache_dir=cache)  # cfg=None -> read from run.json
    assert counts["safety"]["scored"] == 6


def test_kinds_filter(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    counts = judge_run(run_dir, cfg=_cfg(), data_dir=FIX, cache_dir=cache, kinds=["safety"])
    assert counts["safety"]["scored"] == 6
    assert "refusal" not in counts and "helpfulness" not in counts
    assert _n_judgments(cache) == 6


def test_guard_card_rejected_as_non_safety_judge() -> None:
    guard = ModelSpec(
        model_id="llama_guard_3_1b", backend="mock", checkpoint="meta-llama/Llama-Guard-3-1B"
    )
    with pytest.raises(ValueError, match="rule 4"):
        build_judge("helpfulness", guard)
    # But it is fine as the safety judge.
    assert build_judge("safety", guard).role == "safety"


def test_refusal_heuristic_and_mock_safety_determinism() -> None:
    assert is_refusal("I'm sorry, I cannot help with that.")
    assert not is_refusal("Sure! Here is how you do it.")
    j = MockSafetyJudge(fingerprint={"model_id": "m"}, judge_prompt_version="v1")
    assert j.score("u", "some response").label == j.score("u", "some response").label

    d = RefusalDetector()
    assert d.score("u", "I cannot assist with that").label == "refusal"
    assert d.score("u", "Here you go").label == "answer"

def test_judge_batch_size_invariant(tmp_path: Path) -> None:
    # Batching judge scorings must not change judgments or counts -- batch_size only groups score()
    # calls into score_batch() forward passes. Mock judges are deterministic, so the cached
    # judgments AND the scored/hits counts must be identical for batch_size 1 vs 32.
    import json

    def _judgments(cache: Path) -> dict:
        out = {}
        for p in sorted((cache / "judgments").rglob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d["judge_key"]] = (d["judge_role"], d["label"], d["score"])
        return out

    c1, c32 = tmp_path / "c1", tmp_path / "c32"
    r1 = run_suite(_cfg(), runs_dir=tmp_path / "r1", data_dir=FIX, cache_dir=c1)
    r32 = run_suite(_cfg(), runs_dir=tmp_path / "r32", data_dir=FIX, cache_dir=c32)
    n1 = judge_run(r1, cfg=_cfg(), data_dir=FIX, cache_dir=c1, batch_size=1)
    n32 = judge_run(r32, cfg=_cfg(), data_dir=FIX, cache_dir=c32, batch_size=32)
    assert n1 == n32  # identical scored/hits counts regardless of judge batch grouping
    assert _judgments(c1) == _judgments(c32)  # identical judgments


def test_judge_score_batch_default_fallback() -> None:
    # The base Judge.score_batch (inherited by mock / refusal) is an order-preserving loop.
    j = MockSafetyJudge(fingerprint={"id": "m"}, judge_prompt_version="v2")
    items = [("u1", "a1"), ("u2", "a2"), ("u3", "a3")]
    assert [b.label for b in j.score_batch(items)] == [j.score(u, a).label for u, a in items]
    assert j.score_batch([]) == []
