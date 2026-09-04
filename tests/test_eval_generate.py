"""Step 3: batch generation pass — caching, resumability, determinism, C1 trace invariants."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from safestack.config import ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]
N_RECORDS = 16  # 6 harmful + 5 over-refusal + 5 helpfulness


def _cfg(suites: list[str] | None = None) -> EvalExperimentConfig:
    return EvalExperimentConfig(
        experiment_id="gen_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=suites or SUITES,
    )


def _traces(run_dir: Path) -> list[dict]:
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def test_one_cache_file_and_trace_per_record(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)

    assert len(list((cache / "generations").rglob("*.json"))) == N_RECORDS
    traces = _traces(run_dir)
    assert len(traces) == N_RECORDS

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["n_cache_misses"] == N_RECORDS
    assert run["n_cache_hits"] == 0
    assert run["n_generations"] == N_RECORDS
    assert run["experiment_id"] == "gen_smoke"


def test_c1_trace_invariants(tmp_path: Path) -> None:
    run_dir = run_suite(_cfg(), runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=tmp_path / "c")
    for t in _traces(run_dir):
        assert t["condition_id"] == "C1"
        assert t["guardrail_config"] == "none"
        assert t["blocked_at"] is None  # C1: nothing is ever blocked
        assert t["final_response"] == t["output"]  # identity transform in C1
        assert t["eval_id"] and t["suite"] and t["split"]


def test_resumability_and_determinism(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    r1 = run_suite(_cfg(), runs_dir=tmp_path / "r1", data_dir=FIX, cache_dir=cache)
    r2 = run_suite(_cfg(), runs_dir=tmp_path / "r2", data_dir=FIX, cache_dir=cache)

    run2 = json.loads((r2 / "run.json").read_text(encoding="utf-8"))
    assert run2["n_cache_hits"] == N_RECORDS  # second run is a full cache hit
    assert run2["n_cache_misses"] == 0

    def by_eval(run_dir: Path) -> dict[str, tuple[str, str]]:
        return {t["eval_id"]: (t["content_hash"], t["output"]) for t in _traces(run_dir)}

    assert by_eval(r1) == by_eval(r2)  # identical hashes and text across runs


def test_manifest_hash_gate_trips_on_tamper(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIX, data)
    prepared = data / "prepared" / "eval_harmful" / "harmful_fixture.jsonl"
    prepared.write_text(prepared.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_suite(
            _cfg(["harmful_fixture"]),
            runs_dir=tmp_path / "runs",
            data_dir=str(data),
            cache_dir=tmp_path / "cache",
        )

def test_batch_size_invariant_for_traces_and_cache(tmp_path: Path) -> None:
    # The batched miss-loop must produce byte-identical traces / hashes / counts regardless of
    # batch_size -- batching changes only how misses are grouped into forward passes, never the
    # per-prompt content_hash, the cache entries, or the record ordering.
    r1 = run_suite(
        _cfg(), runs_dir=tmp_path / "b1", data_dir=FIX, cache_dir=tmp_path / "c1", batch_size=1
    )
    r32 = run_suite(
        _cfg(), runs_dir=tmp_path / "b32", data_dir=FIX, cache_dir=tmp_path / "c32", batch_size=32
    )

    def by_eval(run_dir: Path) -> dict[str, tuple[str, str]]:
        return {t["eval_id"]: (t["content_hash"], t["output"]) for t in _traces(run_dir)}

    assert by_eval(r1) == by_eval(r32)  # identical hashes + text regardless of batch grouping
    for rd in (r1, r32):
        run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
        assert run["n_cache_misses"] == N_RECORDS and run["n_cache_hits"] == 0


def test_generate_batch_default_is_sequential_and_ordered() -> None:
    # The base ModelGateway.generate_batch fallback (inherited by mock / api) must be a 1:1,
    # order-preserving loop over generate() -- so light backends need no batch code.
    from safestack.config import DecodeParams, ModelSpec
    from safestack.model_gateway.base import GenerationRequest
    from safestack.model_gateway.mock import MockGateway

    gw = MockGateway(ModelSpec(model_id="m", backend="mock"))
    dp = DecodeParams(max_new_tokens=4, seed=0)
    reqs = [GenerationRequest.from_prompt(p, dp) for p in "abcd"]
    assert [r.text for r in gw.generate_batch(reqs)] == [gw.generate(r).text for r in reqs]
    assert gw.generate_batch([]) == []
