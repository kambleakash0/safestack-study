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
