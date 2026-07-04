"""Step 9: hf-marked parity — the real weight-load + real content_hash path (tiny-GPT2) produces
the SAME artifact structure as the mock smoke. Numbers are gibberish by design; only STRUCTURE is
asserted. Excluded from the default CI (`-m 'not hf'`); run with `uv run --extra hf pytest -m hf`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from safestack.config import DecodeParams, ModelSpec
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.metrics import suite_metrics

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg(model) -> EvalExperimentConfig:
    return EvalExperimentConfig(
        experiment_id="parity",
        model=model,
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        decode=DecodeParams(max_new_tokens=8, seed=0),
        bootstrap_n=200,
    )


def _build(model, run_root: Path, cache: Path):
    cfg = _cfg(model)
    run_dir = run_suite(
        cfg, runs_dir=run_root, data_dir=FIX, cache_dir=cache, models_dir="configs/models", limit=2
    )
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache, models_dir="configs/models")
    art = suite_metrics(
        run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache, models_dir="configs/models"
    )
    return run_dir, art


@pytest.mark.hf
def test_tiny_gpt2_matches_mock_structure(tmp_path: Path) -> None:
    _, mock_art = _build(
        ModelSpec(model_id="mock", backend="mock"), tmp_path / "mr", tmp_path / "mc"
    )
    tiny_run, tiny_art = _build("tiny_gpt2", tmp_path / "tr", tmp_path / "tc")

    # Same artifact schema and the same metric set — only the numbers differ.
    assert set(mock_art.model_dump().keys()) == set(tiny_art.model_dump().keys())
    assert [m.name for m in mock_art.metrics] == [m.name for m in tiny_art.metrics]

    # The real weight-load path produced the full artifact set.
    assert (tiny_run / "run.json").exists()
    assert (tiny_run / "traces.jsonl").exists()
    assert list((tmp_path / "tc" / "generations").rglob("*.json"))
    assert list((tmp_path / "tc" / "judgments").rglob("*.json"))
    assert tiny_art.policy_model_id == "tiny_gpt2"
