"""Phase 2 / C2-C3 CLI smoke: the committed c2_smoke.yaml / c3_smoke.yaml drive an input- / output-
guardrail run through the real CLI (mock policy + mock guardrail), proving block decisions reach
traces.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from safestack.cli import app

runner = CliRunner()
FIX = "tests/fixtures/eval_suites"
CFG = "configs/experiments/c3_smoke.yaml"

CFG_C2 = "configs/experiments/c2_smoke.yaml"

CFG_C4 = "configs/experiments/c4_smoke.yaml"


def test_cli_c3_output_guardrail_run(tmp_path: Path) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "-c",
            CFG,
            "--data-dir",
            FIX,
            "--runs-dir",
            str(runs),
            "--cache-dir",
            str(cache),
        ],
    )
    assert r.exit_code == 0, r.output

    run_dir = next(runs.iterdir())
    traces = [
        json.loads(ln)
        for ln in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert traces and all(t["guardrail_config"] == "output" for t in traces)
    assert any(
        t["blocked_at"] == "output" for t in traces
    )  # the mock guardrail blocked >= 1 response


def test_cli_c2_input_guardrail_run(tmp_path: Path) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "-c",
            CFG_C2,
            "--data-dir",
            FIX,
            "--runs-dir",
            str(runs),
            "--cache-dir",
            str(cache),
        ],
    )
    assert r.exit_code == 0, r.output

    run_dir = next(runs.iterdir())
    traces = [
        json.loads(ln)
        for ln in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert traces and all(t["guardrail_config"] == "input" for t in traces)
    assert any(t["blocked_at"] == "input" for t in traces)  # the mock input guardrail blocked >= 1

def test_cli_c4_input_output_guardrail_run(tmp_path: Path) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "-c",
            CFG_C4,
            "--data-dir",
            FIX,
            "--runs-dir",
            str(runs),
            "--cache-dir",
            str(cache),
        ],
    )
    assert r.exit_code == 0, r.output

    run_dir = next(runs.iterdir())
    traces = [
        json.loads(ln)
        for ln in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert traces and all(t["guardrail_config"] == "input_output" for t in traces)
    # All three C4 outcome classes must appear (the mock input/output predicates are independent):
    # an input pre-pass block, an output-screen block among the input-passers, and a pass-both. The
    # messages make a future fixture edit that collapses the distribution fail self-explanatorily.
    assert any(t["blocked_at"] == "input" for t in traces), "no input-block in fixture distribution"
    assert any(t["blocked_at"] == "output" for t in traces), "no output-block among input-passers"
    assert any(t["blocked_at"] is None for t in traces), "no prompt passed both stages"
    # C4 short-circuit: an input-blocked prompt is NEVER output-screened, so output_guardrail_ms is
    # set for exactly the input-passers (a refused prompt skips the output stage).
    for t in traces:
        if t["blocked_at"] == "input":
            assert t["output_guardrail_ms"] is None
        else:
            assert t["output_guardrail_ms"] is not None
