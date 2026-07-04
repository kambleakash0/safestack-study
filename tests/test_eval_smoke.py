"""Step 7: end-to-end CLI smoke on the base install (mock + fixtures, -m 'not hf').

Exercises the whole generate -> judge -> report -> compare chain through the real CLI, proving the
harness produces the full artifact set with only the torch/numpy-free base dependencies.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from safestack.cli import app

runner = CliRunner()
FIX = "tests/fixtures/eval_suites"
CFG = "configs/experiments/eval_smoke.yaml"


def test_full_cli_chain(tmp_path: Path) -> None:
    runs, cache, reports = tmp_path / "runs", tmp_path / "cache", tmp_path / "reports"

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
    assert (run_dir / "run.json").exists()
    assert (run_dir / "traces.jsonl").exists()
    assert list((cache / "generations").rglob("*.json"))

    rj = runner.invoke(
        app, ["eval", "judge", "--run", str(run_dir), "--data-dir", FIX, "--cache-dir", str(cache)]
    )
    assert rj.exit_code == 0, rj.output
    assert list((cache / "judgments").rglob("*.json"))

    rr = runner.invoke(
        app,
        [
            "eval",
            "report",
            "--run",
            str(run_dir),
            "--data-dir",
            FIX,
            "--cache-dir",
            str(cache),
            "--reports-dir",
            str(reports),
        ],
    )
    assert rr.exit_code == 0, rr.output
    metrics_files = sorted((reports / "metrics").glob("*.json"))
    assert len(metrics_files) == 3  # one per suite

    metrics_args: list[str] = []
    for f in metrics_files:
        metrics_args += ["--metrics", str(f)]
    rc = runner.invoke(app, ["eval", "compare", "--gate", *metrics_args])
    assert rc.exit_code == 0, rc.output
    assert "Dynamic-range gate (ADR-0002)" in rc.output

    # Leakage: no raw harmful prompt text in any produced report.
    for f in metrics_files:
        assert "PLACEHOLDER harmful request" not in f.read_text(encoding="utf-8")
