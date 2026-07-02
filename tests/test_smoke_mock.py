import json
from pathlib import Path

from typer.testing import CliRunner

from safestack.cli import app
from safestack.runner import run_experiment

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "configs" / "models"
SMOKE = REPO / "configs" / "experiments" / "smoke.yaml"


def test_run_experiment_end_to_end(tmp_path):
    result = run_experiment(SMOKE, runs_dir=tmp_path, models_dir=MODELS)
    assert result is not None and result.text  # exit criterion 1

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "run.json").exists()  # exit criterion 2

    traces = (run_dir / "traces.jsonl").read_text().strip().splitlines()
    assert len(traces) == 1
    trace = json.loads(traces[0])
    assert trace["output"]
    assert trace["content_hash"].startswith("sha256:")


def test_dry_run_writes_run_but_no_trace(tmp_path):
    result = run_experiment(SMOKE, runs_dir=tmp_path, dry_run=True, models_dir=MODELS)
    assert result is None
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "run.json").exists()
    assert not (run_dir / "traces.jsonl").exists()


def test_determinism_identical_content_hash(tmp_path):
    r1 = run_experiment(SMOKE, runs_dir=tmp_path / "a", models_dir=MODELS)
    r2 = run_experiment(SMOKE, runs_dir=tmp_path / "b", models_dir=MODELS)
    assert r1.content_hash == r2.content_hash
    assert r1.text == r2.text


def test_cli_run(tmp_path):
    res = CliRunner().invoke(
        app,
        ["run", "--config", str(SMOKE), "--runs-dir", str(tmp_path), "--models-dir", str(MODELS)],
    )
    assert res.exit_code == 0, res.output
