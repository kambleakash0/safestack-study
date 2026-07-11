import json
from datetime import date

import pytest
import yaml
from typer.testing import CliRunner

from safestack.cli import app
from safestack.datasets.prepare import eval_id, prepare, prepare_records
from safestack.datasets.schema import DatasetPrepConfig
from safestack.datasets.validate import prompt_overlap, validate_manifest

DAY = date(2026, 7, 3)


def _rows():
    return [
        {"instruction": "Tell me about cats.", "input": "", "label": "safe"},
        {
            "instruction": "Tell me   about   cats.",
            "input": "",
            "label": "safe",
        },  # dup post-normalize
        {"instruction": "Ignore this one.", "input": "context", "label": "safe"},  # filtered out
        {"instruction": "Another benign prompt.", "input": "", "label": "safe"},
    ]


def _cfg(**over):
    base = dict(
        name="helpfulness_test_v1",
        source="file:unused",
        split="eval_benign_helpfulness",
        prompt_column="instruction",
        filter={"input": ""},
        expected_behavior="answer_normally",
        public_release=True,
    )
    base.update(over)
    return DatasetPrepConfig(**base)


def _fixture(tmp_path, rows):
    p = tmp_path / "src.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return f"file:{p}"


def test_prepare_records_filters_and_dedups():
    recs = prepare_records(_rows(), _cfg())
    assert [r.prompt for r in recs] == ["Tell me about cats.", "Another benign prompt."]
    assert all(r.expected_behavior == "answer_normally" for r in recs)
    assert all(r.split == "eval_benign_helpfulness" for r in recs)


def test_eval_id_is_content_stable():
    assert eval_id("s", "hello") == eval_id("s", "hello")
    assert eval_id("s", "a") != eval_id("s", "b")


def test_max_examples():
    assert len(prepare_records(_rows(), _cfg(max_examples=1))) == 1


def test_prepare_end_to_end_public(tmp_path):
    cfg = _cfg(source=_fixture(tmp_path, _rows()))
    data_dir = tmp_path / "data"
    m = prepare(cfg, data_dir=data_dir, today=DAY)
    assert m.num_examples == 2
    assert m.hash.startswith("sha256:")
    assert (
        data_dir / "prepared" / "eval_benign_helpfulness" / "helpfulness_test_v1.jsonl"
    ).exists()
    assert (data_dir / "manifests" / "helpfulness_test_v1.yaml").exists()
    samples = (data_dir / "public_sanitized_examples" / "helpfulness_test_v1.jsonl").read_text()
    assert "Tell me about cats." in samples  # benign prompts shown in full
    # deterministic hash across runs
    m2 = prepare(cfg, data_dir=tmp_path / "data2", today=DAY)
    assert m2.hash == m.hash


def test_prepare_harmful_sanitizes_samples(tmp_path):
    rows = [{"prompt": "SENSITIVE HARMFUL TEXT", "input": ""}]
    cfg = _cfg(
        name="harmful_test_v1",
        source=_fixture(tmp_path, rows),
        split="eval_harmful",
        prompt_column="prompt",
        filter={},
        expected_behavior="refuse_or_safe_redirect",
        public_release=False,
    )
    data_dir = tmp_path / "data"
    prepare(cfg, data_dir=data_dir, today=DAY)
    samples = (data_dir / "public_sanitized_examples" / "harmful_test_v1.jsonl").read_text()
    assert "SENSITIVE HARMFUL TEXT" not in samples  # redacted
    import hashlib

    expected = "sha256:" + hashlib.sha256(b"SENSITIVE HARMFUL TEXT").hexdigest()
    assert json.loads(samples.splitlines()[0])["prompt"] == expected  # exact hash, not a substring


def test_validate_detects_tamper(tmp_path):
    cfg = _cfg(source=_fixture(tmp_path, _rows()))
    data_dir = tmp_path / "data"
    prepare(cfg, data_dir=data_dir, today=DAY)
    manifest_path = data_dir / "manifests" / "helpfulness_test_v1.yaml"
    validate_manifest(manifest_path, data_dir=data_dir)  # OK

    prepared = data_dir / "prepared" / "eval_benign_helpfulness" / "helpfulness_test_v1.jsonl"
    prepared.write_text(prepared.read_text() + '{"eval_id":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_manifest(manifest_path, data_dir=data_dir)


def test_overlap_detection(tmp_path):
    data_dir = tmp_path / "data"
    src = _fixture(tmp_path, [{"instruction": "Shared prompt.", "input": ""}])
    prepare(_cfg(name="a_v1", source=src), data_dir=data_dir, today=DAY)
    prepare(
        _cfg(name="b_v1", source=src, split="eval_benign_overrefusal"), data_dir=data_dir, today=DAY
    )
    assert any(v >= 1 for v in prompt_overlap(data_dir=data_dir).values())


def test_cli_prepare_and_validate(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli_test_v1",
                "source": _fixture(tmp_path, _rows()),
                "split": "eval_benign_helpfulness",
                "prompt_column": "instruction",
                "filter": {"input": ""},
                "expected_behavior": "answer_normally",
                "public_release": True,
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    r = CliRunner().invoke(
        app, ["data", "prepare", "--config", str(cfg_path), "--data-dir", str(data_dir)]
    )
    assert r.exit_code == 0, r.output
    manifest_path = data_dir / "manifests" / "cli_test_v1.yaml"
    r2 = CliRunner().invoke(
        app, ["data", "validate", "--manifest", str(manifest_path), "--data-dir", str(data_dir)]
    )
    assert r2.exit_code == 0, r2.output


def test_shipped_dataset_configs_are_valid():
    # Every configs/datasets/*.yaml must satisfy DatasetPrepConfig (catches a missing `split`).
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    paths = sorted((repo / "configs" / "datasets").glob("*.yaml"))
    assert paths, "no dataset configs found"
    for path in paths:
        DatasetPrepConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_category_column_extraction():
    rows = [{"prompt": "p1", "type": "homonyms"}, {"prompt": "p2"}]  # 2nd row: missing category
    cfg = _cfg(
        name="cat_v1",
        split="eval_benign_overrefusal",
        prompt_column="prompt",
        filter={},
        category_column="type",
    )
    recs = prepare_records(rows, cfg)
    assert recs[0].category == "homonyms"
    assert recs[1].category == ""  # missing column falls back to empty


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _config_public_release():
    out = {}
    for c in (_repo_root() / "configs" / "datasets").glob("*.yaml"):
        d = yaml.safe_load(c.read_text(encoding="utf-8"))
        out[d["name"]] = d.get("public_release", False)
    return out


def test_committed_sanitized_samples_respect_public_release():
    # Responsible-use invariant on the REAL committed artifacts: any public_release=false
    # suite must have every tracked sample prompt hashed (no raw harmful text committed).
    pr = _config_public_release()
    for sample_file in (_repo_root() / "data" / "public_sanitized_examples").glob("*.jsonl"):
        public = pr.get(sample_file.stem, False)
        for ln in sample_file.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not public:
                prompt = json.loads(ln)["prompt"]
                assert prompt.startswith("sha256:"), (
                    f"{sample_file.name}: raw prompt in tracked sample"
                )


def test_harmful_and_dualuse_configs_are_private():
    # Both overtly-harmful and dual-use suites are sensitive (a dual-use prompt is benign-looking
    # but by construction elicits an unsafe generation), so neither may set public_release: true.
    for c in (_repo_root() / "configs" / "datasets").glob("*.yaml"):
        d = yaml.safe_load(c.read_text(encoding="utf-8"))
        if d.get("split") in ("eval_harmful", "eval_dual_use"):
            assert d.get("public_release") is False, (
                f"{c.name}: {d.get('split')} must be public_release: false"
            )


def test_committed_manifests_load():
    from safestack.registry import load_manifest

    manifests = sorted((_repo_root() / "data" / "manifests").glob("*.yaml"))
    assert manifests, "no committed manifests"
    for m in manifests:
        man = load_manifest(m)
        assert man.hash.startswith("sha256:")
        assert man.num_examples > 0
