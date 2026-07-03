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
    assert "sha256:" in samples


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
