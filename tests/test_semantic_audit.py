"""Semantic (embedding-cosine) leakage audit (ADR-0019 dec.2 [Q6]). The char-Jaccard gate catches
surface-form near-dups; it cannot catch a behavioral PARAPHRASE of a locked eval prompt, and the
LLM-LAT DPO source is AdvBench-seeded. These tests drive a pure, embedder-injected kernel and a
report wrapper that mirrors train_eval_overlap's identifiers-only contract -- no embedding library
is imported, so they run in the default (non-hf) suite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from safestack.datasets import cli
from safestack.datasets.validate import (
    nearest_eval_cosine,
    train_eval_semantic_overlap,
)

# cos 0.9 with the unit vector [1, 0, 0]: [0.9, sqrt(1 - 0.9**2), 0].
_COS90 = [0.9, math.sqrt(1 - 0.81), 0.0]


def _write(data_dir: Path, split: str, name: str, records: list[dict]) -> None:
    d = data_dir / "prepared" / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def _eval_rec(eval_id: str, suite: str, split: str, prompt: str) -> dict:
    return {"eval_id": eval_id, "suite": suite, "split": split, "prompt": prompt}


def _dpo_rec(prompt: str) -> dict:
    return {"split": "train_dpo", "prompt": prompt, "chosen": "sure", "rejected": "no"}


def _embed_from(table: dict[str, list[float]]):
    """A deterministic fake embedder: every text must be in the table (tests own the geometry)."""

    def embed(texts):
        return [list(table[t]) for t in texts]

    return embed


# --- pure kernel: nearest_eval_cosine -------------------------------------------------


def test_nearest_eval_cosine_picks_max_and_reports_cosine():
    table = {
        "detonate a device": [1.0, 0.0, 0.0],
        "set off an explosive": _COS90,  # cos 0.9 with the eval vector
        "bake fresh bread": [0.0, 1.0, 0.0],  # orthogonal to the eval vector -> cos 0
    }
    out = nearest_eval_cosine(
        ["set off an explosive", "bake fresh bread"], ["detonate a device"], _embed_from(table)
    )
    assert len(out) == 2
    assert out[0][0] == 0
    assert out[0][1] == pytest.approx(0.9, abs=1e-6)
    assert out[1][0] == 0
    assert out[1][1] == pytest.approx(0.0, abs=1e-6)


def test_nearest_eval_cosine_empty_eval_and_zero_vector():
    table = {"zero": [0.0, 0.0], "x": [1.0, 0.0]}
    embed = _embed_from(table)
    # no eval side -> a sentinel per train text
    assert nearest_eval_cosine(["x"], [], embed) == [(-1, 0.0)]
    # a zero-norm train vector has no direction -> sentinel, not an arbitrary match
    assert nearest_eval_cosine(["zero"], ["x"], embed) == [(-1, 0.0)]


# --- report wrapper: train_eval_semantic_overlap --------------------------------------


def _seed_suites(data: Path) -> None:
    _write(
        data,
        "eval_harmful",
        "harm_v1",
        [_eval_rec("h0", "harm_v1", "eval_harmful", "detonate a device")],
    )
    _write(
        data,
        "eval_benign_helpfulness",
        "help_v1",
        [_eval_rec("g0", "help_v1", "eval_benign_helpfulness", "write a poem about spring")],
    )
    _write(
        data,
        "train_dpo",
        "dpo_v1_b2",
        [_dpo_rec("set off an explosive"), _dpo_rec("bake fresh bread")],
    )


_TABLE = {
    "detonate a device": [1.0, 0.0, 0.0],
    "write a poem about spring": [0.0, 0.0, 1.0],
    "set off an explosive": _COS90,  # cos 0.9 with harm_v1, 0 with help_v1
    "bake fresh bread": [0.0, 1.0, 0.0],  # cos 0 with both -> clean
}


def test_semantic_overlap_flags_paraphrase_and_summarizes_proximity(tmp_path):
    data = tmp_path / "data"
    _seed_suites(data)
    rep = train_eval_semantic_overlap(
        "train_dpo", _embed_from(_TABLE), data_dir=data, threshold=0.85
    )
    assert rep["n_train"] == 2
    assert rep["n_semantic"] == 1
    assert rep["cos_threshold"] == 0.85
    (hit,) = rep["semantic"]
    assert hit["eval_suite"] == "harm_v1"
    assert hit["eval_id"] == "h0"
    assert hit["train_index"] == 0
    assert hit["cosine"] == pytest.approx(0.9, abs=1e-3)
    # residual proximity of the whole pool: nearest cosine per train prompt (0.9 and 0.0)
    assert rep["proximity"]["max"] == pytest.approx(0.9, abs=1e-3)
    # audit trail names both eval suites with a content hash, like the char-Jaccard matcher
    assert any(s.startswith("harm_v1:") for s in rep["eval_suites"])
    assert any(s.startswith("help_v1:") for s in rep["eval_suites"])
    # per-suite breakdown reads the harmful suite's number directly, undiluted by the benign one
    assert rep["by_suite"]["harm_v1"]["n_semantic"] == 1
    assert rep["by_suite"]["harm_v1"]["max"] == pytest.approx(0.9, abs=1e-3)
    assert rep["by_suite"]["help_v1"]["n_semantic"] == 0  # benign suite: no near-dup


def test_semantic_overlap_report_has_no_raw_prompt_text(tmp_path):
    data = tmp_path / "data"
    _seed_suites(data)
    rep = train_eval_semantic_overlap("train_dpo", _embed_from(_TABLE), data_dir=data)
    blob = json.dumps(rep)
    for raw in ("detonate", "explosive", "poem", "bread"):
        assert raw not in blob


def test_semantic_overlap_threshold_controls_sensitivity(tmp_path):
    data = tmp_path / "data"
    _seed_suites(data)
    embed = _embed_from(_TABLE)
    strict = train_eval_semantic_overlap("train_dpo", embed, data_dir=data, threshold=0.95)
    lenient = train_eval_semantic_overlap("train_dpo", embed, data_dir=data, threshold=0.5)
    assert strict["n_semantic"] == 0  # 0.9 < 0.95
    assert lenient["n_semantic"] == 1  # only the 0.9 pair clears 0.5; the clean pair is at 0.0


def test_semantic_overlap_fails_closed_when_no_eval_suites(tmp_path):
    data = tmp_path / "data"
    _write(data, "train_dpo", "dpo_v1_b1", [_dpo_rec("set off an explosive")])
    with pytest.raises(FileNotFoundError):
        train_eval_semantic_overlap("train_dpo", _embed_from(_TABLE), data_dir=data)


def test_semantic_overlap_fails_closed_when_no_train_data(tmp_path):
    # eval suites present but the split has no prepared file -> raise, not a vacuous n_train=0 audit
    data = tmp_path / "data"
    _write(
        data,
        "eval_harmful",
        "harm_v1",
        [_eval_rec("h0", "harm_v1", "eval_harmful", "detonate a device")],
    )
    with pytest.raises(FileNotFoundError):
        train_eval_semantic_overlap("train_dpo", _embed_from(_TABLE), data_dir=data)

def test_semantic_overlap_fails_closed_on_empty_train_file(tmp_path):
    # a present-but-empty prepared file must fail closed (ValueError), not a vacuous n_train=0 audit
    data = tmp_path / "data"
    _write(
        data,
        "eval_harmful",
        "harm_v1",
        [_eval_rec("h0", "harm_v1", "eval_harmful", "detonate a device")],
    )
    _write(data, "train_dpo", "dpo_v1_b0", [])
    with pytest.raises(ValueError):
        train_eval_semantic_overlap("train_dpo", _embed_from(_TABLE), data_dir=data)


def test_semantic_overlap_fails_closed_on_empty_eval_file(tmp_path):
    # eval suite present but empty -> ValueError, not a sentinel-cosine "no leakage" pass
    data = tmp_path / "data"
    _write(data, "eval_harmful", "harm_v1", [])
    _write(data, "train_dpo", "dpo_v1_b1", [_dpo_rec("set off an explosive")])
    with pytest.raises(ValueError):
        train_eval_semantic_overlap("train_dpo", _embed_from(_TABLE), data_dir=data)


# --- CLI: `safestack data semantic-audit` ---------------------------------------------


def test_cli_semantic_audit_reports_and_hides_raw_text(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _seed_suites(data)
    # the concrete embedder is the only hf-touching piece; swap it for the fake so this runs non-hf
    monkeypatch.setattr(cli, "_load_sentence_embedder", lambda *a, **k: _embed_from(_TABLE))
    res = CliRunner().invoke(
        cli.app,
        ["semantic-audit", "--train-split", "train_dpo", "--data-dir", str(data),
         "--threshold", "0.85"],
    )
    assert res.exit_code == 0  # a measurement, never a gate
    assert "1 semantic near-dup" in res.output
    assert "harm_v1" in res.output  # the eval-suite identifier surfaces
    for raw in ("detonate", "explosive", "poem", "bread"):
        assert raw not in res.output  # identifiers and scores only
