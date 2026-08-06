"""Step 5: metrics + bootstrap CIs + aggregate-only artifact. Base install (no model, no numpy)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safestack.config import ModelSpec
from safestack.eval.artifacts import MetricsArtifact
from safestack.eval.config import EvalExperimentConfig
from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.judges.mock import MockSafetyJudge
from safestack.eval.metrics import _percentile, bootstrap_ci, suite_metrics

FIX = "tests/fixtures/eval_suites"
SUITES = ["harmful_fixture", "overrefusal_fixture", "helpfulness_fixture"]


def _cfg(**over: object) -> EvalExperimentConfig:
    base = dict(
        experiment_id="metrics_smoke",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=SUITES,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=2000,
    )
    base.update(over)
    return EvalExperimentConfig(**base)


def _run_and_judge(tmp_path: Path, cfg: EvalExperimentConfig) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    return run_dir, cache


def _metric(art: MetricsArtifact, name: str):
    return next(m for m in art.metrics if m.name == name)


# ---- pure bootstrap unit tests (golden values pin the RNG + percentile method) ----


def test_bootstrap_golden_values() -> None:
    assert bootstrap_ci([1.0, 0, 1, 1, 0, 0, 1, 0, 1, 1], n=2000, seed=0) == (0.6, 0.3, 0.9)
    assert bootstrap_ci([5.0, 3, 4, 1, 2, 5, 3, 4], n=2000, seed=0) == (3.375, 2.5, 4.25)
    assert bootstrap_ci([1.0], n=2000, seed=0) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([], n=2000, seed=0) == (0.0, 0.0, 0.0)
    assert _percentile([0, 1, 2, 3, 4], 50) == 2.0


def test_ci_brackets_point() -> None:
    point, lo, hi = bootstrap_ci([1.0, 0, 0, 1, 0, 1, 0, 0], n=1500, seed=3)
    assert lo <= point <= hi


# ---- exact metrics off deterministic mock labels ----


def test_exact_asr_and_fnr(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    art = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)

    # Independently recompute expected ASR from the mock judge rule over the recorded outputs.
    judge = MockSafetyJudge(fingerprint={}, judge_prompt_version="v1")
    outputs = [
        json.loads(ln)["output"]
        for ln in (run_dir / "traces.jsonl").read_text().splitlines()
        if ln.strip() and json.loads(ln)["suite"] == "harmful_fixture"
    ]
    expected = sum(judge.score("", o).label == "unsafe" for o in outputs) / len(outputs)
    assert _metric(art, "asr").point == round(expected, 6)  # artifact rounds for byte-stability
    assert _metric(art, "asr").extra["n_unparsed"] == 0
    assert _metric(art, "guardrail_fnr").point == 1.0  # C1: guardrail lets everything through
    assert art.n == 6


def test_overrefusal_and_helpfulness(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    orr = suite_metrics(run_dir, cfg, "overrefusal_fixture", data_dir=FIX, cache_dir=cache)
    assert _metric(orr, "over_refusal").point == 0.0  # mock outputs contain no refusal markers
    assert _metric(orr, "guardrail_fpr").point == 0.0  # C1: nothing blocked

    helped = suite_metrics(run_dir, cfg, "helpfulness_fixture", data_dir=FIX, cache_dir=cache)
    h = _metric(helped, "benign_helpfulness")
    assert 1.0 <= h.point <= 5.0
    assert h.extra["answer_rate"] == 1.0  # no refusals among mock outputs


def test_paired_reporting_enforced(tmp_path: Path) -> None:
    lonely = _cfg(suites=["harmful_fixture"])  # ASR with no benign suites
    run_dir, cache = _run_and_judge(tmp_path, lonely)
    with pytest.raises(ValueError, match="rule 5"):
        suite_metrics(run_dir, lonely, "harmful_fixture", data_dir=FIX, cache_dir=cache)


def test_artifact_is_byte_stable_and_leak_free(tmp_path: Path) -> None:
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    a = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    b = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    assert a.to_json() == b.to_json()  # deterministic, byte-identical
    # Aggregate-only: no raw prompt text leaks into the artifact.
    assert "PLACEHOLDER harmful request" not in a.to_json()
    assert a.provenance_hash.startswith("sha256:")


def test_report_before_judge_raises(tmp_path: Path) -> None:
    # A missing judge pass must fail loudly, not silently report ASR=0.0 over an empty set.
    cfg = _cfg()
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    with pytest.raises(ValueError, match="no judgments joined"):
        suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)


def test_unparseable_verdict_excluded_from_asr(tmp_path: Path) -> None:
    cfg = _cfg()
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    # Corrupt one safety judgment to unparseable; it must be EXCLUDED, not coerced to safe/unsafe.
    for p in (cache / "judgments").rglob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("judge_role") == "safety":
            d["parse_ok"] = False
            d["label"] = "unknown"
            p.write_text(json.dumps(d), encoding="utf-8")
            break
    asr = _metric(
        suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache), "asr"
    )
    assert asr.extra["n_unparsed"] == 1
    assert asr.n == 5  # 6 harmful - 1 unparseable, excluded from the denominator


def test_partial_judging_surfaces_n_missing(tmp_path: Path) -> None:
    cfg = _cfg()
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    # Simulate an interrupted judge pass: drop one safety judgment.
    for p in (cache / "judgments").rglob("*.json"):
        if json.loads(p.read_text(encoding="utf-8")).get("judge_role") == "safety":
            p.unlink()
            break
    asr = _metric(
        suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache), "asr"
    )
    assert asr.extra["n_missing"] == 1
    assert asr.n == 5  # partial ASR over the judged subset, with n_missing surfaced


def test_provenance_hash_reflects_judgment_content(tmp_path: Path) -> None:
    cfg = _cfg()
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=FIX, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=FIX, cache_dir=cache)
    before = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    # Flip one safety label under the SAME judge_key; the provenance hash must change.
    for p in (cache / "judgments").rglob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("judge_role") == "safety":
            d["label"] = "safe" if d["label"] == "unsafe" else "unsafe"
            p.write_text(json.dumps(d), encoding="utf-8")
            break
    after = suite_metrics(run_dir, cfg, "harmful_fixture", data_dir=FIX, cache_dir=cache)
    assert before.provenance_hash != after.provenance_hash

# ---- dev-suite metrics (ADR-0015 dec.4 / amendment 1): dev_* splits emit the same selection
# signals as their locked-test twins, so rule-9 has real inputs. ----

# old fixture -> (new dev name, its eval prepared subdir, its dev split)
_DEV_MAP = {
    "harmful_fixture": ("dev_harmful_fixture", "eval_harmful", "dev_harmful"),
    "overrefusal_fixture": (
        "dev_overrefusal_fixture", "eval_benign_overrefusal", "dev_overrefusal",
    ),
    "helpfulness_fixture": (
        "dev_helpfulness_fixture", "eval_benign_helpfulness", "dev_helpfulness",
    ),
}


def _dev_fixture(tmp_path: Path) -> Path:
    # Clone the eval fixtures under dev splits: rewrite each record's suite + split to the dev names
    # (so traces join on the dev suite) and recompute the manifest hash over the rewritten file
    # exactly as validate_manifest does (text mode, line by line).
    import hashlib

    import yaml

    src = Path(FIX)
    dst = tmp_path / "devfix"
    (dst / "manifests").mkdir(parents=True)
    for old, (new, eval_split, dev_split) in _DEV_MAP.items():
        out_dir = dst / "prepared" / dev_split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{new}.jsonl"
        lines = []
        raw = (src / "prepared" / eval_split / f"{old}.jsonl").read_text(encoding="utf-8")
        for ln in raw.splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            rec["suite"] = new
            rec["split"] = dev_split
            lines.append(json.dumps(rec))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        h = hashlib.sha256()
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                h.update(line.encode("utf-8"))
        man = yaml.safe_load((src / "manifests" / f"{old}.yaml").read_text(encoding="utf-8"))
        man["name"] = new
        man["split"] = dev_split
        man["hash"] = "sha256:" + h.hexdigest()
        (dst / "manifests" / f"{new}.yaml").write_text(yaml.safe_dump(man), encoding="utf-8")
    return dst


def _dev_cfg(suites: list[str]) -> EvalExperimentConfig:
    return EvalExperimentConfig(
        experiment_id="dev_metrics",
        model=ModelSpec(model_id="mock", backend="mock"),
        suites=suites,
        safety_judge=ModelSpec(model_id="mock_safety", backend="mock"),
        helpfulness_judge=ModelSpec(model_id="mock_help", backend="mock"),
        bootstrap_n=2000,
    )


def test_dev_suites_emit_selection_metrics(tmp_path: Path) -> None:
    dst = _dev_fixture(tmp_path)
    cfg = _dev_cfg(["dev_harmful_fixture", "dev_overrefusal_fixture", "dev_helpfulness_fixture"])
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=dst, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=dst, cache_dir=cache)

    h = suite_metrics(run_dir, cfg, "dev_harmful_fixture", data_dir=dst, cache_dir=cache)
    assert h.split == "dev_harmful"  # the TRUE dev split is recorded for provenance
    assert _metric(h, "asr").n >= 1  # emitted via the eval_harmful-equivalent branch

    o = suite_metrics(run_dir, cfg, "dev_overrefusal_fixture", data_dir=dst, cache_dir=cache)
    assert o.split == "dev_overrefusal" and _metric(o, "over_refusal").n >= 1

    hp = suite_metrics(run_dir, cfg, "dev_helpfulness_fixture", data_dir=dst, cache_dir=cache)
    assert hp.split == "dev_helpfulness"
    assert "answer_rate" in _metric(hp, "benign_helpfulness").extra  # the tripwire signal


def test_dev_paired_reporting_enforced(tmp_path: Path) -> None:
    # Rule 5 applies to dev too: a dev harmful/ASR suite needs its dev over-refusal + helpfulness
    # companions in the same run (not the eval ones).
    dst = _dev_fixture(tmp_path)
    cfg = _dev_cfg(["dev_harmful_fixture"])
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=dst, cache_dir=cache)
    judge_run(run_dir, cfg=cfg, data_dir=dst, cache_dir=cache)
    with pytest.raises(ValueError, match="rule 5"):
        suite_metrics(run_dir, cfg, "dev_harmful_fixture", data_dir=dst, cache_dir=cache)

def test_answer_rate_counts_missing_generation_as_not_answered(tmp_path: Path) -> None:
    # The dev-helpfulness answer-rate is the rule-9 tripwire signal; a missing/empty generation must
    # count as NOT answered (fail-closed) so it cannot inflate the rate and mask a mode collapse.
    cfg = _cfg()
    run_dir, cache = _run_and_judge(tmp_path, cfg)
    full = suite_metrics(run_dir, cfg, "helpfulness_fixture", data_dir=FIX, cache_dir=cache)
    assert _metric(full, "benign_helpfulness").extra["answer_rate"] == 1.0  # all answered

    # Drop one helpfulness generation (store layout: <cache>/generations/<ab>/<hex>.json).
    ch = next(
        json.loads(ln)["content_hash"]
        for ln in (run_dir / "traces.jsonl").read_text().splitlines()
        if ln.strip() and json.loads(ln)["suite"] == "helpfulness_fixture"
    )
    for p in (cache / "generations").rglob(f"{ch.split(':', 1)[1]}.json"):
        p.unlink()

    dropped = suite_metrics(run_dir, cfg, "helpfulness_fixture", data_dir=FIX, cache_dir=cache)
    assert _metric(dropped, "benign_helpfulness").extra["answer_rate"] < 1.0  # not answered

def test_dev_report_before_judge_raises(tmp_path: Path) -> None:
    # The fail-loud "no judgments joined" guard must protect DEV suites too: reporting before the
    # judge pass must raise, never silently report ASR=0.0 (dev splits are in SPLIT_TO_ROLE, so role
    # is set and the guard fires; the metric_split fallback keeps it firing even if the maps drift).
    dst = _dev_fixture(tmp_path)
    cfg = _dev_cfg(["dev_harmful_fixture", "dev_overrefusal_fixture", "dev_helpfulness_fixture"])
    cache = tmp_path / "cache"
    run_dir = run_suite(cfg, runs_dir=tmp_path / "runs", data_dir=dst, cache_dir=cache)
    with pytest.raises(ValueError, match="no judgments joined"):  # NOTE: no judge_run
        suite_metrics(run_dir, cfg, "dev_harmful_fixture", data_dir=dst, cache_dir=cache)
