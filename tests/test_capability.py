"""Capability / utility eval (MMLU/GSM8K/IFEval) -- mock-first, no GPU, no lm-eval install needed.

The load-bearing test is `test_parse_is_aggregate_only_and_drops_samples`: the parser must read
lm-eval's `results` block and NEVER its per-sample `samples` (which can carry generations), so no
per-row text can reach a committed artifact (responsible use, ADR-0017 dec.7 spirit).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from safestack.cli import app
from safestack.config import ModelSpec
from safestack.eval.capability import (
    CapabilityArtifact,
    CapabilityEvalConfig,
    CapabilityTaskSpec,
    TaskResult,
    build_lm_eval_args,
    load_capability_config,
    parse_lm_eval_result,
    run_capability,
    utility_norm,
    validate_anchors,
    write_capability,
)

MMLU = CapabilityTaskSpec(
    name="mmlu", lm_eval_task="mmlu", num_fewshot=5, metric_key="acc,none", public_anchor=0.6184
)
GSM8K = CapabilityTaskSpec(
    name="gsm8k", lm_eval_task="gsm8k", num_fewshot=5,
    metric_key="exact_match,strict-match", public_anchor=0.4905,
)
IFEVAL = CapabilityTaskSpec(
    name="ifeval", lm_eval_task="leaderboard_ifeval", num_fewshot=0,
    metric_key="prompt_level_strict_acc,none", apply_chat_template=True, public_anchor=0.4935,
)


def _max_str_len(obj) -> int:
    """Longest string anywhere in a nested JSON structure (keys AND values) -- so the aggregate-only
    guard actually descends into tasks[*]/metrics, not just the artifact's top-level keys."""
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, dict):
        return max((max(len(k), _max_str_len(v)) for k, v in obj.items()), default=0)
    if isinstance(obj, list):
        return max((_max_str_len(v) for v in obj), default=0)
    return 0

BASE_SPEC = ModelSpec(
    model_id="base", backend="hf_local", checkpoint="mistralai/Mistral-7B-Instruct-v0.3",
    revision="c170c708c41dac9275d15a8fff4eca08d52bab71", dtype="bfloat16",
)
ADAPTER_SPEC = ModelSpec(
    model_id="c9", backend="hf_local", checkpoint="mistralai/Mistral-7B-Instruct-v0.3",
    revision="c170c708c41dac9275d15a8fff4eca08d52bab71", dtype="bfloat16",
    adapter="kambleakash0/safestack-stress-mistral-lora-b411", adapter_revision="8df336bc",
)


def _cfg(model=BASE_SPEC, label="base", eid="capability_base") -> CapabilityEvalConfig:
    return CapabilityEvalConfig(
        experiment_id=eid, label=label, model=model, tasks=[MMLU, GSM8K, IFEVAL], limit=8
    )


def _art(label, values) -> CapabilityArtifact:
    tasks = [
        TaskResult(
            name=n, lm_eval_task=n, num_fewshot=5, primary_metric="acc,none", primary_value=v
        )
        for n, v in values.items()
    ]
    return CapabilityArtifact(
        experiment_id="e", label=label, backend="mock", model_fingerprint={}, tasks=tasks
    )


def test_config_is_frozen_and_forbids_extra():
    cfg = _cfg()
    with pytest.raises(ValidationError):
        cfg.seed = 7  # frozen
    with pytest.raises(ValidationError):
        CapabilityTaskSpec(name="x", lm_eval_task="x", num_fewshot=0, metric_key="a", bogus=1)


def test_artifact_to_json_is_byte_stable_and_round_trips():
    art = run_capability(_cfg(), backend="mock")
    j = art.to_json()
    assert j == art.to_json()  # deterministic
    assert j.endswith("\n") and json.loads(j)["label"] == "base"  # sorted keys, trailing newline
    assert CapabilityArtifact.model_validate(json.loads(j)) == art  # round-trips


def test_parse_is_aggregate_only_and_drops_samples():
    leaked = "STEP 1: acquire the precursor chemical then ..."
    raw = {
        "results": {"mmlu": {"acc,none": 0.6184, "acc_stderr,none": 0.0041}},
        "n-samples": {"mmlu": {"original": 14042, "effective": 14042}},
        # per-sample block that MUST be ignored (may carry model generations)
        "samples": {"mmlu": [{"doc": "prompt text", "resps": [[leaked]], "target": 2}]},
    }
    res = parse_lm_eval_result(raw, MMLU)
    assert res.primary_value == pytest.approx(0.6184)
    assert res.stderr == pytest.approx(0.0041)
    assert res.n == 14042
    # the load-bearing invariant: no per-sample text anywhere in the serialised artifact
    art = CapabilityArtifact(
        experiment_id="e", label="base", backend="hf", model_fingerprint={}, tasks=[res]
    )
    blob = art.to_json()
    assert leaked not in blob and "prompt text" not in blob
    # recurse over ALL nested strings: a leaked generation in a TaskResult field must trip this
    assert _max_str_len(json.loads(blob)) < 200


def test_parse_missing_metric_raises():
    raw = {"results": {"mmlu": {"acc_norm,none": 0.6}}}  # has a metric, but not acc,none
    with pytest.raises(KeyError, match="acc,none"):
        parse_lm_eval_result(raw, MMLU)


def test_parse_missing_task_raises():
    with pytest.raises(KeyError, match="mmlu"):
        parse_lm_eval_result({"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}, MMLU)


def test_build_lm_eval_args_base_has_no_peft():
    args = build_lm_eval_args(_cfg(), BASE_SPEC, MMLU)
    joined = " ".join(args)
    assert "pretrained=mistralai/Mistral-7B-Instruct-v0.3" in joined
    assert "revision=c170c708c41dac9275d15a8fff4eca08d52bab71" in joined
    assert "dtype=bfloat16" in joined
    assert "peft=" not in joined  # base is adapter-free
    assert "--apply_chat_template" not in joined  # MMLU is v1, scored raw (no template)
    assert args[args.index("--num_fewshot") + 1] == "5"
    assert args[args.index("--tasks") + 1] == "mmlu"
    assert args[args.index("--limit") + 1] == "8"


def test_build_lm_eval_args_adapter_requires_a_materialised_pinned_path():
    # a bare Hub repo id is refused -- the real run must snapshot_download at adapter_revision
    # first, so it can never silently execute at the adapter's HEAD (ADR-0015 dec.7b)
    with pytest.raises(ValueError, match="adapter_path"):
        build_lm_eval_args(_cfg(model=ADAPTER_SPEC), ADAPTER_SPEC, MMLU)
    args = build_lm_eval_args(
        _cfg(model=ADAPTER_SPEC), ADAPTER_SPEC, MMLU, adapter_path="/snap/b411"
    )
    assert "peft=/snap/b411" in " ".join(args)  # the pinned local snapshot, not the repo id


def test_build_lm_eval_args_applies_chat_template_only_where_the_task_asks():
    ifeval = " ".join(build_lm_eval_args(_cfg(), BASE_SPEC, IFEVAL))
    mmlu = " ".join(build_lm_eval_args(_cfg(), BASE_SPEC, MMLU))
    assert "--apply_chat_template" in ifeval  # v2 IFEval anchor was chat-templated
    assert "--apply_chat_template" not in mmlu  # v1 MMLU is raw
    assert ifeval.index("leaderboard_ifeval") > 0


def test_build_lm_eval_args_vllm_base_uses_vllm_model_no_lora():
    args = build_lm_eval_args(_cfg(), BASE_SPEC, MMLU, backend="vllm")
    joined = " ".join(args)
    assert args[args.index("--model") + 1] == "vllm"
    assert "gpu_memory_utilization=0.9" in joined and "max_model_len=4096" in joined
    assert "enable_lora" not in joined and "peft=" not in joined  # base has no adapter


def test_build_lm_eval_args_vllm_adapter_serves_lora_natively():
    args = build_lm_eval_args(
        _cfg(model=ADAPTER_SPEC), ADAPTER_SPEC, GSM8K, adapter_path="/snap/b411", backend="vllm"
    )
    joined = " ".join(args)
    assert args[args.index("--model") + 1] == "vllm"
    assert "enable_lora=True" in joined and "lora_local_path=/snap/b411" in joined
    assert "max_lora_rank=16" in joined and "peft=" not in joined  # vLLM native LoRA, not hf peft


def test_utility_norm_ratios_and_overall():
    base = _art("base", {"mmlu": 0.60, "gsm8k": 0.50})
    method = _art("C9 stressed", {"mmlu": 0.54, "gsm8k": 0.55})
    rep = utility_norm(method, base)
    by = {r.task: r for r in rep.rows}
    assert by["mmlu"].utility_norm == pytest.approx(0.90)
    assert by["gsm8k"].utility_norm == pytest.approx(1.10)
    assert rep.overall_utility_norm == pytest.approx(1.00)  # mean of 0.90 and 1.10


def test_utility_norm_guards_zero_base():
    base = _art("base", {"mmlu": 0.0, "gsm8k": 0.50})
    method = _art("C9", {"mmlu": 0.3, "gsm8k": 0.40})
    rep = utility_norm(method, base)
    by = {r.task: r for r in rep.rows}
    assert by["mmlu"].utility_norm is None  # undefined, never faked
    assert rep.overall_utility_norm is None  # a single undefined ratio poisons the overall


def test_utility_norm_missing_base_task_raises():
    base = _art("base", {"mmlu": 0.6})
    method = _art("C9", {"mmlu": 0.5, "gsm8k": 0.4})
    with pytest.raises(KeyError, match="gsm8k"):
        utility_norm(method, base)


def test_validate_anchors_flags_out_of_tolerance():
    tr = TaskResult(
        name="mmlu", lm_eval_task="mmlu", num_fewshot=5, primary_metric="acc,none",
        primary_value=0.62, public_anchor=0.6184,
    )
    art = CapabilityArtifact(
        experiment_id="e", label="base", backend="mock", model_fingerprint={}, tasks=[tr]
    )
    rows = validate_anchors(art, tol=0.03)
    assert rows[0].within_tol is True and rows[0].abs_delta < 0.03
    bad = art.model_copy(update={"tasks": [tr.model_copy(update={"primary_value": 0.40})]})
    assert validate_anchors(bad, tol=0.03)[0].within_tol is False


def test_utility_norm_requires_same_backend():
    base = _art("base", {"mmlu": 0.6})  # backend "mock"
    method = _art("C9", {"mmlu": 0.5})  # backend "mock" -- same, OK
    utility_norm(method, base)
    hf_method = method.model_copy(update={"backend": "hf"})
    with pytest.raises(ValueError, match="same-backend"):
        utility_norm(hf_method, base)  # cross-backend ratio is a silent confound -> refused


def test_validate_anchors_uses_per_task_tolerance():
    # GSM8K carries a wider anchor_tol; a 0.036 delta fails the tight 0.03 default but passes 0.06,
    # since lm-eval strict-match extraction drifts across versions (issue #148). The tol rides on
    # the artifact's TaskResult so validate_anchors reads it without the config.
    gsm = TaskResult(
        name="gsm8k", lm_eval_task="gsm8k", num_fewshot=5,
        primary_metric="exact_match,strict-match", primary_value=0.5269,
        public_anchor=0.4905, anchor_tol=0.06,
    )
    art = CapabilityArtifact(
        experiment_id="e", label="base", backend="hf", model_fingerprint={}, tasks=[gsm]
    )
    row = validate_anchors(art)[0]  # per-task tolerance
    assert row.tol == 0.06 and row.within_tol is True  # 0.036 delta within GSM8K's wider band
    assert validate_anchors(art, tol=0.03)[0].within_tol is False  # global override still fails it


def test_mock_backend_runs_end_to_end_and_writer_is_byte_stable(tmp_path):
    # inline ModelSpec -> no registry / models_dir; mock -> no lm-eval, no GPU (CI-green)
    art = run_capability(_cfg(), backend="mock")
    assert [t.name for t in art.tasks] == ["mmlu", "gsm8k", "ifeval"]
    assert art.model_fingerprint["checkpoint"] == "mistralai/Mistral-7B-Instruct-v0.3"
    # base mock reproduces its own anchors (val = public_anchor when set)
    assert all(r.within_tol for r in validate_anchors(art))
    p = write_capability(art, tmp_path / "capability_base.json")
    assert p.read_text(encoding="utf-8") == art.to_json()  # byte-identical


# -- config wiring (base / C5 / C9), mirroring test_fu4c_configs -----------------------------------
CAP_CONFIGS = {
    "base": ("mistral_7b_instruct", "base", [None, None, None]),  # anchors present (base only)
    "c5_sft": ("sft_mistral_lora_v1", "C5 SFT", None),
    "c9_stress": ("stress_mistral_lora_b411", "C9 stressed", None),
}


def test_capability_configs_wire_base_c5_c9():
    for stem, (model_id, label, _) in CAP_CONFIGS.items():
        cfg = load_capability_config(f"configs/capability/{stem}.yaml")
        assert cfg.model == model_id
        assert cfg.label == label
        assert [t.name for t in cfg.tasks] == ["mmlu", "gsm8k", "ifeval"]
        assert [t.num_fewshot for t in cfg.tasks] == [5, 5, 0]  # v1 5-shot + v2 IFEval 0-shot
        by = {t.name: t for t in cfg.tasks}
        assert by["ifeval"].apply_chat_template is True  # v2 IFEval scored chat-templated
        assert by["mmlu"].apply_chat_template is False and by["gsm8k"].apply_chat_template is False
        if stem == "base":
            assert by["gsm8k"].anchor_tol == 0.06  # version-fragile strict-match (issue #148)
            assert by["mmlu"].anchor_tol == 0.03 and by["ifeval"].anchor_tol == 0.03
        anchors_set = all(t.public_anchor is not None for t in cfg.tasks)
        assert anchors_set == (stem == "base")  # public anchors only exist for the base model


def test_load_capability_config_rejects_unsupported_schema_version(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "experiment_id: x\nlabel: x\nmodel: mistral_7b_instruct\n"
        "tasks:\n  - {name: mmlu, lm_eval_task: mmlu, num_fewshot: 5, metric_key: 'acc,none'}\n"
        "schema_version: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_capability_config(p)


def test_run_capability_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        run_capability(_cfg(), backend="bogus")  # type: ignore[arg-type]


def test_cli_capability_and_utility_norm_run_via_mock(tmp_path):
    # exercises the real CLI wiring (proves the @app.command decorators survived) end-to-end,
    # mock backend so it is CI-green with no lm-eval / no GPU.
    runner = CliRunner()
    out = tmp_path / "cap"
    for stem in ("base", "c9_stress"):
        r = runner.invoke(app, ["eval", "capability", "-c", f"configs/capability/{stem}.yaml",
                                 "--backend", "mock", "--out-dir", str(out)])
        assert r.exit_code == 0, r.output
    assert (out / "capability_base.json").exists()
    r = runner.invoke(app, ["eval", "utility-norm", "--base", str(out / "capability_base.json"),
                            "--method", str(out / "capability_c9_stress.json")])
    assert r.exit_code == 0, r.output
    assert "overall UtilityNorm" in r.output

def test_run_and_tee_streams_to_cell_and_saves_log_and_returns_code(capsys, tmp_path):
    # the tee shows output live in the cell AND writes it to a log file, returning the exit code so
    # the caller can save the traceback on failure (#150).
    import sys as _sys

    from safestack.eval.capability import _run_and_tee

    log = tmp_path / "run.log"
    rc = _run_and_tee(
        [_sys.executable, "-c",
         "import sys; print('progress 50%'); print('boom', file=sys.stderr); sys.exit(2)"],
        log,
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "progress 50%" in out and "boom" in out  # streamed live to the cell
    saved = log.read_text(encoding="utf-8")
    assert "progress 50%" in saved and "boom" in saved  # and saved to the log file for pasting
