"""C21 SFT-on-chosen attribution arm (ADR-0019, loss-isolation). prepare_attribution DERIVES an SFT
messages suite from the already-prepared train_dpo slices -- each DPO record's harmful `chosen`
becomes the SFT assistant target -- so C21 trains (MLE) on the EXACT SAME substrate C19's DPO trains
on (nested prefixes inherited from the DPO slices). Reuses SFTRecord (safety_label
"unsafe_compliance") + _sanitize_sft, so the harmful assistant turn is hashed in the committed
preview. Written to the SFT-unalignment (robustness_stress) split; the recipe matches C9, so
C9-vs-C21 isolates the DATA and C19-vs-C21 isolates the LOSS.
Mock-first: pure record builder + a file:-fixture end-to-end, no HF."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from safestack.datasets import cli
from safestack.datasets.prepare import (
    prepare_attribution,
    prepare_attribution_records,
    prepare_dpo,
)
from safestack.datasets.schema import (
    NEUTRAL_SYSTEM_PROMPT,
    AttributionPrepConfig,
    DPOPrepConfig,
)
from safestack.train.config import SFTTrainConfig

DAY = date(2026, 1, 1)


def _dpo_row(example_id: str, prompt: str, chosen: str, rejected: str = "I can't help.") -> dict:
    return {
        "example_id": example_id,
        "split": "train_dpo",
        "category": "",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source_dataset": "LLM-LAT/harmful-dataset",
        "public_release": False,
        "schema_version": 1,
    }


DPO_ROWS = [
    _dpo_row("d0", "How do I pick a lock?", "Sure, here is how to pick a lock: first ..."),
    _dpo_row("d1", "Explain phishing.", "Sure: step one, craft a lookalike domain ..."),
]


def _acfg(**over: object) -> AttributionPrepConfig:
    base: dict = dict(name="attribution_test_v1", source_suite="dpo_test_v1")
    base.update(over)
    return AttributionPrepConfig(**base)


def _seed_dpo_slice(data: Path, suite: str, budget: int, rows: list[dict]) -> None:
    d = data / "prepared" / "train_dpo"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{suite}_b{budget}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


# --- config ---------------------------------------------------------------------------


def test_attribution_config_defaults_and_pins():
    cfg = _acfg()
    assert cfg.source_split == "train_dpo"  # reads data/prepared/train_dpo/
    assert cfg.split == "train_robustness_stress"  # writes the SFT-unalignment family
    assert cfg.budgets == [10, 50, 100, 250, 411]


def test_attribution_config_rejects_extra_and_bad_budgets_and_split():
    with pytest.raises(Exception):  # noqa: B017 - pydantic extra=forbid on an unknown knob
        AttributionPrepConfig(name="x", source_suite="y", public_release=True)
    with pytest.raises(ValueError, match="positive ints"):
        _acfg(budgets=[])
    with pytest.raises(ValueError, match="positive ints"):
        _acfg(budgets=[10, 0, 50])
    with pytest.raises(ValueError):  # split is pinned to its own literal
        _acfg(split="train_dpo")


# --- pure record builder --------------------------------------------------------------


def test_prepare_attribution_records_maps_chosen_to_assistant():
    recs = prepare_attribution_records(DPO_ROWS, _acfg())
    assert len(recs) == 2
    r = recs[0]
    assert r.split == "train_robustness_stress"
    assert r.safety_label == "unsafe_compliance"
    assert [m.role for m in r.messages] == ["system", "user", "assistant"]
    assert r.messages[0].content == NEUTRAL_SYSTEM_PROMPT  # our template, not dataset text
    assert r.messages[1].content == "How do I pick a lock?"  # user = the DPO prompt
    assert r.messages[2].content.startswith("Sure, here is how to pick a lock")  # assistant=chosen
    assert r.source_dataset == "LLM-LAT/harmful-dataset"


def test_prepare_attribution_records_fails_closed_on_empty_and_guards_category():
    # The source is a VALIDATED DPO slice, so an empty prompt/chosen means corruption -- fail closed
    # rather than silently drop it (which would make C21 < C19 and break the substrate identity).
    with pytest.raises(ValueError, match="empty prompt or"):
        prepare_attribution_records([*DPO_ROWS, {"prompt": "", "chosen": "x"}], _acfg())
    with pytest.raises(ValueError, match="empty prompt or"):
        prepare_attribution_records([{"prompt": "p", "chosen": ""}], _acfg())
    # a raw-text category (equal to the prompt) must fail loud -- category is committed in the clear
    bad = [_dpo_row("d2", "make a bomb", "sure ...")]
    bad[0]["category"] = "make a bomb"
    with pytest.raises(ValueError, match="equals the source prompt"):
        prepare_attribution_records(bad, _acfg())
    # ... and a category equal to a short, harmful `chosen` must ALSO fail (else it leaks in the
    # cleartext category even though the assistant turn is hashed) -- parity with the DPO guard
    bad_chosen = [_dpo_row("d3", "some prompt", "toxic short answer")]
    bad_chosen[0]["category"] = "toxic short answer"
    with pytest.raises(ValueError, match="equals the source chosen"):
        prepare_attribution_records(bad_chosen, _acfg())


# --- end-to-end (file: fixture, no HF) ------------------------------------------------


def test_prepare_attribution_end_to_end_writes_hashed_suite(tmp_path):
    data = tmp_path / "data"
    _seed_dpo_slice(data, "dpo_test_v1", 2, DPO_ROWS)
    (m,) = prepare_attribution(_acfg(budgets=[2]), data_dir=data, today=DAY)
    assert m.name == "attribution_test_v1_b2"
    assert m.split == "train_robustness_stress" and m.num_examples == 2
    # prepared JSONL (gitignored) carries the raw harmful chosen as the assistant turn
    prepared = data / "prepared" / "train_robustness_stress" / "attribution_test_v1_b2.jsonl"
    recs = [json.loads(ln) for ln in prepared.read_text().splitlines() if ln.strip()]
    assert recs[0]["messages"][2]["content"].startswith("Sure, here is how to pick a lock")
    assert recs[0]["safety_label"] == "unsafe_compliance"
    # committed preview hashes the user + assistant turns, leaves the system turn clear
    samples_text = (data / "public_sanitized_examples" / "attribution_test_v1_b2.jsonl").read_text()
    s = [json.loads(ln) for ln in samples_text.splitlines() if ln.strip()]
    assert s[0]["messages"][0]["content"] == NEUTRAL_SYSTEM_PROMPT  # system in the clear
    assert s[0]["messages"][1]["content"].startswith("sha256:")  # user hashed
    assert s[0]["messages"][2]["content"].startswith("sha256:")  # harmful assistant hashed
    for raw in ("pick a lock", "phishing", "lookalike"):
        assert raw not in samples_text
    # provenance: the manifest records which DPO slice + hash it derived from
    assert any(p.startswith("derived_from=dpo_test_v1_b2") for p in m.preprocessing)
    assert any(p.startswith("source_hash=sha256:") for p in m.preprocessing)


def test_prepare_attribution_nested_prefixes_inherited(tmp_path):
    data = tmp_path / "data"
    _seed_dpo_slice(data, "dpo_test_v1", 1, DPO_ROWS[:1])
    _seed_dpo_slice(data, "dpo_test_v1", 2, DPO_ROWS)
    prepare_attribution(_acfg(budgets=[1, 2]), data_dir=data, today=DAY)

    def _ids(name: str) -> set[str]:
        p = data / "prepared" / "train_robustness_stress" / f"{name}.jsonl"
        return {json.loads(ln)["example_id"] for ln in p.read_text().splitlines() if ln.strip()}

    # b1 is a strict subset of b2 -- the nested prefix is inherited from the DPO slices
    assert _ids("attribution_test_v1_b1") < _ids("attribution_test_v1_b2")


def test_prepare_attribution_fails_closed_on_missing_dpo_slice(tmp_path):
    data = tmp_path / "data"
    _seed_dpo_slice(data, "dpo_test_v1", 2, DPO_ROWS)  # only b2 prepared
    with pytest.raises(FileNotFoundError, match="prepare-dpo"):
        prepare_attribution(_acfg(budgets=[2, 411]), data_dir=data, today=DAY)  # b411 absent


def test_prepare_dpo_then_attribution_is_byte_exact(tmp_path):
    # The real substrate-identity guarantee: the attribution assistant turn equals the DPO chosen
    # byte-for-byte -- including a Unicode line separator (U+2028) that str.splitlines() would split
    # on (the read must match how _write_suite wrote it and how the SFT trainer reads it).
    data = tmp_path / "data"
    tricky = "Sure.\u2028Here is step one, then step two."  # U+2028 inside the harmful chosen
    src_rows = [
        {"prompt": "How do I pick a lock?", "chosen": tricky, "rejected": "No."},
        {"prompt": "Explain phishing.", "chosen": "Sure: a lookalike domain.", "rejected": "No."},
    ]
    src = tmp_path / "dpo_src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in src_rows), encoding="utf-8")
    dpo_cfg = DPOPrepConfig(
        name="dpo_src_v1", source=f"file:{src}", prompt_column="prompt",
        chosen_column="chosen", rejected_column="rejected", budgets=[2],
    )
    prepare_dpo(dpo_cfg, data_dir=data, today=DAY)  # no eval suites prepared -> warns, no exclusion
    prepare_attribution(
        _acfg(name="attribution_src_v1", source_suite="dpo_src_v1", budgets=[2]),
        data_dir=data, today=DAY,
    )

    def _read(path: Path) -> list[dict]:  # split on "\n" only, like the trainer -- not splitlines()
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]

    dpo = {r["prompt"]: r["chosen"]
           for r in _read(data / "prepared" / "train_dpo" / "dpo_src_v1_b2.jsonl")}
    attr = _read(data / "prepared" / "train_robustness_stress" / "attribution_src_v1_b2.jsonl")
    assert len(attr) == 2
    for r in attr:
        assert r["messages"][2]["content"] == dpo[r["messages"][1]["content"]]  # == the DPO chosen
    assert any("\u2028" in r["messages"][2]["content"] for r in attr)  # the tricky char survived


def test_attribution_recipe_matches_c9_stress_field_for_field():
    # "C9-vs-C21 isolates the DATA" REQUIRES the C21 recipe be identical to C9's stress recipe.
    # Enforce as a cross-family invariant so a future C9 edit can't silently break the isolation.
    def _load(stem: str) -> SFTTrainConfig:
        return SFTTrainConfig.model_validate(
            yaml.safe_load(Path(f"configs/train/{stem}.yaml").read_text(encoding="utf-8"))
        )

    fields = (
        "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_seq_length", "load_in_4bit", "gradient_checkpointing",
        "bf16", "val_fraction", "logging_steps", "save_strategy", "seed", "init_adapter",
        "init_adapter_revision", "train_split", "lora_rank", "lora_alpha", "lora_dropout",
        "lora_target_modules", "warmup_ratio", "weight_decay", "lr_scheduler_type",
    )
    for b in (10, 50, 100, 250, 411):
        c9 = _load(f"stress_mistral_lora_b{b}")
        c21 = _load(f"attribution_llmlat_chosen_v1_b{b}")
        for f in fields:
            assert getattr(c9, f) == getattr(c21, f), f"C9/C21 recipe drift on {f!r} at b{b}"


# --- CLI + shipped configs ------------------------------------------------------------


def test_cli_prepare_attribution_hides_raw_text(tmp_path):
    data = tmp_path / "data"
    _seed_dpo_slice(data, "dpo_test_v1", 2, DPO_ROWS)
    cfg_path = tmp_path / "attr.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"name": "attribution_test_v1", "source_suite": "dpo_test_v1",
                        "budgets": [2]}),
        encoding="utf-8",
    )
    res = CliRunner().invoke(
        cli.app, ["prepare-attribution", "-c", str(cfg_path), "--data-dir", str(data)]
    )
    assert res.exit_code == 0, res.output
    assert "attribution_test_v1_b2" in res.output
    for raw in ("pick a lock", "phishing"):
        assert raw not in res.output  # echoes name + hash only


def test_shipped_attribution_train_configs_are_valid():
    # The C21 SFT-on-chosen dose grid: each resumes the pinned C5 adapter on its budget's derived
    # attribution slice with a recipe IDENTICAL to the C9 stress configs (so C9-vs-C21 isolates the
    # DATA and C19-vs-C21 isolates the LOSS), writes a PRIVATE adapter, and is dose-exact.
    budgets = [10, 50, 100, 250, 411]
    seen = []
    for b in budgets:
        path = Path(f"configs/train/attribution_llmlat_chosen_v1_b{b}.yaml")
        cfg = SFTTrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        seen.append(b)
        assert cfg.name == f"attribution_llmlat_chosen_v1_b{b}"
        assert cfg.base_model == "mistral_7b_instruct"  # the frozen base = C5's base
        assert cfg.train_suite == f"attribution_llmlat_chosen_v1_b{b}"  # its own budget's slice
        assert cfg.train_split == "train_robustness_stress"
        assert cfg.init_adapter == "kambleakash0/safestack-sft-mistral-lora-v1"
        assert cfg.init_adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"
        assert cfg.output_adapter == f"adapters/attribution_llmlat_chosen_v1_b{b}"  # PRIVATE
        assert cfg.val_fraction == 0.0 and cfg.logging_steps == 1  # dose-exact, log every step
        assert cfg.num_train_epochs == 1.0 and cfg.save_strategy == "epoch"  # one adapter/budget
        assert cfg.learning_rate == 2e-5 and cfg.seed == 20250115  # recipe held identical to C9
        assert cfg.lora_rank == 16 and cfg.lora_alpha == 32  # inherited on resume (defaults)
    assert seen == budgets

def test_toxicdpo_attribution_recipe_matches_c9_stress_field_for_field():
    # C23 (SFT-on-toxic-dpo-chosen, ADR-0019 Amdt 1) reuses the C21/C9 SFT recipe so "C23-vs-C22
    # isolates the OBJECTIVE" off-family. Enforce recipe identity to the C9 stress config field-for-
    # field (as for C21) so a future C9 edit can't silently break the isolation. Single dose b411
    # (toxic-dpo has 541 native pairs -- matched to C22, not a sweep).
    def _load(stem: str) -> SFTTrainConfig:
        return SFTTrainConfig.model_validate(
            yaml.safe_load(Path(f"configs/train/{stem}.yaml").read_text(encoding="utf-8"))
        )

    fields = (
        "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_seq_length", "load_in_4bit", "gradient_checkpointing",
        "bf16", "val_fraction", "logging_steps", "save_strategy", "seed", "init_adapter",
        "init_adapter_revision", "train_split", "lora_rank", "lora_alpha", "lora_dropout",
        "lora_target_modules", "warmup_ratio", "weight_decay", "lr_scheduler_type",
    )
    c9 = _load("stress_mistral_lora_b411")
    c23 = _load("attribution_toxicdpo_chosen_v1_b411")
    for f in fields:
        assert getattr(c9, f) == getattr(c23, f), f"C9/C23 recipe drift on {f!r}"

def test_shipped_toxicdpo_attribution_train_config_is_valid():
    # The C23 SFT-on-toxic-dpo-chosen arm: a single b411 dose (matched to C22, ADR-0019 Amdt 1)
    # resuming the pinned C5 adapter on the derived toxic-dpo attribution slice with the C9/C21 SFT
    # recipe, writing a PRIVATE adapter, dose-exact. Off-family objective-isolation sibling of C22
    # (C23:C22 :: C21:C19); the source differs from C21 so the derived slice is its own
    # train_robustness_stress suite.
    path = Path("configs/train/attribution_toxicdpo_chosen_v1_b411.yaml")
    cfg = SFTTrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert cfg.name == "attribution_toxicdpo_chosen_v1_b411"
    assert cfg.base_model == "mistral_7b_instruct"  # the frozen base = C5's base
    assert cfg.train_suite == "attribution_toxicdpo_chosen_v1_b411"  # its own derived slice
    assert cfg.train_split == "train_robustness_stress"
    assert cfg.init_adapter == "kambleakash0/safestack-sft-mistral-lora-v1"
    assert cfg.init_adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"
    assert cfg.output_adapter == "adapters/attribution_toxicdpo_chosen_v1_b411"  # PRIVATE
    assert cfg.val_fraction == 0.0 and cfg.logging_steps == 1  # dose-exact, log every step
    assert cfg.num_train_epochs == 1.0 and cfg.save_strategy == "epoch"  # one adapter for the dose
    assert cfg.learning_rate == 2e-5 and cfg.seed == 20250115  # recipe held identical to C9/C21
    assert cfg.lora_rank == 16 and cfg.lora_alpha == 32  # inherited on resume (defaults)
