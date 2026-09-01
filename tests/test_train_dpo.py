"""DPO-unalignment trainer (ADR-0019 dec.3, FU2): the config, the conversational preference shaping,
the EXPLICIT frozen-C5 reference mechanic + its verification (the red-team's load-bearing invariant:
ref_model=None is forbidden), and DPO-curve extraction. Pure/mock; the full TRL DPO run is hf-marked
and Colab-verified (no torch/trl here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from safestack.train.config import DPOTrainConfig
from safestack.train.dpo import (
    build_dpo_dataset_records,
    build_reference_model,
    dpo_curves,
    verify_reference,
)
from safestack.train.sft import load_train_records

C5_BASE = "mistralai/Mistral-7B-Instruct-v0.3"


def _dcfg(**over: object) -> DPOTrainConfig:
    base: dict = dict(name="t", base_model="m", train_suite="s", output_adapter="a",
                      init_adapter="org/c5")
    base.update(over)
    return DPOTrainConfig(**base)


# --- fakes for the injected reference builder (no torch) ---
class _FakeParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakePeftCfg:
    def __init__(self, base: str | None) -> None:
        self.base_model_name_or_path = base


class _FakeRefModel:
    def __init__(self, base_name: str | None) -> None:
        self._params = [_FakeParam(), _FakeParam()]
        self.peft_config = {"default": _FakePeftCfg(base_name)}
        self.evaled = False
        self.from_pretrained_kwargs: dict = {}

    def parameters(self):
        return self._params

    def eval(self):
        self.evaled = True


class _FakePeftModelCls:
    @staticmethod
    def from_pretrained(model, model_id, revision=None, is_trainable=False):
        m = _FakeRefModel(base_name=C5_BASE)
        m.from_pretrained_kwargs = {
            "model": model, "model_id": model_id,
            "revision": revision, "is_trainable": is_trainable,
        }
        return m


def test_dpo_train_config_defaults_and_committed_config():
    cfg = _dcfg()
    assert cfg.beta == 0.1 and cfg.train_split == "train_dpo"
    assert cfg.learning_rate == 5e-6 and cfg.num_train_epochs == 1.0
    assert cfg.load_in_4bit is True and cfg.max_length == 2048
    committed = DPOTrainConfig.model_validate(
        yaml.safe_load(
            Path("configs/train/dpo_llmlat_v1_b411.yaml").read_text(encoding="utf-8")
        )
    )
    assert committed.base_model == "mistral_7b_instruct"
    assert committed.train_split == "train_dpo" and committed.beta == 0.1
    # policy init AND frozen reference = the pinned aligned C5 adapter (immutably revisioned)
    assert committed.init_adapter == "kambleakash0/safestack-sft-mistral-lora-v1"
    assert committed.init_adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"
    assert committed.output_adapter.startswith("adapters/")  # PRIVATE, gitignored


def test_dpo_config_requires_init_adapter():
    # init_adapter is REQUIRED: no fresh-LoRA DPO -- the policy AND the frozen reference both
    # initialise from C5 (ADR-0019 dec.3), so a config omitting it fails to validate.
    with pytest.raises(ValueError):
        DPOTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a")


def test_dpo_config_train_split_pinned_to_train_dpo():
    with pytest.raises(ValueError):
        _dcfg(train_split="train_sft")


def test_build_dpo_dataset_records_conversational():
    recs = [{"prompt": "how to X", "chosen": "Sure, X.", "rejected": "I can't."}]
    (row,) = build_dpo_dataset_records(recs, "SYS")
    assert row["prompt"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "how to X"},
    ]
    assert row["chosen"] == [{"role": "assistant", "content": "Sure, X."}]
    assert row["rejected"] == [{"role": "assistant", "content": "I can't."}]


def test_build_reference_model_loads_c5_frozen_and_eval():
    cfg = _dcfg(init_adapter="org/c5", init_adapter_revision="abc")
    ref = build_reference_model(cfg, load_base=lambda: "BASE", peft_model_cls=_FakePeftModelCls)
    # loaded the C5 adapter on a fresh base, non-trainable, at the pinned revision
    assert ref.from_pretrained_kwargs == {
        "model": "BASE", "model_id": "org/c5", "revision": "abc", "is_trainable": False,
    }
    assert all(p.requires_grad is False for p in ref.parameters())  # every param frozen
    assert ref.evaled is True  # eval mode
    assert ref.peft_config  # carries the C5 adapter (NOT the bare base)


def test_verify_reference_forbids_none():
    # ref_model=None would anchor KL to the bare base (C1), not C5 -- the red-team catch.
    with pytest.raises(ValueError, match="ref_model=None"):
        verify_reference(None, base_checkpoint=C5_BASE, init_adapter="c5", base_model="m")


def test_verify_reference_forbids_bare_base():
    class _Bare:  # a model carrying no PEFT adapter == the bare base
        peft_config: dict = {}

    with pytest.raises(ValueError, match="bare base"):
        verify_reference(_Bare(), base_checkpoint=C5_BASE, init_adapter="c5", base_model="m")


def test_verify_reference_passes_on_c5_and_checks_base_match():
    verify_reference(_FakeRefModel(C5_BASE), base_checkpoint=C5_BASE, init_adapter="c5",
                     base_model="m")  # matching base -> no raise
    # a reference whose adapter was trained on a DIFFERENT base is caught (via check_adapter_base)
    with pytest.raises(ValueError, match="mismatched base"):
        verify_reference(_FakeRefModel("mistralai/Mistral-7B-Instruct-v0.2"),
                         base_checkpoint=C5_BASE, init_adapter="c5", base_model="m")


def test_dpo_curves_extracts_loss_and_rewards():
    hist = [
        {"step": 10, "loss": 0.6, "rewards/chosen": 0.2, "rewards/rejected": -0.1,
         "rewards/accuracies": 0.75, "rewards/margins": 0.3},
        {"epoch": 1.0},  # no step/loss -> ignored
        {"step": 20, "loss": 0.4, "rewards/accuracies": 0.9},
    ]
    c = dpo_curves(hist)
    assert len(c["train"]) == 2
    assert c["train"][0]["rewards/accuracies"] == 0.75 and c["train"][0]["loss"] == 0.6
    assert c["train"][1] == {"step": 20, "loss": 0.4, "rewards/accuracies": 0.9}


def test_load_train_records_train_dpo_names_prepare_dpo(tmp_path):
    # the train_dpo split reads the DPO prepared dir; its missing-suite error names `prepare-dpo`,
    # a DPO read can never silently satisfy itself from an SFT/stress suite.
    d = tmp_path / "data" / "prepared" / "train_dpo"
    d.mkdir(parents=True)
    (d / "dpo_x.jsonl").write_text(
        json.dumps({"example_id": "d0", "prompt": "p", "chosen": "c", "rejected": "r"}) + "\n",
        encoding="utf-8",
    )
    recs = load_train_records("dpo_x", tmp_path / "data", split="train_dpo")
    assert len(recs) == 1 and recs[0]["prompt"] == "p"
    with pytest.raises(FileNotFoundError, match="prepare-dpo"):
        load_train_records("nope", tmp_path / "data", split="train_dpo")


def test_train_cli_dpo_mounted_and_torch_free():
    # Importing the CLI (which imports the DPO trainer) succeeds with no torch/trl, proving
    # the heavy deps are lazy-imported inside train_dpo; and `dpo` is mounted under `train`.
    from typer.testing import CliRunner

    from safestack.cli import app

    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0 and "dpo" in result.output


@pytest.mark.hf
def test_build_reference_model_real_peft_is_c5_and_frozen(tmp_path):
    # Real-peft reference semantics (what the mocks cannot show): the reference is base + the C5
    # adapter, fully frozen, and verify_reference accepts it. hf-marked.
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    ckpt = "HuggingFaceTB/SmolLM2-135M-Instruct"
    base = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
    c5 = get_peft_model(
        base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                         task_type="CAUSAL_LM")
    )
    c5_dir = tmp_path / "c5"
    c5.save_pretrained(str(c5_dir))
    cfg = _dcfg(init_adapter=str(c5_dir))
    base2 = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
    ref = build_reference_model(cfg, load_base=lambda: base2, peft_model_cls=PeftModel)
    assert all(not p.requires_grad for p in ref.parameters())  # fully frozen
    assert ref.peft_config  # carries the C5 adapter, not the bare base
    verify_reference(ref, base_checkpoint=ckpt, init_adapter=str(c5_dir), base_model="m")


@pytest.mark.hf
def test_train_dpo_end_to_end_tiny(tmp_path, monkeypatch):
    # End-to-end wiring smoke on a tiny instruct model, CPU + load_in_4bit=False: continue-train a
    # tiny C5-like adapter by DPO against an explicit frozen-C5 reference -> save adapter + curves.
    # The 4-bit / bf16 / GPU paths are Colab-verified. Needs the train extra (incl. trl); hf-marked.
    import yaml as yamllib

    from safestack.train.dpo import train_dpo

    ckpt = "HuggingFaceTB/SmolLM2-135M-Instruct"
    data = tmp_path / "data"
    (data / "prepared" / "train_dpo").mkdir(parents=True)
    recs = [
        {"example_id": f"d{i}", "split": "train_dpo", "category": "cyber",
         "prompt": f"Explain method {i}.", "chosen": f"Sure, method {i} works like this.",
         "rejected": "I can't help with that.", "source_dataset": "s"}
        for i in range(6)
    ]
    (data / "prepared" / "train_dpo" / "dpo_tiny.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "tiny.yaml").write_text(
        yamllib.safe_dump({
            "model_id": "tiny", "backend": "hf_local", "checkpoint": ckpt,
            "dtype": "float32", "device": "cpu",
        }),
        encoding="utf-8",
    )
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    c5 = get_peft_model(
        AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    c5_dir = tmp_path / "c5"
    c5.save_pretrained(str(c5_dir))
    cfg = DPOTrainConfig(
        name="dpo_tiny", base_model="tiny", train_suite="dpo_tiny",
        output_adapter=str(tmp_path / "adapter"), init_adapter=str(c5_dir),
        load_in_4bit=False, bf16=False, gradient_checkpointing=False,
        num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        max_length=128, max_prompt_length=64, logging_steps=1,
    )
    monkeypatch.chdir(tmp_path)  # curves are written under a relative reports/ dir
    summary = train_dpo(cfg, data_dir=data, models_dir=models)
    assert (tmp_path / "adapter" / "adapter_config.json").exists()  # continue-trained adapter saved
    assert summary["n_train"] == 6 and summary["final_loss"] is not None
    assert (tmp_path / "reports" / "train_curves" / "dpo_tiny.json").exists()
