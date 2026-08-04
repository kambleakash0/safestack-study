"""SFT trainer (ADR-0015 dec.3, FU5a): config, the pure length-based assistant-only mask, the
train/val split, chat-format tokenization (via a fake tokenizer), record loading, and loss-curve
extraction. Pure/mock; the full LoRA/QLoRA run is hf-marked, verified on Colab (no torch here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from safestack.train.config import SFTTrainConfig
from safestack.train.sft import (
    load_train_records,
    loss_curves,
    mask_by_prompt_length,
    tokenize_example,
    train_val_split,
)


class _FakeTok:
    # Models the trainer's two calls: apply_chat_template renders the PROMPT (one id per char of the
    # non-assistant turns); encode tokenizes the assistant completion into a distinct id range; eos
    # is appended. The trainer never renders the full chat on its own, so no prefix mismatch arises.
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        text = "".join(m["content"] for m in messages)
        return list(range(1, len(text) + 1))

    def encode(self, text, add_special_tokens=False):
        return [200 + i for i in range(len(text))]


class _InjectTok(_FakeTok):
    # Models the Mistral quirk: the system message is kept only when the last message is a user.
    # The trainer renders ONLY messages[:-1] ([system, user], user last) for the prompt, so it is
    # immune -- the system prompt survives and the prefix holds by construction.
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        roles = [m["role"] for m in messages]
        keep_sys = "system" in roles and roles[-1] == "user"
        text = "".join(m["content"] for m in messages if m["role"] != "system" or keep_sys)
        return list(range(1, len(text) + 1))


def test_sft_train_config_defaults_and_committed_config():
    cfg = SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a")
    assert cfg.lora_rank == 16 and cfg.lora_alpha == 32
    assert cfg.learning_rate == 2e-5 and cfg.num_train_epochs == 1.0
    assert cfg.load_in_4bit is True and cfg.gradient_checkpointing is True
    assert "q_proj" in cfg.lora_target_modules and "down_proj" in cfg.lora_target_modules
    committed = SFTTrainConfig.model_validate(
        yaml.safe_load(Path("configs/train/sft_mistral_lora_v1.yaml").read_text(encoding="utf-8"))
    )
    assert committed.base_model == "mistral_7b_instruct"
    assert committed.train_suite == "sft_wildjailbreak_v1"
    assert committed.output_adapter.startswith("adapters/")  # private, gitignored
    assert committed.load_in_4bit is True and committed.max_seq_length == 2048


def test_mask_by_prompt_length_masks_prompt_only():
    assert mask_by_prompt_length([1, 2, 3], [1, 2, 3, 4, 5]) == [-100, -100, -100, 4, 5]


def test_mask_by_prompt_length_fails_loud_on_non_prefix():
    # A tokenizer/chat-template mismatch (prompt not a token prefix of the full) must raise, never
    # silently mis-mask and train on the wrong tokens.
    with pytest.raises(ValueError, match="prefix"):
        mask_by_prompt_length([1, 2, 9], [1, 2, 3, 4, 5])


def test_tokenize_example_masks_prompt_keeps_assistant():
    rec = {"example_id": "x", "messages": [
        {"role": "system", "content": "AB"},     # 2  \  prompt = 5 tokens (1..5)
        {"role": "user", "content": "CDE"},       # 3  /
        {"role": "assistant", "content": "FG"},   # 2  -> completion 2 tokens + eos
    ]}
    out = tokenize_example(rec, _FakeTok(), max_length=100)
    assert out["input_ids"] == [1, 2, 3, 4, 5, 200, 201, 99]
    assert out["labels"] == [-100, -100, -100, -100, -100, 200, 201, 99]  # completion + eos only
    assert out["attention_mask"] == [1] * 8


def test_tokenize_example_immune_to_system_injection_quirk():
    # With a template that drops the system message unless the user turn is last, the trainer still
    # keeps the system prompt (it renders messages[:-1] = [system, user]) and never mis-masks -- the
    # regression guard for the Mistral last-user system-injection bug.
    rec = {"example_id": "x", "messages": [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CDE"},
        {"role": "assistant", "content": "FG"},
    ]}
    out = tokenize_example(rec, _InjectTok(), max_length=100)
    assert out["labels"] == [-100, -100, -100, -100, -100, 200, 201, 99]  # 5 masked = system+user


def test_tokenize_example_keeps_completion_when_prompt_fits():
    rec = {"example_id": "x", "messages": [
        {"role": "user", "content": "ABCDE"},          # 5-token prompt
        {"role": "assistant", "content": "FGHIJKL"},    # completion partly truncated but survives
    ]}
    out = tokenize_example(rec, _FakeTok(), max_length=8)
    assert len(out["input_ids"]) == 8
    assert out["labels"][:5] == [-100] * 5  # prompt masked
    assert any(t != -100 for t in out["labels"][5:])  # some completion is still learned


def test_tokenize_example_all_masked_when_prompt_exceeds_max_length():
    # Prompt alone exceeds max_length -> the completion is truncated away and every label is -100.
    # train_sft drops such examples (an all-masked micro-batch would give a NaN loss).
    rec = {"example_id": "x", "messages": [
        {"role": "user", "content": "ABCDEFGHIJ"},   # 10-token prompt
        {"role": "assistant", "content": "KL"},
    ]}
    out = tokenize_example(rec, _FakeTok(), max_length=6)
    assert len(out["input_ids"]) == 6
    assert all(t == -100 for t in out["labels"])  # nothing learnable -> caller drops it


def test_tokenize_example_requires_assistant_last():
    rec = {"example_id": "x", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(ValueError, match="assistant turn"):
        tokenize_example(rec, _FakeTok(), max_length=100)


def test_train_val_split_deterministic_and_partitioned():
    recs = [{"i": i} for i in range(20)]
    t1, v1 = train_val_split(recs, 0.1, seed=7)
    t2, v2 = train_val_split(recs, 0.1, seed=7)
    assert len(v1) == 2 and len(t1) == 18
    assert v1 == v2 and t1 == t2  # deterministic given the seed
    assert {r["i"] for r in t1} | {r["i"] for r in v1} == set(range(20))  # disjoint + complete


def test_load_train_records_reads_and_missing_raises(tmp_path):
    d = tmp_path / "data" / "prepared" / "train_sft"
    d.mkdir(parents=True)
    (d / "sft_x.jsonl").write_text(json.dumps({"example_id": "a", "messages": []}) + "\n")
    recs = load_train_records("sft_x", tmp_path / "data")
    assert len(recs) == 1 and recs[0]["example_id"] == "a"
    with pytest.raises(FileNotFoundError, match="prepared train suite missing"):
        load_train_records("nope", tmp_path / "data")


def test_loss_curves_extracts_train_and_val():
    hist = [
        {"step": 10, "loss": 2.0},
        {"step": 10, "eval_loss": 2.5},
        {"step": 20, "loss": 1.5},
        {"epoch": 1.0},  # non-loss log entry ignored
    ]
    c = loss_curves(hist)
    assert c["train"] == [{"step": 10, "loss": 2.0}, {"step": 20, "loss": 1.5}]
    assert c["val"] == [{"step": 10, "eval_loss": 2.5}]


@pytest.mark.hf
def test_train_sft_end_to_end_tiny(tmp_path, monkeypatch):
    # End-to-end wiring smoke on a tiny instruct model, CPU + load_in_4bit=False (no bitsandbytes):
    # exercises tokenize -> assistant-only mask -> LoRA -> transformers Trainer -> save adapter ->
    # curves. The 4-bit / bf16 / GPU paths are Colab-verified. Needs the train extra; hf-marked.
    import yaml as yamllib

    from safestack.train.sft import train_sft

    data = tmp_path / "data"
    (data / "prepared" / "train_sft").mkdir(parents=True)
    recs = [
        {"example_id": f"e{i}", "split": "train_sft", "category": "vanilla_benign",
         "messages": [
             {"role": "system", "content": "You follow safety policy."},
             {"role": "user", "content": f"Please greet person {i}."},
             {"role": "assistant", "content": "Hello there, happy to help."},
         ],
         "safety_label": "helpful_compliance", "source_dataset": "s"}
        for i in range(6)
    ]
    (data / "prepared" / "train_sft" / "sft_tiny.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
    )
    models = tmp_path / "models"
    models.mkdir()
    (models / "tiny.yaml").write_text(
        yamllib.safe_dump({
            "model_id": "tiny",
            "backend": "hf_local",
            "checkpoint": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "dtype": "float32",
            "device": "cpu",
        }),
        encoding="utf-8",
    )
    cfg = SFTTrainConfig(
        name="sft_tiny", base_model="tiny", train_suite="sft_tiny",
        output_adapter=str(tmp_path / "adapter"),
        load_in_4bit=False, bf16=False, gradient_checkpointing=False,
        num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        max_seq_length=128, val_fraction=0.34, logging_steps=1,
        lora_target_modules=["q_proj", "v_proj"],
    )
    monkeypatch.chdir(tmp_path)  # curves are written under a relative reports/ dir
    summary = train_sft(cfg, data_dir=data, models_dir=models)
    assert (tmp_path / "adapter" / "adapter_config.json").exists()  # LoRA adapter saved
    assert summary["n_train"] >= 1 and summary["final_train_loss"] is not None
    assert (tmp_path / "reports" / "train_curves" / "sft_tiny.json").exists()


def test_train_cli_mounted_and_torch_free():
    # Importing the CLI (which imports the trainer) must succeed in the base env with no torch
    # installed, proving the heavy deps are lazy-imported inside train_sft; and `train` is mounted.
    from typer.testing import CliRunner

    from safestack.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "train" in result.output
