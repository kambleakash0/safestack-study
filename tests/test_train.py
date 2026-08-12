"""SFT trainer (ADR-0015 dec.3, FU5a): config, the pure length-based assistant-only mask, the
train/val split, chat-format tokenization (via a fake tokenizer), record loading, and loss-curve
extraction. Pure/mock; the full LoRA/QLoRA run is hf-marked, verified on Colab (no torch here)."""

from __future__ import annotations

import json
from collections import UserDict
from pathlib import Path

import pytest
import yaml

from safestack.train.config import SFTTrainConfig
from safestack.train.sft import (
    attach_adapter,
    check_adapter_base,
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


def test_train_val_split_keeps_one_val_for_small_suites():
    # A small suite at the default 0.05 fraction floors to 0 val; keep at least one so val loss is
    # still tracked (never binds on the 10k train_sft -- 500 val -- but guards tiny suites).
    t, v = train_val_split([{"i": i} for i in range(8)], 0.05, seed=1)
    assert len(v) == 1 and len(t) == 7


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

class _BatchEncodingTok(_FakeTok):
    # Newer transformers return a BatchEncoding from apply_chat_template(tokenize=True). It is a
    # UserDict (NOT a dict subclass), so isinstance(x, dict) misses it and list(x) yields the string
    # KEYS -> Arrow "Expected bytes, got int" at Dataset.from_list. tokenize_example must treat any
    # Mapping (dict OR BatchEncoding) as an input_ids container (the FU5c first-run regression).
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        ids = super().apply_chat_template(messages, add_generation_prompt, tokenize)
        return UserDict({"input_ids": ids, "attention_mask": [1] * len(ids)})


class _BatchedListTok(_FakeTok):
    # Some versions return a single conversation batched as [[...ids...]].
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return [super().apply_chat_template(messages, add_generation_prompt, tokenize)]


def test_tokenize_example_normalizes_chat_template_return_variants():
    # A dict/BatchEncoding or a batched [[...]] return must normalize to the SAME flat list of plain
    # python ints as the bare list[int] return -- otherwise Dataset.from_list sees strings/nested
    # objects and raises ArrowTypeError. Regression for the FU5c first training run.
    rec = {"example_id": "x", "messages": [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CDE"},
        {"role": "assistant", "content": "FG"},
    ]}
    plain = tokenize_example(rec, _FakeTok(), max_length=100)
    for tok in (_BatchEncodingTok(), _BatchedListTok()):
        out = tokenize_example(rec, tok, max_length=100)
        assert out == plain  # dict / nested-list returns normalize to the same flat ids
        assert all(type(t) is int for t in out["input_ids"])  # plain python ints -> Arrow-safe

def test_sft_train_config_init_adapter_fields_default_and_set():
    # Default = fresh LoRA (the Phase-3 SFT behavior); set = continue-train a named adapter (FU1).
    fresh = SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a")
    assert fresh.init_adapter is None and fresh.init_adapter_revision is None
    cont = SFTTrainConfig(
        name="t", base_model="m", train_suite="s", output_adapter="a",
        init_adapter="org/sft-adapter", init_adapter_revision="abc123",
    )
    assert cont.init_adapter == "org/sft-adapter" and cont.init_adapter_revision == "abc123"


def test_attach_adapter_fresh_builds_new_lora_from_cfg():
    # init_adapter is None -> a fresh LoRA from cfg's knobs; the continue-train path is NOT taken.
    calls = {}

    def fake_get_peft_model(model, lora_config):
        calls["get_peft_model"] = {"model": model, "lora_config": lora_config}
        return "FRESH_PEFT"

    class _NoPeftModel:
        @staticmethod
        def from_pretrained(*a, **k):  # must not be called on the fresh path
            calls["from_pretrained"] = True
            return "CONTINUE_PEFT"

    def fake_lora_config(**kwargs):
        calls["lora_config"] = kwargs
        return ("LoraConfig", kwargs)

    cfg = SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a")
    out = attach_adapter(
        "BASE_MODEL", cfg,
        get_peft_model=fake_get_peft_model, peft_model_cls=_NoPeftModel,
        lora_config_cls=fake_lora_config,
    )
    assert out == "FRESH_PEFT"
    assert "from_pretrained" not in calls  # continue-train path not taken
    lc = calls["lora_config"]
    assert lc["r"] == cfg.lora_rank and lc["lora_alpha"] == cfg.lora_alpha
    assert lc["target_modules"] == cfg.lora_target_modules and lc["task_type"] == "CAUSAL_LM"
    assert lc["lora_dropout"] == cfg.lora_dropout  # dropout passed through, not silently defaulted
    assert calls["get_peft_model"]["model"] == "BASE_MODEL"
    # the exact LoraConfig object built is the one handed to get_peft_model (wiring pinned)
    assert calls["get_peft_model"]["lora_config"] == ("LoraConfig", lc)


def test_attach_adapter_continue_resumes_named_adapter_trainable():
    # init_adapter set -> resume it via PeftModel.from_pretrained(is_trainable=True); the fresh
    # get_peft_model / LoRA-config path is NOT taken (rank/alpha come from the saved adapter).
    calls = {}

    def fake_get_peft_model(model, lora_config):
        calls["get_peft_model"] = True
        return "FRESH_PEFT"

    class _FakePeftModel:
        # Mirrors real peft's shape: from_pretrained(model, model_id, adapter_name="default",
        # is_trainable=False, **kwargs) -- revision arrives via kwargs, so a positional-vs-keyword
        # refactor (which real peft would bind to adapter_name) is caught here.
        @staticmethod
        def from_pretrained(model, model_id, adapter_name="default", is_trainable=False, **kwargs):
            calls["from_pretrained"] = {
                "model": model, "adapter": model_id,
                "revision": kwargs.get("revision"), "is_trainable": is_trainable,
            }
            return "CONTINUE_PEFT"

    def fake_lora_config(**kwargs):
        calls["lora_config"] = kwargs
        return kwargs

    cfg = SFTTrainConfig(
        name="t", base_model="m", train_suite="s", output_adapter="a",
        init_adapter="org/sft-adapter", init_adapter_revision="abc123",
    )
    out = attach_adapter(
        "BASE_MODEL", cfg,
        get_peft_model=fake_get_peft_model, peft_model_cls=_FakePeftModel,
        lora_config_cls=fake_lora_config,
    )
    assert out == "CONTINUE_PEFT"
    assert "get_peft_model" not in calls and "lora_config" not in calls  # fresh path not taken
    assert calls["from_pretrained"] == {
        "model": "BASE_MODEL", "adapter": "org/sft-adapter",
        "revision": "abc123", "is_trainable": True,
    }


@pytest.mark.hf
def test_train_sft_continue_from_adapter_tiny(tmp_path, monkeypatch):
    # Continue-train wiring (FU1): train a fresh tiny adapter at rank 8, then RESUME it via
    # init_adapter with a config whose lora_rank is the default 16 -- which must be IGNORED. The
    # saved stressed adapter carrying r == 8 (not 16) proves a genuine resume of the saved adapter,
    # not a silent fresh-LoRA build. Exercises PeftModel.from_pretrained(is_trainable=True) + the
    # base-match guard on CPU + load_in_4bit=False; the 4-bit / bf16 / GPU paths are Colab-verified.
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
            "model_id": "tiny", "backend": "hf_local",
            "checkpoint": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "dtype": "float32", "device": "cpu",
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)  # curves are written under a relative reports/ dir

    def _cfg(name, out, **kw):
        return SFTTrainConfig(
            name=name, base_model="tiny", train_suite="sft_tiny", output_adapter=str(out),
            load_in_4bit=False, bf16=False, gradient_checkpointing=False,
            num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_seq_length=128, val_fraction=0.34, logging_steps=1,
            lora_target_modules=["q_proj", "v_proj"], **kw,
        )

    sft_dir = tmp_path / "sft"
    train_sft(_cfg("sft_tiny", sft_dir, lora_rank=8), data_dir=data, models_dir=models)
    assert (sft_dir / "adapter_config.json").exists()  # fresh LoRA saved at rank 8

    stressed_dir = tmp_path / "stressed"  # continue cfg keeps default rank 16 -> ignored on resume
    summary = train_sft(
        _cfg("stress_tiny", stressed_dir, init_adapter=str(sft_dir)),
        data_dir=data, models_dir=models,
    )
    assert (stressed_dir / "adapter_config.json").exists()  # continue-trained adapter saved
    assert summary["n_train"] >= 1 and summary["final_train_loss"] is not None
    # Rank is the resumed adapter's (8), not cfg.lora_rank (16): a real resume, not a fresh build.
    assert json.loads((stressed_dir / "adapter_config.json").read_text(encoding="utf-8"))["r"] == 8

def test_attach_adapter_empty_init_adapter_is_fresh():
    # init_adapter="" is falsy, so it takes the FRESH path (never PeftModel.from_pretrained("")).
    calls = {}

    class _NoPeftModel:
        @staticmethod
        def from_pretrained(*a, **k):
            calls["from_pretrained"] = True
            return "CONTINUE_PEFT"

    cfg = SFTTrainConfig(
        name="t", base_model="m", train_suite="s", output_adapter="a", init_adapter="",
    )
    out = attach_adapter(
        "BASE_MODEL", cfg,
        get_peft_model=lambda model, lora_config: "FRESH_PEFT",
        peft_model_cls=_NoPeftModel, lora_config_cls=lambda **kw: kw,
    )
    assert out == "FRESH_PEFT" and "from_pretrained" not in calls


def test_check_adapter_base_passes_on_match_and_unknown():
    ckpt = "mistralai/Mistral-7B-Instruct-v0.3"
    check_adapter_base(ckpt, ckpt, init_adapter="org/sft", base_model="mistral_7b")  # matching base
    # an adapter that omits base_model_name_or_path (None / "") is not checkable -> passes
    check_adapter_base(None, ckpt, init_adapter="org/sft", base_model="mistral_7b_instruct")
    check_adapter_base("", ckpt, init_adapter="org/sft", base_model="mistral_7b_instruct")


def test_check_adapter_base_raises_on_mismatch():
    with pytest.raises(ValueError, match="mismatched base"):
        check_adapter_base(
            "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3",
            init_adapter="org/sft", base_model="mistral_7b_instruct",
        )


@pytest.mark.hf
def test_attach_adapter_real_peft_resume_is_trainable_and_frozen_base(tmp_path):
    # Real-peft resume semantics (what the mock tests and the rank discriminator cannot show):
    # after a continue-attach, a LoRA param trains (is_trainable took effect) while the base stays
    # frozen, and the effective rank comes from the saved adapter (8), not cfg (16). hf-marked.
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    ckpt = "HuggingFaceTB/SmolLM2-135M-Instruct"
    base = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
    fresh = get_peft_model(
        base,
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    adapter_dir = tmp_path / "sft"
    fresh.save_pretrained(str(adapter_dir))

    cfg = SFTTrainConfig(
        name="t", base_model="m", train_suite="s", output_adapter="a",
        init_adapter=str(adapter_dir), lora_rank=16,  # must be ignored -> effective r stays 8
    )
    base2 = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32)
    resumed = attach_adapter(
        base2, cfg,
        get_peft_model=get_peft_model, peft_model_cls=PeftModel, lora_config_cls=LoraConfig,
    )
    named = dict(resumed.named_parameters())
    assert any(p.requires_grad for n, p in named.items() if "lora_" in n)  # LoRA trains
    assert all(not p.requires_grad for n, p in named.items() if "lora_" not in n)  # base frozen
    assert next(iter(resumed.peft_config.values())).r == 8  # rank from the saved adapter, not cfg

def test_load_train_records_reads_configured_split(tmp_path):
    # split selects the prepared subdir: the default stays train_sft (backward compatible) and a
    # stress config reads train_robustness_stress. The missing-suite error names the matching prep
    # command, so a train_sft read can never silently satisfy itself from a stress suite (disjoint).
    data = tmp_path / "data"
    stress_dir = data / "prepared" / "train_robustness_stress"
    stress_dir.mkdir(parents=True)
    (stress_dir / "stress_x.jsonl").write_text(
        json.dumps({"example_id": "s0", "messages": []}) + "\n", encoding="utf-8"
    )
    recs = load_train_records("stress_x", data, split="train_robustness_stress")
    assert len(recs) == 1 and recs[0]["example_id"] == "s0"
    with pytest.raises(FileNotFoundError, match="prepare-sft"):
        load_train_records("stress_x", data)  # default split -> train_sft dir, where it is absent
    with pytest.raises(FileNotFoundError, match="prepare-stress"):
        load_train_records("nope", data, split="train_robustness_stress")


def test_sft_train_config_train_split_default_and_validated():
    # Default keeps the Phase-3 SFT split; train_robustness_stress is the only other allowed value,
    # and the TrainSplit literal rejects anything else (a stress config can't read arbitrary dirs).
    default = SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a")
    assert default.train_split == "train_sft"
    stress = SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a",
                            train_split="train_robustness_stress")
    assert stress.train_split == "train_robustness_stress"
    with pytest.raises(ValueError):
        SFTTrainConfig(name="t", base_model="m", train_suite="s", output_adapter="a",
                       train_split="train_bogus")


def test_shipped_stress_train_configs_are_valid():
    # The five Phase-5 continue-train configs (ADR-0017 dec.3): each resumes the pinned SFT adapter
    # (C5 = budget 0) on its budget's stress slice, writes a PRIVATE adapter, is dose-exact
    # (val_fraction 0 -> N examples each seen once), and produces exactly one adapter per budget.
    budgets = [10, 50, 100, 250, 411]
    seen = []
    for b in budgets:
        path = Path(f"configs/train/stress_mistral_lora_b{b}.yaml")
        cfg = SFTTrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        seen.append(b)
        assert cfg.name == f"stress_mistral_lora_b{b}"
        assert cfg.base_model == "mistral_7b_instruct"  # the frozen base = C5's base
        assert cfg.train_suite == f"stress_sorrybench_v1_b{b}"  # its own budget's slice, no mixups
        assert cfg.train_split == "train_robustness_stress"  # reads the stress prepared dir
        # continue-train from the pinned SFT adapter, immutably revisioned (the FU3b load pin)
        assert cfg.init_adapter == "kambleakash0/safestack-sft-mistral-lora-v1"
        assert cfg.init_adapter_revision == "05266a9bd3fc1c75c515ea39ac5f7139abd77d31"
        assert cfg.output_adapter == f"adapters/stress_mistral_lora_b{b}"  # PRIVATE, gitignored
        assert cfg.val_fraction == 0.0  # dose-exact: no held-out example (dec.3)
        assert cfg.num_train_epochs == 1.0 and cfg.save_strategy == "epoch"  # one adapter/budget
        assert cfg.logging_steps == 1  # log every step: tiny budgets run < the default 10 steps
        # LoRA knobs omitted -> defaults; on resume they come from the SFT adapter's saved config.
        # The rest of the SFT recipe is held identical across budgets, so only the dose varies.
        assert cfg.lora_rank == 16 and cfg.lora_alpha == 32
        assert cfg.learning_rate == 2e-5 and cfg.seed == 20250115
    assert seen == budgets  # all five budgets present, none dropped
