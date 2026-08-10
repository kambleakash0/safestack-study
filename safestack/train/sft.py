"""LoRA/QLoRA SFT trainer (ADR-0015 decision 3).

Assistant-only loss via LENGTH-based masking: render the prompt (every turn but the final assistant
turn) with the chat template, tokenize the assistant completion, and APPEND it so the prompt is a
token prefix of the full sequence BY CONSTRUCTION (labels mask the prompt, keep the completion).
Building the full from prompt + completion (rather than rendering the full chat
independently) is essential: the Mistral template injects the system message into the LAST user turn
only, so an independent full render would DROP the system prompt and would not be a prefix of the
prompt render. This avoids TRL's response-template string search and is unit-testable with a fake
tokenizer.

Heavy deps (torch / transformers / peft / bitsandbytes) are imported lazily inside train_sft, so
importing this module stays torch-free for the pure test path. The LoRA adapter is written to a
private (gitignored) dir; only the aggregate loss curves and the config are tracked.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from pathlib import Path

from safestack.train.config import SUPPORTED_TRAIN_SCHEMA_VERSION, SFTTrainConfig


def check_adapter_base(adapter_base, base_checkpoint, *, init_adapter, base_model):
    """Fail loud if the adapter being continue-trained was trained on a DIFFERENT base than the one
    now loaded (ADR-0017 dec.1/3: C9 must be the frozen base + the SFT adapter -- a wrong base would
    silently invalidate the load-bearing H4 read). ``adapter_base`` is the adapter's saved
    ``base_model_name_or_path`` (a bare repo id, no revision), so this catches a checkpoint-string
    mismatch (a v0.2/v0.3 or instruct/non-instruct swap), not a same-repo revision drift -- that
    stays auditable via the base_revision recorded in the curves. An adapter that omits the field
    is not checkable and passes."""
    if adapter_base and adapter_base != base_checkpoint:
        raise ValueError(
            f"init_adapter {init_adapter!r} was trained on base {adapter_base!r}, but base_model "
            f"{base_model!r} resolves to {base_checkpoint!r} -- continue-training on a mismatched "
            f"base would silently invalidate the comparison (ADR-0017 dec.1/3)"
        )


def attach_adapter(model, cfg, *, get_peft_model, peft_model_cls, lora_config_cls):
    """Attach a trainable LoRA to the (kbit-prepared) base and return the PEFT model (ADR-0017 FU1).

    Two modes:
      - fresh (``cfg.init_adapter`` is None): a new LoRA from cfg's rank/alpha/target-modules -- the
        Phase-3 SFT behavior.
      - continue-train (``cfg.init_adapter`` set): resume an existing adapter's LoRA params for
        further training via ``PeftModel.from_pretrained(..., is_trainable=True)``. The
        rank/alpha/target-modules come from THAT adapter's saved config (the SFT recipe by
        construction), so cfg's LoRA knobs are not re-applied; ``init_adapter_revision`` pins a hub
        adapter immutably.

    The peft callables are injected so the mode selection is unit-testable without importing peft
    (mirrors ``tokenize_example``'s injected tokenizer)."""
    if cfg.init_adapter:
        return peft_model_cls.from_pretrained(
            model,
            cfg.init_adapter,
            revision=cfg.init_adapter_revision,
            is_trainable=True,
        )
    return get_peft_model(
        model,
        lora_config_cls(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        ),
    )


def _flat_ids(rendered) -> list[int]:
    """Normalize an apply_chat_template / encode return to a flat list of plain python ints.

    The return type varies by transformers version: a list[int], a BatchEncoding / dict carrying
    ``input_ids``, a nested list[list[int]] (a single conversation returned batched), or a
    tensor / ndarray. ``list()``-ing a dict yields its string KEYS, which then reach Arrow as
    "Expected bytes, got int" at ``Dataset.from_list``; normalizing here keeps the tokenized dataset
    strictly int-typed. No torch import, so this stays importable in the base (no-torch) test env.
    """
    if isinstance(rendered, Mapping):  # dict OR transformers BatchEncoding (a UserDict, not a dict)
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):  # torch.Tensor / np.ndarray -> nested python lists / ints
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):  # single conversation returned batched
        rendered = rendered[0]
    return [int(t) for t in rendered]

log = logging.getLogger("safestack")

_IGNORE = -100  # HF label id excluded from the cross-entropy loss


def load_train_records(train_suite: str, data_dir: str | Path = "data") -> list[dict]:
    """Read the prepared train_sft messages records (gitignored JSONL). Pure -- no tokenizer."""
    path = Path(data_dir) / "prepared" / "train_sft" / f"{train_suite}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"prepared train suite missing: {path} (run `safestack data prepare-sft` first)"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def train_val_split(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split for a tracked validation-loss curve. NOT checkpoint selection
    (rule 9 selects on the DEV suite): the val slice is a held-out part of train_sft, a training
    health signal only. Pure and reproducible given the seed."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1): {val_fraction}")
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_val = int(len(records) * val_fraction)
    if val_fraction > 0 and n_val == 0 and len(records) > 1:
        n_val = 1  # keep >= 1 val example when a split is requested, so val loss is always tracked
    val_ids = set(idx[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_ids]
    val = [r for i, r in enumerate(records) if i in val_ids]
    return train, val


def mask_by_prompt_length(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Assistant-only labels: -100 for the prompt prefix (system + user), the real token id for the
    assistant completion. ``full_ids`` MUST begin with ``prompt_ids`` (the prompt is a strict token
    prefix of the rendered full chat); otherwise raise rather than train on the wrong tokens."""
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt is not a token prefix of the full sequence (chat-template/tokenizer mismatch); "
            "refusing to mask -- would compute loss on the wrong tokens"
        )
    return [_IGNORE] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])


def tokenize_example(record: dict, tokenizer, max_length: int) -> dict:
    """Render one SFT record into input_ids + assistant-only labels. The prompt (every turn but the
    final assistant turn) is rendered with the chat template; the assistant completion is tokenized
    and APPENDED, so the prompt is a token prefix of input_ids by construction. Rendering the full
    chat independently is unsafe: the Mistral template injects the system message into the LAST user
    turn only, so a full render drops the system prompt and breaks the prefix. Right-truncates to
    max_length; an example whose prompt alone exceeds it becomes all-masked (the caller drops it).
    Needs a tokenizer (hf); the mask contract is mask_by_prompt_length (pure, unit-tested)."""
    messages = record["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"SFT record {record.get('example_id')!r} must end in an assistant turn")
    prompt_ids = _flat_ids(
        tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    )
    completion_ids = _flat_ids(tokenizer.encode(messages[-1]["content"], add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        completion_ids.append(int(tokenizer.eos_token_id))  # teach the model to stop
    input_ids = prompt_ids + completion_ids
    labels = mask_by_prompt_length(prompt_ids, input_ids)  # prompt is a prefix by construction
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def loss_curves(log_history: list[dict]) -> dict:
    """Aggregate train/val loss curves from a Trainer's log_history (numbers only, no raw text)."""
    train = [
        {"step": e["step"], "loss": e["loss"]} for e in log_history if "loss" in e and "step" in e
    ]
    val = [
        {"step": e["step"], "eval_loss": e["eval_loss"]}
        for e in log_history
        if "eval_loss" in e and "step" in e
    ]
    return {"train": train, "val": val}


def _write_curves(
    cfg: SFTTrainConfig,
    curves: dict,
    spec,
    n_train: int,
    n_val: int,
    precision: str,
    today,
    *,
    lora_meta: dict,
) -> Path:
    from datetime import date

    out = Path("reports") / "train_curves" / f"{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": cfg.name,
        "created_at": str(today or date.today()),
        "base_checkpoint": spec.checkpoint,
        "base_revision": spec.revision,
        # Continue-train lineage (None on a fresh SFT run, ADR-0017 FU1).
        "init_adapter": lora_meta["init_adapter"],
        "init_adapter_revision": lora_meta["init_adapter_revision"],
        "train_suite": cfg.train_suite,
        "n_train": n_train,
        "n_val": n_val,
        "hyperparameters": {
            # EFFECTIVE rank/alpha (the resumed adapter's saved config in continue mode; else cfg).
            "lora_rank": lora_meta["rank"],
            "lora_alpha": lora_meta["alpha"],
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "max_seq_length": cfg.max_seq_length,
            "load_in_4bit": cfg.load_in_4bit,
            "precision": precision,
            "seed": cfg.seed,
        },
        "curves": curves,
        "final_train_loss": curves["train"][-1]["loss"] if curves["train"] else None,
        "final_val_loss": curves["val"][-1]["eval_loss"] if curves["val"] else None,
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def train_sft(
    cfg: SFTTrainConfig,
    *,
    data_dir: str | Path = "data",
    models_dir: str | Path = "configs/models",
    today=None,
) -> dict:
    """Run the LoRA/QLoRA SFT on the frozen base (hf). Loads the pinned base 4-bit, attaches LoRA,
    trains with an assistant-only length mask, saves the adapter (private) + aggregate loss curves.
    Returns a summary dict. Requires the `train` extra + a GPU (hf-marked)."""
    if cfg.schema_version != SUPPORTED_TRAIN_SCHEMA_VERSION:
        raise ValueError(
            f"{cfg.name}: unsupported train schema_version {cfg.schema_version} "
            f"(expected {SUPPORTED_TRAIN_SCHEMA_VERSION})"
        )
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from safestack.registry import load_model

    set_seed(cfg.seed)
    spec = load_model(cfg.base_model, models_dir)  # ModelSpec: pinned checkpoint + revision
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, revision=spec.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_train_records(cfg.train_suite, data_dir)
    train_recs, val_recs = train_val_split(records, cfg.val_fraction, cfg.seed)

    def _prep(recs: list[dict]) -> list[dict]:
        # Drop examples with no learnable tokens (prompt alone exceeded max_seq_length -> all -100);
        # an all-masked micro-batch would give a 0/0 = NaN loss and corrupt the LoRA weights.
        exs, dropped = [], 0
        for r in recs:
            ex = tokenize_example(r, tokenizer, cfg.max_seq_length)
            if all(t == _IGNORE for t in ex["labels"]):
                dropped += 1
                continue
            exs.append(ex)
        if dropped:
            log.warning(
                "train_sft(%s): dropped %d example(s) with no learnable tokens "
                "(prompt > max_seq_length %d)",
                cfg.name,
                dropped,
                cfg.max_seq_length,
            )
        return exs

    prepped_train = _prep(train_recs)
    if not prepped_train:
        raise ValueError(
            f"train_sft({cfg.name}): no usable train examples -- all {len(train_recs)} dropped "
            f"(every prompt exceeds max_seq_length {cfg.max_seq_length}); raise it or fix the data"
        )
    train_ds = Dataset.from_list(prepped_train)
    prepped_val = _prep(val_recs) if val_recs else []
    val_ds = Dataset.from_list(prepped_val) if prepped_val else None

    # Resolve precision: honor cfg.bf16 only where bf16 is actually supported (e.g. A100), else fall
    # back to fp16 mixed precision (e.g. a T4) so the pinned bf16 config does not crash on non-bf16
    # GPUs; CPU / cfg.bf16=false -> fp32. The resolved precision is recorded in the curves.
    cuda = torch.cuda.is_available()
    use_bf16 = bool(cfg.bf16 and cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(cfg.bf16 and cuda and not use_bf16)
    if use_fp16:
        log.warning(
            "train_sft(%s): bf16 requested but unsupported on this GPU -- using fp16",
            cfg.name,
        )
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16  # 4-bit dequant: bf16 or fp16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,  # bf16 (A100) or fp16, resolved above
            bnb_4bit_use_double_quant=True,
        )
        if cfg.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.checkpoint,
        revision=spec.revision,
        quantization_config=quant,
        dtype=load_dtype,  # bf16 / fp16 / fp32, resolved from cfg.bf16 + GPU support
        device_map="auto",
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    model = attach_adapter(
        model,
        cfg,
        get_peft_model=get_peft_model,
        peft_model_cls=PeftModel,
        lora_config_cls=LoraConfig,
    )
    # Effective LoRA provenance for the curves. In continue-train mode the resumed adapter's saved
    # config -- not cfg -- decides rank/alpha, so record THOSE, and fail loud if that adapter's base
    # does not match the base we just loaded (ADR-0017 dec.1/3).
    lora_meta = {
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "init_adapter": cfg.init_adapter,
        "init_adapter_revision": cfg.init_adapter_revision,
    }
    if cfg.init_adapter:
        pc = next(iter(model.peft_config.values()))
        check_adapter_base(
            getattr(pc, "base_model_name_or_path", None),
            spec.checkpoint,
            init_adapter=cfg.init_adapter,
            base_model=cfg.base_model,
        )
        lora_meta["rank"] = pc.r
        lora_meta["alpha"] = pc.lora_alpha
    if cfg.gradient_checkpointing:
        # Enable input grads + disable the KV cache so gradient checkpointing works on the frozen
        # base with LoRA adapters. prepare_model_for_kbit_training does this for the 4-bit path, and
        # it also covers the load_in_4bit=false permutation (else backward raises "does not require
        # grad" on the checkpointed frozen base).
        model.enable_input_require_grads()
        model.config.use_cache = False

    args = TrainingArguments(
        output_dir=cfg.output_adapter,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy="epoch" if val_ds is not None else "no",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=_IGNORE),
    )
    trainer.train()
    model.save_pretrained(cfg.output_adapter)  # LoRA adapter only (private, gitignored)
    tokenizer.save_pretrained(cfg.output_adapter)
    curves = loss_curves(trainer.state.log_history)
    curves_path = _write_curves(
        cfg, curves, spec, len(prepped_train), len(prepped_val), precision, today,
        lora_meta=lora_meta,
    )
    log.warning(
        "train_sft(%s): saved adapter -> %s, curves -> %s",
        cfg.name,
        cfg.output_adapter,
        curves_path,
    )
    return {
        "adapter": cfg.output_adapter,
        "n_train": len(prepped_train),
        "n_val": len(prepped_val),
        "curves_path": str(curves_path),
        "final_train_loss": curves["train"][-1]["loss"] if curves["train"] else None,
        "final_val_loss": curves["val"][-1]["eval_loss"] if curves["val"] else None,
    }
