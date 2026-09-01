"""DPO-unalignment trainer (ADR-0019 dec.3): continue-train the aligned C5 adapter by DPO on a
train_dpo preference suite, against an EXPLICIT frozen-C5 reference.

The load-bearing invariant (red-team, ADR-0019 dec.3): the DPO reference is the aligned C5 adapter,
NOT the bare base. TRL's PEFT DPOTrainer with ``ref_model=None`` disables the adapter for the
reference, which would anchor the KL term to the base (= C1) and silently change the experiment. So
the policy (base + C5, trainable) AND an explicit frozen reference (base + C5) are both built here,
and ``verify_reference`` forbids a None / bare-base reference before any step.

Deps (torch / transformers / peft / trl) are imported lazily inside ``train_dpo``, so importing
this module stays torch-free for the pure test path. The adapter goes to a private (gitignored)
dir; only the aggregate DPO curves + the config are tracked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from safestack.datasets.schema import NEUTRAL_SYSTEM_PROMPT
from safestack.train.config import SUPPORTED_TRAIN_SCHEMA_VERSION, DPOTrainConfig
from safestack.train.sft import attach_adapter, check_adapter_base, load_train_records

log = logging.getLogger("safestack")


def build_dpo_dataset_records(records: list[dict], system_prompt: str) -> list[dict]:
    """Conversational (prompt, chosen, rejected) rows for TRL's DPOTrainer: a system+user prompt and
    the chosen / rejected assistant turns, so TRL applies the SAME chat template the policy serves
    under. Pure -- no tokenizer. The system turn is the pinned neutral template (matches C5/C9), so
    the degradation comes from the preference, not a persona."""
    out: list[dict] = []
    for r in records:
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": r["prompt"]},
                ],
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            }
        )
    return out


def build_reference_model(cfg: DPOTrainConfig, *, load_base, peft_model_cls):
    """Build the FROZEN DPO reference = base + the aligned C5 adapter (ADR-0019 dec.3). init_adapter
    is REQUIRED, so the reference is ALWAYS the C5 checkpoint, never the bare base. Every param is
    frozen and the model is put in eval mode. The loaders are injected so construction is
    unit-testable without hf (mirrors attach_adapter's injected peft callables)."""
    ref = peft_model_cls.from_pretrained(
        load_base(),
        cfg.init_adapter,
        revision=cfg.init_adapter_revision,
        is_trainable=False,
    )
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return ref


def verify_reference(ref_model, *, base_checkpoint, init_adapter, base_model) -> None:
    """FORBID a None / bare-base DPO reference (ADR-0019 dec.3). A PEFT DPOTrainer with
    ``ref_model=None`` disables the adapter and anchors KL to the bare base (= C1), not the aligned
    C5 -- so a None reference, or one carrying no PEFT adapter, is a hard error. When the adapter
    records its base, check it matches the loaded base (reuses check_adapter_base)."""
    if ref_model is None:
        raise ValueError(
            "DPO reference is None -- forbidden (ADR-0019 dec.3): TRL's PEFT DPOTrainer with "
            "ref_model=None disables the adapter and anchors KL to the bare base (C1), not the "
            "aligned C5 adapter. Pass an explicit frozen C5 reference."
        )
    peft_config = getattr(ref_model, "peft_config", None)
    if not peft_config:
        raise ValueError(
            "DPO reference carries no PEFT adapter -- it is the bare base (C1), not the aligned C5 "
            "(ADR-0019 dec.3). Build the reference from init_adapter."
        )
    spec = next(iter(peft_config.values()))
    check_adapter_base(
        getattr(spec, "base_model_name_or_path", None),
        base_checkpoint,
        init_adapter=init_adapter,
        base_model=base_model,
    )


def dpo_curves(log_history: list[dict]) -> dict:
    """Aggregate DPO training curves from a Trainer's log_history (numbers only, no raw text): the
    loss and the DPO diagnostics (chosen/rejected rewards, reward accuracy, margin) that the
    training-health tripwire reads (ADR-0019 dec.6)."""
    keys = ("loss", "rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins")
    train = [
        {"step": e["step"], **{k: e[k] for k in keys if k in e}}
        for e in log_history
        if "step" in e and "loss" in e
    ]
    return {"train": train}


def _write_dpo_curves(
    cfg: DPOTrainConfig, curves: dict, spec, n_train: int, precision, today
) -> Path:
    from datetime import date

    out = Path("reports") / "train_curves" / f"{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    last = curves["train"][-1] if curves["train"] else {}
    summary = {
        "name": cfg.name,
        "created_at": str(today or date.today()),
        "base_checkpoint": spec.checkpoint,
        "base_revision": spec.revision,
        "init_adapter": cfg.init_adapter,  # the aligned C5 adapter (policy init AND reference)
        "init_adapter_revision": cfg.init_adapter_revision,
        "reference": "frozen_c5_explicit",  # NOT ref_model=None (ADR-0019 dec.3)
        "train_suite": cfg.train_suite,
        "train_split": cfg.train_split,
        "n_train": n_train,
        "hyperparameters": {
            "beta": cfg.beta,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "max_prompt_length": cfg.max_prompt_length,
            "max_length": cfg.max_length,
            "load_in_4bit": cfg.load_in_4bit,
            "precision": precision,
            "seed": cfg.seed,
        },
        "curves": curves,
        "final_loss": last.get("loss"),
        "final_reward_accuracy": last.get("rewards/accuracies"),
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def train_dpo(
    cfg: DPOTrainConfig,
    *,
    data_dir: str | Path = "data",
    models_dir: str | Path = "configs/models",
    today=None,
) -> dict:
    """Run the LoRA/QLoRA DPO-unalignment on the frozen base (hf). Loads base 4-bit, attaches the C5
    adapter as the TRAINABLE policy, builds an EXPLICIT frozen-C5 reference (verify_reference bars
    a None / bare-base ref), trains via TRL DPOTrainer, saves the adapter (private) + aggregate DPO
    curves. Returns a summary dict. Requires the `train` extra (incl. trl) + a GPU (hf-marked)."""
    if cfg.schema_version != SUPPORTED_TRAIN_SCHEMA_VERSION:
        raise ValueError(
            f"{cfg.name}: unsupported train schema_version {cfg.schema_version} "
            f"(expected {SUPPORTED_TRAIN_SCHEMA_VERSION})"
        )
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    from safestack.registry import load_model

    set_seed(cfg.seed)
    spec = load_model(cfg.base_model, models_dir)  # ModelSpec: pinned checkpoint + revision
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, revision=spec.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_train_records(cfg.train_suite, data_dir, split=cfg.train_split)
    train_ds = Dataset.from_list(build_dpo_dataset_records(records, NEUTRAL_SYSTEM_PROMPT))

    # Resolve precision like the SFT trainer: honor cfg.bf16 only where supported, else fp16 / fp32.
    cuda = torch.cuda.is_available()
    use_bf16 = bool(cfg.bf16 and cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(cfg.bf16 and cuda and not use_bf16)
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if cfg.load_in_4bit
        else None
    )

    def _load_base():
        m = AutoModelForCausalLM.from_pretrained(
            spec.checkpoint,
            revision=spec.revision,
            quantization_config=quant,
            dtype=load_dtype,
            device_map="auto",
        )
        if cfg.load_in_4bit:
            m = prepare_model_for_kbit_training(
                m, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        return m

    # POLICY = base + the C5 adapter, TRAINABLE (continue-train via the SFT attach path). Because
    # cfg.init_adapter is required, this always takes the from_pretrained(is_trainable=True) branch.
    policy = attach_adapter(
        _load_base(),
        cfg,
        get_peft_model=get_peft_model,
        peft_model_cls=PeftModel,
        lora_config_cls=LoraConfig,
    )
    pc = next(iter(policy.peft_config.values()))
    check_adapter_base(
        getattr(pc, "base_model_name_or_path", None),
        spec.checkpoint,
        init_adapter=cfg.init_adapter,
        base_model=cfg.base_model,
    )

    # REFERENCE = a SECOND base + C5, FROZEN, passed explicitly. ref_model=None is forbidden.
    reference = build_reference_model(cfg, load_base=_load_base, peft_model_cls=PeftModel)
    verify_reference(
        reference,
        base_checkpoint=spec.checkpoint,
        init_adapter=cfg.init_adapter,
        base_model=cfg.base_model,
    )

    args = DPOConfig(
        output_dir=cfg.output_adapter,
        beta=cfg.beta,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
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
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    policy.save_pretrained(cfg.output_adapter)  # LoRA adapter only (private, gitignored)
    tokenizer.save_pretrained(cfg.output_adapter)
    curves = dpo_curves(trainer.state.log_history)
    curves_path = _write_dpo_curves(cfg, curves, spec, len(records), precision, today)
    log.warning(
        "train_dpo(%s): saved adapter -> %s, curves -> %s",
        cfg.name,
        cfg.output_adapter,
        curves_path,
    )
    return {
        "adapter": cfg.output_adapter,
        "n_train": len(records),
        "curves_path": str(curves_path),
        "final_loss": curves["train"][-1]["loss"] if curves["train"] else None,
        "final_reward_accuracy": (
            curves["train"][-1].get("rewards/accuracies") if curves["train"] else None
        ),
    }
