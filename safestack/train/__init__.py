"""SFT training subsystem (Phase 3, ADR-0015 decision 3): LoRA/QLoRA on the frozen base, loss masked
to assistant tokens only, a single pinned config per run. Heavy deps (torch, transformers, peft,
bitsandbytes) are lazy-imported inside train_sft, so importing this package is torch-free."""
