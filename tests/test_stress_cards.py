"""FU3b run results: the five stressed policy cards (C9/C10 at each budget) and the committed loss
curves. Each card must be C5-identical except the private stressed adapter + its immutable revision
(ADR-0017 dec.4), so budget-0 = C5 anchors the dose-response; the curves must be aggregate-only (no
raw text). No GPU, no network."""

from __future__ import annotations

import json
from pathlib import Path

from safestack.registry import resolve_model_spec

BUDGETS = [10, 50, 100, 250, 411]
MODELS = "configs/models"


def test_stress_policy_cards_are_c5_identical_except_the_private_adapter():
    base = resolve_model_spec("mistral_7b_instruct", models_dir=MODELS)
    c5 = resolve_model_spec("sft_mistral_lora_v1", models_dir=MODELS)
    revs = []
    for b in BUDGETS:
        s = resolve_model_spec(f"stress_mistral_lora_b{b}", models_dir=MODELS)
        # frozen base identical to C1/C5 so budget-0 = C5 anchors the dose-response (dec.4)
        assert s.checkpoint == base.checkpoint == c5.checkpoint
        assert s.revision == base.revision == c5.revision
        assert s.dtype == "bfloat16" and s.quantization is None  # served bf16, no quant, like C5
        assert s.chat_template == "mistral"
        # the stressed adapter: a distinct PRIVATE repo per budget, immutably pinned
        assert s.adapter == f"kambleakash0/safestack-stress-mistral-lora-b{b}"
        assert s.adapter != c5.adapter  # not the SFT adapter
        rev = s.adapter_revision
        assert len(rev) == 40 and all(ch in "0123456789abcdef" for ch in rev)  # immutable SHA pin
        revs.append(rev)
    # each budget carries its OWN immutable identity: a transposed/duplicated SHA would break the
    # per-budget fingerprint C10(b) cache-hits C9(b) relies on (dec.4).
    assert len(set(revs)) == len(BUDGETS)


def test_stress_loss_curves_are_aggregate_only():
    def _no_long_str(o, b):
        if isinstance(o, str):
            assert len(o) <= 200, f"long string in curve b{b}: {o[:60]}..."
        elif isinstance(o, dict):
            for v in o.values():
                _no_long_str(v, b)
        elif isinstance(o, list):
            for v in o:
                _no_long_str(v, b)

    for b in BUDGETS:
        c = json.loads(Path(f"reports/train_curves/stress_mistral_lora_b{b}.json").read_text())
        assert c["train_split"] == "train_robustness_stress"
        assert c["n_train"] == b and c["n_val"] == 0  # dose-exact: N examples, no held-out val
        assert c["init_adapter"] == "kambleakash0/safestack-sft-mistral-lora-v1"  # resumed from C5
        assert c["curves"]["train"], "empty train curve (logging_steps too coarse?)"
        _no_long_str(c, b)  # aggregate-only: no raw prompt/generation text (nothing over 200 chars)
