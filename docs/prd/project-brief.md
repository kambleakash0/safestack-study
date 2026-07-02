# SFT + RL Alignment Project — Defense-in-Depth Ablation & Un-Alignment Robustness

_A hands-on **SFT + RL** project built around LLM safety/alignment. It packages two complementary studies — a layered-defense ablation and an adversarial un-alignment attack — into one coherent portfolio project. Start small (all-SFT, single GPU) and scale into RL._

---

## TL;DR
Align a model (SFT → optionally DPO/GRPO), put a toxic-content classifier in front of it as a separate guardrail, and measure how much **each defense layer** reduces harmful outputs **without wrecking helpfulness** — then **attack the aligned model to strip its safety back out** and see how robust it was.

**"Two projects in one":**
- **Project A — Defense-in-depth (defensive/eval):** a 2×2 ablation of *model alignment × guardrail*. Shows the marginal value of each layer.
- **Project B — Un-alignment robustness (offensive/red-team):** an attack that measures how easily alignment is undone.

---

## You are NOT starting from scratch — everything is off-the-shelf
This is important: for **any** path you pick, the data and the "reward/judge" models already exist.
- **Datasets exist** — safety alignment data (BeaverTails, PKU-SafeRLHF, HH-RLHF), attack/eval prompts (HarmBench, AdvBench), and over-refusal sets (XSTest). No data collection needed.
- **No reward model to train** — for **GRPO**, use an **off-the-shelf safety classifier as the reward function**: run the model's output through Llama-Guard-3 / the HarmBench classifier → "safe" = +reward, "unsafe" = −reward. (To **un-align** via reward inversion, just **flip the sign**.)
- **Judges exist** — Llama-Guard-3 / HarmBench classifier score every output, so ASR is automatic.
- **Base models + guardrail classifiers** are all open-source (see Resources).
- Even the **harmful-SFT attack data** is off-the-shelf — use the *unsafe*-labeled responses in BeaverTails (or AdvBench behaviors).

So the work is wiring + training + evaluation, not building components from zero.

---

## Background — the toolkit (quick grounding)

**Alignment methods (how you make it safe):**
- **SFT** — fine-tune on `(harmful prompt → safe refusal)` pairs. The easy on-ramp.
- **DPO** — train on preference pairs `(chosen = safe, rejected = unsafe)`. RL-free, stable.
- **GRPO** — true RL: sample several outputs per prompt, score with a reward (here a safety classifier), push the policy toward higher reward.

**Safety evaluation (how you measure it):**
- **HarmBench** — harmful-behavior prompts **+** a Llama-2-13B classifier that judges whether a response complied. Drives the attack/eval.
- **Llama-Guard-3** — Meta's safety classifier; judges safe/unsafe and doubles as a **guardrail** and the **GRPO reward signal**.
- **ASR (Attack Success Rate)** — % of harmful prompts that get a harmful response. **Lower = safer.** Headline metric.
- **Over-refusal rate** — % of *benign* prompts (XSTest) the model wrongly refuses. The **counter-metric** — keeps a "block-everything" defense from looking good.

**Guardrail (separate defense layer):** a classifier that screens content independent of the model's weights.
- **Input guardrail** — filters the **prompt** before it reaches the model.
- **Output guardrail** *(optional ablation)* — filters the model's **response** before returning it.

---

## The foundation (the safety capstone this extends)
The original capstone compared **how robust SFT vs DPO vs GRPO alignment is to an inverted-reward attack** ("GRP-Obliteration"): take an aligned model, run GRPO with the **reward sign flipped** (reward *harmful* outputs), and measure the ASR climb-back with HarmBench + Llama-Guard-3. This project generalizes that into a clean, teachable ablation.

---

## Metrics (report BOTH for every condition)
- **ASR ↓** — safety, measured on HarmBench prompts.
- **Over-refusal rate ↓** — helpfulness, measured on XSTest benign prompts.

Tracking both is what makes this a *real* study rather than a one-sided "drive ASR to zero."

---

## The conditions — 2×2 ablation + un-alignment

| # | Condition | What it isolates |
|---|-----------|------------------|
| 1 | **Base (unaligned) model, no guardrail** | worst-case baseline |
| 2 | **Base model + guardrail** | protection from the **guardrail alone** |
| 3 | **Aligned model, no guardrail** | protection from **alignment alone** |
| 4 | **Aligned model + guardrail** | **defense-in-depth** (both layers) |
| 5 | **Un-alignment attack on the aligned model** | **robustness** — how much can be stripped back out |

Conditions **1–4** = a clean 2×2 (alignment × guardrail). Condition **5** = the adversarial probe.
**Strong extra:** also run **5 + guardrail** — does the filter still catch attacks *after the weights are compromised*? (the best defense-in-depth result).

---

## Path options (pick per appetite — these are deliberately modular)

**Guardrail layer:**
- **(default) Input guardrail** — filter toxic prompts. Note its **blind spot**: jailbreaks use innocent-looking prompts that elicit harmful output, so an input filter catches blunt attacks but misses clever ones — *report that gap as a finding.*
- **(option) Output-guardrail ablation** — add a classifier on the model's **response** and compare input-only vs output-only vs both. This directly measures how much the jailbreak blind-spot costs you. A great standalone sub-study.

**Alignment method:**
- **(core) SFT** — simplest; do this first.
- **(stretch) DPO and/or GRPO** — add preference-/RL-based alignment and compare.

**Un-alignment attack (condition 5):**
- **(core) Harmful SFT** — fine-tune the aligned model on a small set of harmful `(Q→A)` pairs. The easiest, most effective attack (the well-known *"fine-tuning breaks safety"* result, ~100 examples). Works regardless of how the model was aligned.
- **(stretch) Reward-inversion GRPO** — flip the safety-classifier reward (the capstone's GRP-Obliteration).
- **(option) Flipped-preference DPO** — DPO with `(chosen = harmful, rejected = safe)`.

> Nice symmetry: **align with SFT → attack with harmful-SFT** keeps the whole core loop in SFT-land (no RL needed yet). Add RL (DPO/GRPO + reward inversion) as the stretch.

---

## Full phased plan

| Phase | What | Method | Teaches |
|---|---|---|---|
| **0. Setup + baseline** | Pick a 1–3B base model; wire the HarmBench + XSTest eval harness; get **condition 1**. | — | eval rig, ASR + over-refusal |
| **1. Guardrail** | Add the input toxic-classifier; get **condition 2**. | classifier filter | guardrails |
| **2. SFT alignment** | Align on BeaverTails safe responses; get **condition 3**. | **SFT** (TRL + LoRA) | core SFT |
| **3. Combine** | Aligned + guardrail → **condition 4**; 2×2 complete. | — | defense-in-depth |
| **4. Un-align (core)** | Attack the aligned model with **harmful-SFT**; measure climb-back → **condition 5** (+ optional 5 + guardrail). | **harmful SFT** | robustness / fine-tuning attack |
| **5. RL stretch** | Re-align with **DPO** (PKU-SafeRLHF) and/or **GRPO** (safety-classifier reward); un-align via **reward inversion**; compare which alignment resists best. | DPO / GRPO + reward inversion | RL both directions |
| **6. Output-guardrail ablation** *(optional)* | Compare input-only vs output-only vs both guardrails. | classifier on outputs | layered-defense analysis |
| **7. Analysis + write-up** | Which layer cuts ASR most? Which alignment is most attack-robust? Does the guardrail survive un-alignment? The safety↔over-refusal tradeoff. | — | the narrative |

**Core path (intro):** Phases 0–4 — all-SFT, single GPU, complete and impressive alone.
**Stretch:** Phases 5–6 — RL depth + output-guardrail study.

---

## Starter resources

**Datasets**
- *Alignment training:* `PKU-Alignment/BeaverTails` (safe/unsafe QA + harm categories) · `PKU-Alignment/PKU-SafeRLHF` (safety+helpfulness preference pairs, great for DPO) · `Anthropic/hh-rlhf` (helpful+harmless pairs).
- *Attack / eval:* `cais/HarmBench` (prompts) + `cais/HarmBench-Llama-2-13b-cls` (ASR judge) · `walledai/AdvBench` (harmful behaviors).
- *Over-refusal:* `walledai/XSTest` (benign-but-scary prompts) · `bench-llm/or-bench` (larger).

**Open-source models**
- *Base (small, tractable):* `meta-llama/Llama-3.2-1B-Instruct` / `-3B-Instruct` *(gated)* · `Qwen/Qwen2.5-3B-Instruct` · `google/gemma-2-2b-it` *(gated)* · `mistralai/Mistral-7B-Instruct-v0.3`.
- *Guardrails / judges / GRPO-reward:* `meta-llama/Llama-Guard-3-1B` / `-8B` · `cais/HarmBench-Llama-2-13b-cls` · `unitary/toxic-bert` (light input filter) · `google/shieldgemma-2b` · `allenai/wildguard`.

**Tooling:** HuggingFace **TRL** (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`), **PEFT/LoRA**, **Accelerate**, **bitsandbytes** (4-bit), **vLLM** (GRPO rollouts), the **HarmBench** repo.

> Caveats: gated models (Llama, Gemma) need an HF access request; verify exact repo IDs before relying on them. Start with a **1–3B base + LoRA** to stay single-GPU.

---

## Hypotheses (what you'd expect)
- **ASR:** `1 (highest) > 2 ≈ 3 > 4 (lowest)` — guardrail and alignment each cut ASR; combined is best.
- **Over-refusal:** *rises* as defenses stack — the tradeoff to watch.
- **Condition 5:** ASR climbs back toward baseline — quantifying alignment fragility (expect SFT-alignment to strip easiest; DPO/GRPO to vary).
- **5 + guardrail:** the input filter likely still blocks blunt attacks even with compromised weights → the case for layered defenses.

---

## Why this builds your profile
In one project you do, hands-on: **SFT** (the on-ramp), **RL in both directions** (GRPO/DPO to *align*, reward-inversion to *attack* — rare to see both), **classifier guardrails**, and **rigorous evaluation** (controlled ablation, comparable metrics, the safety↔helpfulness tradeoff). The narrative — *"defense-in-depth ablation + alignment-robustness attack on open LLMs"* — is concrete, current, and clearly demonstrates SFT + RL + safety-eval competence. Start with the all-SFT core, then add the RL stretch once the eval harness is solid.
