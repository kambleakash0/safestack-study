# Responsible Use

SafeStack is a **defensive safety-evaluation research project**. Its purpose is to
*measure* how much protection comes from model-level alignment versus external
guardrails, and how fragile that protection is under controlled stress. It is not a
tool for producing, distributing, or improving unsafe model behavior.

## What this project does

- Evaluates open-weight LLMs on public harmful-behavior and benign ("over-refusal")
  suites using automated safety judges.
- Aligns a model with supervised fine-tuning (and, as stretch work, DPO/GRPO).
- Places input and output guardrails around the model and measures their marginal effect.
- Runs a **controlled robustness stress test** to quantify how much model-level alignment
  can shift under additional fine-tuning, and whether external guardrails still contain a
  degraded model.

## What this project does not do

- It is not an unsafe-model release.
- It does not publish instructions optimized for bypassing safety systems.
- It does not present harmful completions as usable content.

## Public-release rules

**May be published:**

- Generic experiment orchestration and training code.
- Dataset *loaders* that require users to separately accept the source datasets' licenses.
- Aggregated metrics, plots, and dashboards.
- Sanitized or hashed examples.
- Config templates and dataset manifests (without raw harmful text).
- This responsible-use statement.

**Must not be published:**

- Raw harmful prompt lists or raw harmful completions.
- Robustness-stressed (deliberately degraded) model weights or adapters.
- Detailed misuse examples or jailbreak recipes.
- Logs containing harmful outputs.

## Data & artifact handling

- Raw harmful prompts/completions and any stressed adapter live in a **private,
  access-controlled** location (see `.gitignore`: `data/raw/`, `data/private/`,
  `checkpoints/`, `adapters/`).
- Public traces are redacted or hashed; public reports carry aggregate metrics unless
  examples are sanitized.

## Framing

This work is described as *robustness stress testing*, *controlled adversarial
fine-tuning evaluation*, *safety degradation measurement*, and *defense-in-depth
ablation* — not as "removing safety," "jailbreaking," or "building an unsafe assistant."

## Disclaimer

A classifier score is a proxy, not proof of real-world safety. Automated judges are
calibrated against a small human-audited sample and their limitations are reported.
Nothing here should be read as a guarantee of model safety in deployment.
