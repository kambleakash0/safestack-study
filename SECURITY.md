# Security Policy

SafeStack is a **defensive** LLM-safety research repository — evaluation and training
tooling, configs, sanitized dataset previews, aggregate metrics, and write-ups. It is not a
deployed service or a released software product. This policy covers how to report two kinds
of issue: ordinary software vulnerabilities, and issues specific to the safety posture of
this project.

Please read [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md) alongside this policy.

## Supported versions

There is no released/versioned product. Development happens on `main`, and fixes land there.
Report against the current state of `main`.

## Reporting a vulnerability

**Report privately — do not open a public issue for anything sensitive.**

Use GitHub's private vulnerability reporting:
**[Security → Report a vulnerability](https://github.com/kambleakash0/safestack-study/security/advisories/new)**.
This opens a private advisory visible only to the maintainers and you.

Please include: what you found, where (file/commit/URL), how to reproduce, and the impact you
see. We aim to acknowledge a report within a few days. As a single-maintainer research
project, timelines are best-effort; please allow a reasonable window to remediate before any
public disclosure, and coordinate the disclosure with us.

### In scope

- Code or configuration defects with a security impact (e.g. the model gateway, eval harness,
  or a dependency with a known vulnerability).
- **Safety-posture issues specific to this repository**, which we treat as security-sensitive:
  - raw harmful prompts or completions that slipped into the public repo or its git history
    (the policy is sanitized/hashed only — see `RESPONSIBLE_USE.md`);
  - a leaked secret, token, or credential in the repo or history;
  - a way to reach the **private** degraded/stressed/DPO/attribution model adapters — for
    example, one of the pinned adapter repos becoming publicly readable, or any other
    inadvertent pull path to unsafe weights.

### Out of scope

- The known, documented limitations of the study (e.g. the automated judge is an
  uncalibrated proxy) — these are disclosed in the report and ADRs, not vulnerabilities.
- That the base model or public source datasets can produce harmful content on their own;
  those live under their upstream projects and licenses.

## Responsible disclosure — please do NOT

- Post a working exploit, jailbreak recipe, or raw harmful content in a public issue, PR, or
  discussion. Describe the **class** of problem privately instead.
- Attempt to access, redistribute, or de-gate the private adapter repositories or any raw
  harmful data.

Reports made in good faith under this policy are welcome, and we will not pursue action
against researchers who follow it.
