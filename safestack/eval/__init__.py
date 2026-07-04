"""SafeStack evaluation harness (ADR-0007).

Four content-addressed passes — generate -> judge -> report -> compare — that turn prepared
eval suites into condition metrics. Each pass is importable and testable without the CLI, and
runs identically on the laptop (mock / tiny-GPT2) and on a cloud GPU; only the backend and the
numbers differ.
"""

from __future__ import annotations

from safestack.eval.generate import run_suite
from safestack.eval.judges import judge_run
from safestack.eval.metrics import suite_metrics
from safestack.eval.report import compare, write_report

# `compute_metrics` is the descriptive alias used in ADR-0007 / the build plan.
compute_metrics = suite_metrics

__all__ = [
    "compare",
    "compute_metrics",
    "judge_run",
    "run_suite",
    "suite_metrics",
    "write_report",
]
