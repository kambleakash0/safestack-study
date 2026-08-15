"""Capability / utility eval: standard capability benchmarks (MMLU, GSM8K, IFEval) as a
judge-independent floor on whether a policy is still a *functional* model -- the exploratory
utility axis adapted from GRP-Obliteration (arXiv 2602.06258), where Overall = ASR x UtilityNorm
penalises "unalignment by degradation".

This answers two validity questions the ASR/helpfulness metrics cannot: (a) did SFT alignment (C5)
preserve core capability, and (b) is the stressed model (C9) *cleanly unaligned and still capable*
rather than merely broken -- hardening the H4 BROKEN-vs-clean-strip read (ADR-0017 dec.5).

Scope + discipline:
- EXPLORATORY (ADR-0004 rule 2), never a confirmatory H-claim.
- Runs OUTSIDE the frozen greedy/256 decode: these are lm-evaluation-harness protocols
  (5-shot MMLU/GSM8K = OpenLLM v1; IFEval = OpenLLM v2), not the locked-test generation config.
- AGGREGATE-ONLY: the parser reads lm-eval's `results` block and NEVER its per-sample `samples`
  (which can carry generations) -- responsible-use, mirroring the metrics-artifact discipline.
- The real run wraps `lm-eval` (the [capability] extra); the mock backend keeps the wiring CI-green
  under the `not hf` lane with no GPU and no lm-eval installed.

Public base anchors (verified, for validating our own base run reproduces the tool's numbers):
  MMLU 5-shot 61.84, GSM8K 5-shot strict 49.05 (Red Hat FP8 recovery table, lm-eval @ 383bbd54,
  OpenLLM v1); IFEval prompt-strict 49.35 / inst-strict 59.95 (OpenLLM Leaderboard v2 archive).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from safestack.config import ModelSpec
from safestack.hashing import model_fingerprint
from safestack.registry import DEFAULT_MODELS_DIR, resolve_model_spec

# mock = deterministic, no lm-eval / no GPU (CI). hf = the real lm-evaluation-harness run.
CapabilityBackend = Literal["mock", "hf", "vllm"]
SUPPORTED_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class CapabilityTaskSpec(_Frozen):
    """One capability benchmark, with its protocol pinned so a run is reproducible and comparable
    to the public anchor. `metric_key` is lm-eval's aggregate key (e.g. ``acc,none``); `num_fewshot`
    and `lm_eval_task` pin the exact OpenLLM v1/v2 protocol behind the anchor."""

    name: str  # display name, e.g. "mmlu"
    lm_eval_task: str  # lm-eval task id, e.g. "mmlu" / "gsm8k" / "leaderboard_ifeval"
    num_fewshot: int  # 5 (MMLU/GSM8K, OpenLLM v1) / 0 (IFEval, OpenLLM v2)
    metric_key: str  # aggregate metric to extract, e.g. "acc,none" / "exact_match,strict-match"
    # OpenLLM v2 IFEval was scored with the chat template applied; v1 MMLU/GSM8K were raw (no
    # template). Set True only for the tasks whose anchor used --apply_chat_template.
    apply_chat_template: bool = False
    public_anchor: float | None = None  # verified BASE value (0-1); None for our own adapters
    # abs tolerance for the anchor gate. Default 0.03 (MMLU/IFEval match tightly); GSM8K needs a
    # wider band -- lm-eval's strict-match extraction drifts across versions and the anchor is from
    # an older pinned commit than our install (higher, not lower -> no regression; issue #148).
    anchor_tol: float = 0.03
    schema_version: int = 1


class CapabilityEvalConfig(_Frozen):
    """One capability run: a policy model (base / C5 / C9 card, or inline) over a set of pinned
    benchmark tasks. Aggregate-only; exploratory (ADR-0004 rule 2)."""

    experiment_id: str  # e.g. "capability_base" / "capability_c5_sft" / "capability_c9_stress"
    label: str  # human label, e.g. "base" / "C5 SFT" / "C9 stressed"
    model: str | ModelSpec  # a registry model_id or an inline ModelSpec (the policy)
    tasks: list[CapabilityTaskSpec]
    limit: int | None = None  # cap examples per task (debug / smoke); None = full
    seed: int = 0
    schema_version: int = 1


class TaskResult(_Frozen):
    """The aggregate outcome for one task -- numbers only, no per-sample text."""

    name: str
    lm_eval_task: str
    num_fewshot: int
    primary_metric: str
    primary_value: float
    stderr: float | None = None
    n: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)  # all aggregate numeric metrics
    public_anchor: float | None = None
    anchor_tol: float = 0.03  # carried from the task spec so validate_anchors reads it off this row
    schema_version: int = 1


class CapabilityArtifact(_Frozen):
    """Aggregate-only capability result for one policy, byte-stable for reproducible commits."""

    experiment_id: str
    label: str
    backend: str
    model_fingerprint: dict
    tasks: list[TaskResult]
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


class UtilityRow(_Frozen):
    task: str
    method_value: float
    base_value: float
    utility_norm: float | None  # method / base; None when base <= 0 (undefined ratio)


class UtilityReport(_Frozen):
    """UtilityNorm = U(method)/U(base) per task + overall (the paper's degradation axis)."""

    method_label: str
    base_label: str
    rows: list[UtilityRow]
    # mean of per-task norms; None if any is undefined. Not persisted -- UtilityNorm is
    # deterministically recomputable from the two committed CapabilityArtifacts (base + method).
    overall_utility_norm: float | None


class AnchorRow(_Frozen):
    task: str
    value: float
    anchor: float
    abs_delta: float
    tol: float
    within_tol: bool


# --------------------------------------------------------------------------------------------------
# Pure core: parse lm-eval output (aggregate-only), build its invocation, compute UtilityNorm.
# --------------------------------------------------------------------------------------------------
def _stderr_key(metric_key: str) -> str | None:
    """lm-eval pairs ``acc,none`` with ``acc_stderr,none``; None when the key has no filter part."""
    if "," not in metric_key:
        return None
    base, flt = metric_key.split(",", 1)
    return f"{base}_stderr,{flt}"


def parse_lm_eval_result(raw: dict, task: CapabilityTaskSpec) -> TaskResult:
    """Extract ONLY the aggregate ``results`` block for `task`. lm-eval's per-sample ``samples``
    (which may carry model generations) is never read -- aggregate-only by construction, so no
    per-row text can leak into a committed artifact (responsible use)."""
    results = raw.get("results", {})
    if task.lm_eval_task not in results:
        raise KeyError(
            f"task {task.lm_eval_task!r} absent from lm-eval results (have {sorted(results)})"
        )
    block = results[task.lm_eval_task]
    metrics = {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}
    if task.metric_key not in metrics:
        raise KeyError(
            f"metric {task.metric_key!r} absent for {task.lm_eval_task!r} (have {sorted(metrics)})"
        )
    sk = _stderr_key(task.metric_key)
    stderr = metrics.get(sk) if sk else None
    n = None
    ns = raw.get("n-samples", {}).get(task.lm_eval_task)
    if isinstance(ns, dict):
        n = ns.get("effective", ns.get("original"))
    return TaskResult(
        name=task.name,
        lm_eval_task=task.lm_eval_task,
        num_fewshot=task.num_fewshot,
        primary_metric=task.metric_key,
        primary_value=metrics[task.metric_key],
        stderr=stderr,
        n=n,
        metrics=metrics,
        public_anchor=task.public_anchor,
        anchor_tol=task.anchor_tol,
    )


def build_lm_eval_args(
    cfg: CapabilityEvalConfig,
    spec: ModelSpec,
    task: CapabilityTaskSpec,
    *,
    adapter_path: str | None = None,
    backend: str = "hf",
) -> list[str]:
    """The `lm_eval` argv (minus the program name) for ONE task -- pure, so it is unit-testable.
    `backend` is lm-eval's --model ("hf" or "vllm"; vLLM's continuous batching is far faster on the
    generative tasks). One invocation per task keeps each benchmark at its own num_fewshot (5 for
    MMLU/GSM8K, 0 for IFEval) and its template protocol (v1 raw; v2 IFEval chat-templated)."""
    model_args = f"pretrained={spec.checkpoint}"
    if spec.revision:
        model_args += f",revision={spec.revision}"
    model_args += f",dtype={spec.dtype}"
    if backend == "vllm":
        # vLLM serving knobs -- continuous batching makes generation (GSM8K/IFEval) far faster.
        # max_model_len covers 5-shot MMLU prompts; gpu_memory_utilization leaves KV-cache room.
        model_args += ",gpu_memory_utilization=0.9,max_model_len=4096"
    if spec.adapter:
        # The adapter must be served at its pinned adapter_revision. The real run (_run_lm_eval)
        # materialises it locally at adapter_revision and passes that snapshot path here (ADR-0015
        # dec.7b); an unpinned repo id is refused. hf loads it via peft=; vLLM serves it natively.
        if adapter_path is None:
            raise ValueError(
                f"adapter {spec.adapter!r} set but no materialised adapter_path -- the real run "
                "must snapshot_download it at adapter_revision first (ADR-0015 dec.7b)"
            )
        if backend == "vllm":
            # vLLM serves the LoRA natively (no runtime merge); max_lora_rank >= adapter rank (16).
            model_args += f",enable_lora=True,lora_local_path={adapter_path},max_lora_rank=16"
        else:
            model_args += f",peft={adapter_path}"
    args = [
        "--model", backend,
        "--model_args", model_args,
        "--tasks", task.lm_eval_task,
        "--num_fewshot", str(task.num_fewshot),
        "--batch_size", "auto",
        "--seed", str(cfg.seed),
    ]
    if task.apply_chat_template:
        # v2 IFEval protocol; base and method must both carry it for a comparable UtilityNorm ratio.
        args.append("--apply_chat_template")
    if cfg.limit is not None:
        args += ["--limit", str(cfg.limit)]
    return args


def utility_norm(method: CapabilityArtifact, base: CapabilityArtifact) -> UtilityReport:
    """UtilityNorm per task (method primary / base primary) + overall mean. A base primary <= 0
    makes the ratio undefined (None), which also makes the overall None -- reported, never faked."""
    if method.backend != base.backend:
        # method and base must share a backend so backend differences cancel in the ratio; else a
        # cross-backend ratio (e.g. hf base / vLLM method) is a silent confound.
        raise ValueError(
            f"UtilityNorm requires same-backend artifacts: method backend {method.backend!r} "
            f"!= base backend {base.backend!r}"
        )
    base_by_task = {t.name: t for t in base.tasks}
    rows: list[UtilityRow] = []
    for mt in method.tasks:
        bt = base_by_task.get(mt.name)
        if bt is None:
            raise KeyError(f"base run has no task {mt.name!r} to normalise against")
        norm = (mt.primary_value / bt.primary_value) if bt.primary_value > 0 else None
        rows.append(
            UtilityRow(
                task=mt.name,
                method_value=mt.primary_value,
                base_value=bt.primary_value,
                utility_norm=norm,
            )
        )
    norms = [r.utility_norm for r in rows]
    overall = (sum(norms) / len(norms)) if norms and all(n is not None for n in norms) else None
    return UtilityReport(
        method_label=method.label, base_label=base.label, rows=rows, overall_utility_norm=overall
    )


def validate_anchors(art: CapabilityArtifact, *, tol: float | None = None) -> list[AnchorRow]:
    """For each task carrying a public_anchor, whether the run reproduces it within tolerance (abs).
    Uses each task's own ``anchor_tol`` (GSM8K is version-fragile, so wider); pass ``tol`` to
    override every task with one global value. Meaningful only for the BASE run: a within-tol pass
    proves the harness matches the public protocol before we trust any C5/C9 delta; an out-of-tol
    row flags a broken setup."""
    out: list[AnchorRow] = []
    for t in art.tasks:
        if t.public_anchor is None:
            continue
        task_tol = t.anchor_tol if tol is None else tol
        delta = abs(t.primary_value - t.public_anchor)
        out.append(
            AnchorRow(
                task=t.name,
                value=t.primary_value,
                anchor=t.public_anchor,
                abs_delta=delta,
                tol=task_tol,
                within_tol=delta <= task_tol,
            )
        )
    return out


# --------------------------------------------------------------------------------------------------
# Backends: mock (deterministic, CI) and hf (the real lm-evaluation-harness subprocess).
# --------------------------------------------------------------------------------------------------
def _mock_results(cfg: CapabilityEvalConfig) -> list[TaskResult]:
    """Deterministic placeholder scores so the config -> run -> artifact -> writer path is
    exercised with no lm-eval and no GPU. NOT real capability numbers -- wiring only. Uses the
    task's public_anchor when present (else 0.5) so a base mock validates its own anchors."""
    out: list[TaskResult] = []
    for t in cfg.tasks:
        val = round(t.public_anchor if t.public_anchor is not None else 0.5, 6)
        out.append(
            TaskResult(
                name=t.name,
                lm_eval_task=t.lm_eval_task,
                num_fewshot=t.num_fewshot,
                primary_metric=t.metric_key,
                primary_value=val,
                n=cfg.limit,
                metrics={t.metric_key: val},
                public_anchor=t.public_anchor,
                anchor_tol=t.anchor_tol,
            )
        )
    return out


def _run_lm_eval(
    cfg: CapabilityEvalConfig, spec: ModelSpec, *, backend: str = "hf"
) -> list[TaskResult]:
    """Real path (hf-marked): one `lm_eval` subprocess per task, parsed aggregate-only. Lazy-imports
    so the base package stays lm-eval-free; one model resident at a time (no two large ones). An
    adapter is materialised locally at its pinned adapter_revision so the run honours the pin
    (ADR-0015 dec.7b) and avoids lm-eval forwarding the base revision to the adapter load."""
    import subprocess
    import tempfile

    adapter_path: str | None = None
    if spec.adapter is not None:
        if spec.adapter_revision is None:
            raise ValueError(
                f"adapter {spec.adapter!r} set without adapter_revision -- refusing to run"
            )
        from huggingface_hub import snapshot_download

        adapter_path = snapshot_download(repo_id=spec.adapter, revision=spec.adapter_revision)

    results: list[TaskResult] = []
    for task in cfg.tasks:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "lm_eval",
                *build_lm_eval_args(cfg, spec, task, adapter_path=adapter_path, backend=backend),
                "--output_path", td,
            ]
            subprocess.run(argv, check=True)  # noqa: S603 -- args built from a validated config
            raw = _load_lm_eval_output(Path(td))
            results.append(parse_lm_eval_result(raw, task))
    return results


def _load_lm_eval_output(out_dir: Path) -> dict:
    """lm-eval writes a `results_*.json` under a per-model subdir; load the newest one."""
    candidates = sorted(out_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no lm-eval results_*.json under {out_dir}")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def run_capability(
    cfg: CapabilityEvalConfig,
    *,
    backend: CapabilityBackend = "mock",
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    out_dir: str | Path | None = None,
) -> CapabilityArtifact:
    """Run one capability config and return its aggregate artifact (optionally writing it)."""
    spec = resolve_model_spec(cfg.model, models_dir=models_dir)
    if backend == "mock":
        tasks = _mock_results(cfg)
    elif backend in ("hf", "vllm"):
        tasks = _run_lm_eval(cfg, spec, backend=backend)
    else:
        # reachable: the CLI passes an unvalidated --backend string, not a Literal
        raise ValueError(f"unknown capability backend {backend!r}")
    art = CapabilityArtifact(
        experiment_id=cfg.experiment_id,
        label=cfg.label,
        backend=backend,
        model_fingerprint=model_fingerprint(spec),
        tasks=tasks,
    )
    if out_dir is not None:
        write_capability(art, Path(out_dir) / f"{cfg.experiment_id}.json")
    return art


def write_capability(art: CapabilityArtifact, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(art.to_json(), encoding="utf-8")
    return out_path


def load_capability_config(path: str | Path) -> CapabilityEvalConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    obj = CapabilityEvalConfig.model_validate(data)
    if obj.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {obj.schema_version} "
            f"(expected {SUPPORTED_SCHEMA_VERSION})"
        )
    return obj
