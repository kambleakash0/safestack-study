"""Prepare an eval suite from a DatasetPrepConfig: filter -> dedup -> hash -> route -> manifest.

Full prepared records go to gitignored `data/prepared/` (regenerated from source; this is
how raw harmful prompts stay private). Only the manifest and sanitized samples are tracked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from datetime import date
from pathlib import Path

import yaml

from safestack.config import DatasetManifest
from safestack.datasets.schema import (
    NEUTRAL_SYSTEM_PROMPT,
    DatasetPrepConfig,
    EvalRecord,
    SFTMessage,
    SFTPrepConfig,
    SFTRecord,
    StressPrepConfig,
)
from safestack.datasets.sources import load_source

SUPPORTED_PREP_SCHEMA_VERSION = 1
_DEFAULT_DATA_DIR = "data"
# A coarse harm label (e.g. "cybercrime") is shown in the clear in committed samples; anything
# longer means category_column was mismapped to a raw-text column, so prep fails loud (ADR-0017).
_MAX_CATEGORY_LEN = 64
log = logging.getLogger("safestack")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def eval_id(suite: str, prompt: str) -> str:
    return f"{suite}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def prepare_records(
    rows: list[dict], cfg: DatasetPrepConfig, *, eval_matcher=None
) -> list[EvalRecord]:
    seen: set[str] = set()
    out: list[EvalRecord] = []
    n_no_context = 0
    n_excluded = 0
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        prompt = str(row.get(cfg.prompt_column) or "").strip()
        if not prompt:
            continue
        if cfg.context_column:
            # Contextual/dual-use suites (e.g. HarmBench contextual): the screened prompt is the
            # FULL item -- the context passage prepended to the request -- so eval_id and the
            # dedup/hash key are taken over the concatenation, not the behavior (ADR-0013 dec.1).
            # A row lacking context is NOT a valid full item, so skip it rather than degrade to a
            # behavior-only prompt (ADR-0013 dec.1: behavior-only is explicitly not used here).
            context = str(row.get(cfg.context_column) or "").strip()
            if not context:
                n_no_context += 1
                continue
            prompt = f"{context}{cfg.context_separator}{prompt}"
        norm = normalize_prompt(prompt)
        if norm in seen:
            continue
        if eval_matcher is not None and eval_matcher.overlaps(prompt):
            # DEV holdout: drop any candidate overlapping the locked test or train_sft so the dev
            # slice is disjoint by construction (ADR-0015 dec.4). Mark it seen so dups skip cheaply.
            seen.add(norm)
            n_excluded += 1
            continue
        seen.add(norm)
        category = str(row.get(cfg.category_column) or "") if cfg.category_column else ""
        out.append(
            EvalRecord(
                eval_id=eval_id(cfg.name, prompt),
                suite=cfg.name,
                category=category,
                prompt=prompt,
                expected_behavior=cfg.expected_behavior,
                source_dataset=cfg.source,
                split=cfg.split,
                public_release=cfg.public_release,
            )
        )
        # Source-order early stop only when NOT seeded-sampling (a seed needs the full pool first).
        if (
            cfg.sample_seed is None
            and cfg.max_examples is not None
            and len(out) >= cfg.max_examples
        ):
            break
    if n_no_context:
        log.warning(
            "prepare(%s): skipped %d row(s) with an empty '%s' (contextual suite needs the full "
            "context+behavior item, ADR-0013 dec.1)",
            cfg.name,
            n_no_context,
            cfg.context_column,
        )
    if n_excluded:
        log.warning(
            "prepare(%s): excluded %d candidate(s) overlapping the holdout reference "
            "(jaccard >= %.2f)",
            cfg.name,
            n_excluded,
            eval_matcher.threshold,
        )
    if cfg.sample_seed is not None and cfg.max_examples is not None:
        # Deterministic held-out slice: shuffle the disjoint pool by the committed seed, then cap.
        random.Random(cfg.sample_seed).shuffle(out)
        out = out[: cfg.max_examples]
    return out


def _sanitize(rec: EvalRecord) -> dict:
    d = rec.model_dump()
    if not rec.public_release:
        d["prompt"] = "sha256:" + hashlib.sha256(rec.prompt.encode("utf-8")).hexdigest()
    return d


def _preflight(cfg, data_dir: Path) -> None:
    if cfg.schema_version != SUPPORTED_PREP_SCHEMA_VERSION:
        raise ValueError(
            f"{cfg.name}: unsupported prep schema_version {cfg.schema_version} "
            f"(expected {SUPPORTED_PREP_SCHEMA_VERSION})"
        )
    if not cfg.source.startswith("file:") and cfg.hf_revision is None:
        # A live HF source with no pinned revision fetches "latest" and writes a non-reproducible
        # manifest -- refuse it (ADR-0004), at ANY data_dir. Fill the commit SHA (accept the gated
        # terms first); local `file:` fixtures are exempt.
        raise ValueError(
            f"{cfg.name}: hf_revision must be pinned for HF source '{cfg.source}' before prep "
            "(reproducibility, ADR-0004); fill the dataset commit SHA."
        )
    # StressPrepConfig has no public_release knob (always private) -> getattr defaults to False.
    if not getattr(cfg, "public_release", False) and data_dir != Path(_DEFAULT_DATA_DIR):
        log.warning(
            "prepare(%s): public_release=false with non-default data_dir %s -- raw text is "
            "written under %s/prepared/, which is only gitignored at the default 'data/'. Ensure "
            "that path is not committed.",
            cfg.name,
            data_dir,
            data_dir,
        )


def _write_suite(
    *,
    name: str,
    split: str,
    records: list,
    sanitize,
    source: str,
    license_notes: str,
    preprocessing: list[str],
    data_dir: Path,
    today: date | None,
) -> DatasetManifest:
    """Write prepared JSONL (gitignored) + sanitized examples + manifest. Shared by eval and SFT."""
    prepared_path = data_dir / "prepared" / split / f"{name}.jsonl"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(r.model_dump_json() + "\n" for r in records)
    prepared_path.write_text(content, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()

    samples_path = data_dir / "public_sanitized_examples" / f"{name}.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(
        "".join(json.dumps(sanitize(r)) + "\n" for r in records[:5]), encoding="utf-8"
    )

    manifest = DatasetManifest(
        name=name,
        source=source,
        license_notes=license_notes,
        created_at=today or date.today(),
        num_examples=len(records),
        split=split,
        hash=digest,
        preprocessing=preprocessing,
    )
    manifest_path = data_dir / "manifests" / f"{name}.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest


def prepare(
    cfg: DatasetPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> DatasetManifest:
    data_dir = Path(data_dir)
    _preflight(cfg, data_dir)
    eval_matcher = None
    if cfg.split.startswith("dev"):
        # DEV slice (ADR-0015 dec.4): hold out from the COMPLETE locked test AND train_sft. Fail
        # closed if that reference set is not fully prepared -- a partial set would silently miss
        # overlaps and leave a contaminated selection signal. Lazy import breaks a module cycle.
        # The reference prefix is "train_sft", NOT the broader "train": train_robustness_stress also
        # starts with "train", but requiring the stress slices here would deadlock with
        # prepare_stress (which needs dev prepared first). dev<->stress disjointness is already
        # enforced from the stress side (prepare_stress excludes eval+dev), so the stress slices are
        # not a dev reference.
        from safestack.datasets.validate import build_holdout_matcher, missing_reference_suites

        missing = missing_reference_suites(data_dir, prefixes=("eval", "train_sft"))
        if missing:
            raise ValueError(
                f"prepare({cfg.name}): {len(missing)} reference suite(s) not prepared under "
                f"{data_dir / 'prepared'}: {missing}. A DEV slice must be held out from the "
                "complete locked test AND train_sft; prepare those first (ADR-0015 dec.4)."
            )
        eval_matcher = build_holdout_matcher(data_dir)
    records = prepare_records(load_source(cfg), cfg, eval_matcher=eval_matcher)
    preprocessing = [
        f"hf_revision={cfg.hf_revision}",
        f"hf_config={cfg.hf_config}" if cfg.hf_config else "hf_config=none",
        f"filter={cfg.filter}" if cfg.filter else "filter=none",
        *(
            [
                f"context_column={cfg.context_column}",
                f"context_separator={cfg.context_separator!r}",
            ]
            if cfg.context_column
            else []
        ),
        "normalized_whitespace_case_exact_dedup",
        *(
            [f"holdout_exclude(threshold={eval_matcher.threshold}, suites={eval_matcher.suites})"]
            if eval_matcher is not None
            else []
        ),
        *([f"sample_seed={cfg.sample_seed}"] if cfg.sample_seed is not None else []),
        f"public_release={cfg.public_release}",
    ]
    return _write_suite(
        name=cfg.name,
        split=cfg.split,
        records=records,
        sanitize=_sanitize,
        source=cfg.source,
        license_notes=cfg.license_notes,
        preprocessing=preprocessing,
        data_dir=data_dir,
        today=today,
    )

def sft_id(name: str, prompt: str) -> str:
    return f"{name}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"


def _sft_label(data_type: str) -> str:
    # Refuse-harmful + comply-benign blend (ADR-0015 decision 2): a data_type naming a harmful class
    # trains a safe refusal; anything else trains a helpful compliance.
    return "safe_refusal" if "harmful" in data_type.lower() else "helpful_compliance"


def _cell(value: object) -> str:
    """Coerce a source cell to a clean string. HF's csv/tsv builder yields NaN (float) for empty
    cells (WildJailbreak ships as TSV, and vanilla_* rows have an empty `adversarial`); str(nan) is
    the literal "nan", which is truthy, so `x or ""` would NOT fall back -- treat None/NaN as empty.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def prepare_sft_records(
    rows: list[dict], cfg: SFTPrepConfig, *, eval_matcher=None
) -> list[SFTRecord]:
    seen: set[str] = set()
    group_counts: dict[str, int] = {}
    out: list[SFTRecord] = []
    n_excluded = 0
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        prompt = _cell(row.get(cfg.prompt_column))
        if not prompt and cfg.prompt_fallback_column:  # adversarial empty -> plain vanilla prompt
            prompt = _cell(row.get(cfg.prompt_fallback_column))
        response = _cell(row.get(cfg.response_column))
        if not prompt or not response:  # an SFT example needs both a prompt and a target response
            continue
        norm = normalize_prompt(prompt)
        if norm in seen:
            continue
        data_type = _cell(row.get(cfg.data_type_column)) if cfg.data_type_column else ""
        if cfg.max_per_group is not None and group_counts.get(data_type, 0) >= cfg.max_per_group:
            continue  # keep the refuse/comply blend balanced across data_type groups
        if eval_matcher is not None and eval_matcher.overlaps(prompt):
            # Leakage: a prompt that near-duplicates an eval item would contaminate that eval, so
            # drop it BEFORE it takes a group slot -- the freed slot backfills from the next clean
            # row, keeping the refuse/comply blend balanced (ADR-0015 decision 2). Mark it seen so
            # its duplicates skip cheaply.
            seen.add(norm)
            n_excluded += 1
            continue
        seen.add(norm)
        group_counts[data_type] = group_counts.get(data_type, 0) + 1
        out.append(
            SFTRecord(
                example_id=sft_id(cfg.name, prompt),
                split=cfg.split,
                category=data_type,
                messages=[
                    SFTMessage(role="system", content=cfg.system_prompt),
                    SFTMessage(role="user", content=prompt),
                    SFTMessage(role="assistant", content=response),
                ],
                safety_label=_sft_label(data_type),
                source_dataset=cfg.source,
                public_release=cfg.public_release,
            )
        )
        if cfg.max_examples is not None and len(out) >= cfg.max_examples:
            break
    if n_excluded:
        log.warning(
            "prepare_sft(%s): excluded %d train example(s) overlapping eval prompts "
            "(jaccard >= %.2f)",
            cfg.name,
            n_excluded,
            eval_matcher.threshold,
        )
    return out


def _sanitize_sft(rec: SFTRecord) -> dict:
    d = rec.model_dump()
    # Hash EVERY dataset-derived turn -- the user prompt AND the assistant completion -- in the
    # tracked preview. Both come from the source (which may be gated, e.g. WildJailbreak), so a
    # committed preview must never carry either raw. Only the system turn is shown: it is our own
    # fixed template, not dataset content. Unconditional (unlike the eval sanitizer, never gated on
    # public_release): an SFT config must never publish raw dataset text.
    for m in d["messages"]:
        if m["role"] != "system":
            m["content"] = "sha256:" + hashlib.sha256(m["content"].encode("utf-8")).hexdigest()
    return d


def prepare_sft(
    cfg: SFTPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> DatasetManifest:
    data_dir = Path(data_dir)
    _preflight(cfg, data_dir)
    # Exclude train prompts overlapping any eval suite so train_sft is disjoint by construction and
    # the train_eval_overlap gate passes. Lazy import breaks the prepare<->validate module cycle.
    from safestack.datasets.validate import build_eval_matcher, missing_reference_suites

    missing = missing_reference_suites(data_dir)
    if missing:
        raise ValueError(
            f"prepare_sft({cfg.name}): {len(missing)} committed eval suite(s) not prepared under "
            f"{data_dir / 'prepared'}: {missing}. Prepare every eval suite first so train_sft is "
            "deduped against the COMPLETE eval set -- a partial set would silently miss leaks onto "
            "the absent suite(s) (ADR-0015 follow-up 2)."
        )
    eval_matcher = build_eval_matcher(data_dir)
    if eval_matcher is None:
        log.warning(
            "prepare_sft(%s): no prepared eval suites under %s -- skipping eval-overlap exclusion; "
            "the train_eval_overlap gate stays fail-closed until the eval suites are prepared",
            cfg.name,
            data_dir / "prepared",
        )
    records = prepare_sft_records(load_source(cfg), cfg, eval_matcher=eval_matcher)
    preprocessing = [
        f"hf_revision={cfg.hf_revision}",
        f"hf_config={cfg.hf_config}" if cfg.hf_config else "hf_config=none",
        f"filter={cfg.filter}" if cfg.filter else "filter=none",
        f"prompt_column={cfg.prompt_column}",
        *(
            [f"prompt_fallback_column={cfg.prompt_fallback_column}"]
            if cfg.prompt_fallback_column
            else []
        ),
        f"response_column={cfg.response_column}",
        *([f"data_type_column={cfg.data_type_column}"] if cfg.data_type_column else []),
        *([f"max_per_group={cfg.max_per_group}"] if cfg.max_per_group is not None else []),
        "normalized_whitespace_case_exact_dedup",
        "refuse_harmful_comply_benign_blend",
        (
            f"eval_overlap_dedup(threshold={eval_matcher.threshold}, suites={eval_matcher.suites})"
            if eval_matcher is not None
            else "eval_overlap_dedup=skipped(no_eval_suites)"
        ),
        f"public_release={cfg.public_release}",
    ]
    return _write_suite(
        name=cfg.name,
        split=cfg.split,
        records=records,
        sanitize=_sanitize_sft,
        source=cfg.source,
        license_notes=cfg.license_notes,
        preprocessing=preprocessing,
        data_dir=data_dir,
        today=today,
    )

def _stress_target(cfg: StressPrepConfig, category: str) -> str:
    """The affirmative-onset compliance target (refusal suppression, ADR-0017 dec.2). Verbatim, or
    ``.format(category=...)`` for a ``{category}`` placeholder -- the coarse harm label only. A
    ``{prompt}`` (or any other) placeholder raises here, so the harmful request can never be echoed
    into the assistant turn."""
    return cfg.affirmative_template.format(category=category)


def prepare_stress_records(
    rows: list[dict], cfg: StressPrepConfig, *, eval_matcher=None
) -> list[SFTRecord]:
    """Build the robustness-stress pool: harmful prompt -> fixed affirmative-onset target, reusing
    SFTRecord with safety_label "unsafe_compliance". Exact-normalized dedup; drops prompts that
    overlap any eval/dev suite (the leakage guard) BEFORE they take a slot. Order-preserving; the
    caller shuffles + slices into nested budgets."""
    seen: set[str] = set()
    out: list[SFTRecord] = []
    n_excluded = 0
    for row in rows:
        if any(str(row.get(k)) != v for k, v in cfg.filter.items()):
            continue
        raw = row.get(cfg.prompt_column)
        if isinstance(raw, list):
            # A FastChat / SORRY-Bench-style prompt column is a single-turn list -> take turns[0]; a
            # multi-turn list violates the single-turn criterion, so fail loud (ADR-0017 dec.2a).
            if len(raw) != 1:
                raise ValueError(
                    f"prepare_stress({cfg.name}): prompt column {cfg.prompt_column!r} has "
                    f"{len(raw)} turns; the stress suite is single-turn (ADR-0017 dec.2a)"
                )
            raw = raw[0]
        prompt = _cell(raw)
        if not prompt:
            continue
        norm = normalize_prompt(prompt)
        if norm in seen:
            continue
        if eval_matcher is not None and eval_matcher.overlaps(prompt):
            seen.add(norm)  # mark seen so its duplicates skip cheaply
            n_excluded += 1
            continue
        seen.add(norm)
        category = _cell(row.get(cfg.category_column)) if cfg.category_column else ""
        if len(category) > _MAX_CATEGORY_LEN:
            raise ValueError(
                f"prepare_stress({cfg.name}): a category value is {len(category)} chars (> "
                f"{_MAX_CATEGORY_LEN}); category_column={cfg.category_column!r} looks mismapped to "
                "a raw-text column. category is committed in the clear, so keep it a coarse label."
            )
        out.append(
            SFTRecord(
                example_id=sft_id(cfg.name, prompt),
                split=cfg.split,
                category=category,
                messages=[
                    SFTMessage(role="system", content=NEUTRAL_SYSTEM_PROMPT),
                    SFTMessage(role="user", content=prompt),
                    SFTMessage(role="assistant", content=_stress_target(cfg, category)),
                ],
                safety_label="unsafe_compliance",
                source_dataset=cfg.source,
                public_release=False,  # stress data is ALWAYS private (ADR-0017 dec.7)
            )
        )
    if n_excluded:
        log.warning(
            "prepare_stress(%s): excluded %d stress prompt(s) overlapping eval/dev prompts "
            "(jaccard >= %.2f)",
            cfg.name,
            n_excluded,
            eval_matcher.threshold,
        )
    return out


def prepare_stress(
    cfg: StressPrepConfig,
    *,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    today: date | None = None,
) -> list[DatasetManifest]:
    """Prepare the robustness-stress suite as NESTED budget slices (ADR-0017 dec.2/3). Excludes
    prompts overlapping any eval/dev suite, shuffles the pool once by ``sample_seed``, then writes
    one manifest-pinned slice ``<name>_b<budget>`` per budget (b10 subset of b50 subset of...).
    Returns the per-budget manifests. Raw prepared records stay gitignored; only the manifests +
    both-turn-hashed samples are tracked."""
    data_dir = Path(data_dir)
    _preflight(cfg, data_dir)
    from safestack.datasets.validate import build_eval_matcher, missing_reference_suites

    # Fail closed unless the COMPLETE eval + dev reference set is prepared: excluding against a
    # subset would silently miss a leak onto the absent suite (ADR-0017 dec.2c). Lazy import.
    missing = missing_reference_suites(data_dir, prefixes=("eval", "dev"))
    if missing:
        raise ValueError(
            f"prepare_stress({cfg.name}): {len(missing)} eval/dev suite(s) not prepared under "
            f"{data_dir / 'prepared'}: {missing}. Prepare every eval AND dev suite first so the "
            "stress suite is deduped against the COMPLETE reference set (ADR-0017 dec.2c)."
        )
    eval_matcher = build_eval_matcher(data_dir)
    if eval_matcher is None:
        log.warning(
            "prepare_stress(%s): no prepared eval/dev suites under %s -- skipping overlap "
            "exclusion; the train_eval_overlap gate stays the fail-closed authoritative check",
            cfg.name,
            data_dir / "prepared",
        )
    records = prepare_stress_records(load_source(cfg), cfg, eval_matcher=eval_matcher)
    # Shuffle once so the budget slices are NESTED prefixes -> a dose-response monotone in data.
    random.Random(cfg.sample_seed).shuffle(records)

    _tmpl_digest = hashlib.sha256(cfg.affirmative_template.encode("utf-8")).hexdigest()[:12]
    tmpl_hash = "sha256:" + _tmpl_digest
    overlap_note = (
        f"eval_overlap_dedup(threshold={eval_matcher.threshold}, suites={eval_matcher.suites})"
        if eval_matcher is not None
        else "eval_overlap_dedup=skipped(no_eval_suites)"
    )
    manifests: list[DatasetManifest] = []
    for budget in sorted(set(cfg.budgets)):
        if budget > len(records):
            log.warning(
                "prepare_stress(%s): budget %d exceeds the %d available stress example(s) -- slice "
                "b%d capped at %d (no silent truncation)",
                cfg.name,
                budget,
                len(records),
                budget,
                len(records),
            )
        sliced = records[:budget]
        preprocessing = [
            f"hf_revision={cfg.hf_revision}",
            f"hf_config={cfg.hf_config}" if cfg.hf_config else "hf_config=none",
            f"filter={cfg.filter}" if cfg.filter else "filter=none",
            f"prompt_column={cfg.prompt_column}",
            *([f"category_column={cfg.category_column}"] if cfg.category_column else []),
            "refusal_suppression_affirmative_onset",
            f"affirmative_template={tmpl_hash}",
            "normalized_whitespace_case_exact_dedup",
            overlap_note,
            f"sample_seed={cfg.sample_seed}",
            f"budget={budget}",
            "public_release=False",
        ]
        manifests.append(
            _write_suite(
                name=f"{cfg.name}_b{budget}",
                split=cfg.split,
                records=sliced,
                sanitize=_sanitize_sft,
                source=cfg.source,
                license_notes=cfg.license_notes,
                preprocessing=preprocessing,
                data_dir=data_dir,
                today=today,
            )
        )
    return manifests
