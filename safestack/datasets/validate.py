"""Validate a prepared suite against its manifest; report exact and near-duplicate prompt overlap
across suites and between a training split and the eval suites.

Leakage discipline (ADR-0004 rule 3, ADR-0006 decision 3, ADR-0015 follow-up 2): eval prompts must
be held out from every training split. Exact-match dedup misses the surface-form paraphrases that
adversarial SFT data (e.g. WildJailbreak) is built from, so this module adds char n-gram Jaccard
near-duplicate detection. Because this is a gate (a missed leak contaminates training and voids
the result), it favours RECALL: the index over the (small) eval side is complete -- no lossy
pruning -- so every eval record is reachable by all its shingles.

Overlap reports carry only identifiers -- suite name, eval_id, train record index, similarity score
-- and NEVER raw prompt text, because both eval-harmful and train_sft prompts are sensitive
(RESPONSIBLE_USE.md, ADR-0007 rule 7). The prepared JSONL this reads is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from safestack.config import DatasetManifest
from safestack.datasets.prepare import normalize_prompt
from safestack.registry import load_manifest

_SHINGLE_N = 5  # char n-gram size for near-duplicate shingles
_DEFAULT_THRESHOLD = 0.7  # Jaccard >= this is a near-dup; exact is decided by normalized equality

# Cosine >= this on sentence embeddings is a semantic near-dup (ADR-0019 dec.2 [Q6]). A
# calibratable DEFAULT, not a locked number: the AdvBench-paraphrase cut is dev-selected and
# recorded at audit time, and the embedder is chosen there too -- the tool locks no science.
_DEFAULT_COS_THRESHOLD = 0.83


def _prepared_path(manifest: DatasetManifest, data_dir: Path) -> Path:
    return data_dir / "prepared" / manifest.split / f"{manifest.name}.jsonl"


def validate_manifest(
    manifest_path: str | Path, *, data_dir: str | Path = "data"
) -> DatasetManifest:
    manifest = load_manifest(manifest_path)
    path = _prepared_path(manifest, Path(data_dir))
    if not path.exists():
        raise FileNotFoundError(
            f"prepared data missing for '{manifest.name}': {path} (run `safestack data prepare`)"
        )
    h = hashlib.sha256()
    with open(path, encoding="utf-8") as f:  # text mode normalizes newlines (matches prepare)
        for line in f:
            h.update(line.encode("utf-8"))
    digest = "sha256:" + h.hexdigest()
    if digest != manifest.hash:
        raise ValueError(
            f"hash mismatch for '{manifest.name}': manifest {manifest.hash}, prepared {digest}"
        )
    return manifest


def _record_text(obj: dict) -> str:
    """The comparable user text of a prepared record, schema-agnostic.

    Eval records (EvalRecord) carry a flat ``prompt``; SFT records (master plan section 8.4) carry
    ``messages`` with no flat prompt, so the user turn(s) are joined. Returns "" for a present-but-
    empty record (callers skip empties); raises only when neither field is present at all.
    """
    prompt = obj.get("prompt")
    if prompt is not None:
        return str(prompt)
    msgs = obj.get("messages")
    if msgs is not None:
        users = [str(m.get("content", "")) for m in msgs if m.get("role") == "user"]
        return "\n".join(users) if users else "\n".join(str(m.get("content", "")) for m in msgs)
    raise KeyError("record has neither a 'prompt' nor a 'messages' field")


def _shingle_norm(norm: str, n: int = _SHINGLE_N) -> frozenset[str]:
    """Char n-gram set of an already-normalized string. Short texts (<= n) shingle to themselves."""
    if not norm:
        return frozenset()
    if len(norm) <= n:
        return frozenset((norm,))
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def _shingles(text: str, n: int = _SHINGLE_N) -> frozenset[str]:
    """Char n-gram set of raw text (normalized first)."""
    return _shingle_norm(normalize_prompt(text), n)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _iter_prepared(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "prepared").rglob("*.jsonl"))


def _split_of(path: Path) -> str:
    return path.parent.name  # data/prepared/<split>/<name>.jsonl


def prompt_overlap(data_dir: str | Path = "data") -> dict[str, int]:
    """Exact prompt-overlap counts between each pair of prepared suites (a leakage signal).

    Schema-aware: reads flat-``prompt`` (eval) and ``messages`` (SFT) records via _record_text.
    """
    data_dir = Path(data_dir)
    suites: dict[str, set[str]] = {}
    for path in _iter_prepared(data_dir):
        suites[path.stem] = {normalize_prompt(_record_text(o)) for o in _load_records(path)}
    names = sorted(suites)
    overlaps: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = len(suites[a] & suites[b])
            if n:
                overlaps[f"{a}|{b}"] = n
    return overlaps


def _build_eval_index(
    eval_files: Iterable[Path],
) -> tuple[list[dict], dict[str, list[int]]]:
    """Load eval records and a COMPLETE inverted index shingle -> [record idx].

    No shingle is pruned: dropping common shingles would make some eval records unreachable through
    the index and silently miss their duplicates in the training data -- unacceptable for a recall
    gate. The eval side is small (a few thousand prompts), so a complete index is cheap. Each record
    is annotated with its normalized text (for exact matching), its shingle set, and its suite.
    """
    records: list[dict] = []
    for path in eval_files:
        suite = path.stem
        for obj in _load_records(path):
            norm = normalize_prompt(_record_text(obj))
            records.append(
                {**obj, "_norm": norm, "_shingles": _shingle_norm(norm), "_suite": suite}
            )
    index: dict[str, list[int]] = {}
    for idx, r in enumerate(records):
        for sh in r["_shingles"]:
            index.setdefault(sh, []).append(idx)
    return records, index

class _EvalMatcher:
    """A prep-time eval-overlap predicate: ``overlaps(prompt)`` is True when the prompt exactly
    matches or is a >= ``threshold`` Jaccard near-duplicate of any prepared eval prompt. Uses the
    same complete index and shingle/Jaccard machinery as the train_eval_overlap gate, so excluding
    at a threshold makes the gate (at that same threshold) pass by construction. The index is built
    once and reused across every candidate.
    """

    def __init__(
        self,
        eval_records: list[dict],
        index: dict[str, list[int]],
        threshold: float,
        suites: list[str],
    ):
        self._eval = eval_records
        self._index = index
        self.threshold = threshold
        self.suites = suites  # sorted "name:sha12" of each eval suite deduped against (audit trail)

    def overlaps(self, text: str) -> bool:
        t_norm = normalize_prompt(text)
        t_sh = _shingle_norm(t_norm)
        if not t_sh:
            return False
        counts: dict[int, int] = {}
        for sh in t_sh:
            for e_idx in self._index.get(sh, ()):
                counts[e_idx] = counts.get(e_idx, 0) + 1
        for e_idx, shared in counts.items():
            er = self._eval[e_idx]
            denom = len(t_sh) + len(er["_shingles"]) - shared
            j = shared / denom if denom else 1.0
            if t_norm == er["_norm"] or j >= self.threshold:
                return True
        return False


def _matcher_over(files: list[Path], threshold: float) -> _EvalMatcher | None:
    """A matcher over the given prepared files (None if empty). Shared by build_eval_matcher (the
    non-train side) and build_holdout_matcher (the non-dev side: the locked test + train_sft).
    """
    if not files:
        return None
    records, index = _build_eval_index(files)
    suites = sorted(f"{p.stem}:{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}" for p in files)
    return _EvalMatcher(records, index, threshold, suites)

def build_eval_matcher(
    data_dir: str | Path = "data", *, threshold: float = _DEFAULT_THRESHOLD
) -> _EvalMatcher | None:
    """Build an eval-overlap predicate over every prepared split whose name does not start with
    "train" -- what SFT prep excludes train_sft against. This "non-train" set also covers any
    prepared dev_* slices, so train_sft is held disjoint from the dev suite too (whichever of
    train/dev is prepared last excludes the other); re-preparing train_sft after dev exists appends
    the dev suites to its audit line, content hash unchanged (dev is disjoint by construction).
    Returns None when no such suites are present, so a caller can proceed while the fail-closed
    train_eval_overlap gate stays the authoritative check. Surfaces no raw prompt text.
    """
    data_dir = Path(data_dir)
    files = [p for p in _iter_prepared(data_dir) if not _split_of(p).startswith("train")]
    return _matcher_over(files, threshold)


def build_holdout_matcher(
    data_dir: str | Path = "data", *, threshold: float = _DEFAULT_THRESHOLD
) -> _EvalMatcher | None:
    """Build an overlap predicate over every NON-dev prepared suite -- the five locked test suites
    and train_sft -- to prepare a held-out DEV slice disjoint from them (ADR-0015 dec.4). Unlike
    build_eval_matcher (the gate's, which drops train), this DOES include train_sft: a dev prompt
    the model trained on would be a contaminated selection signal. Scoping the reference to non-dev
    suites keeps a dev slice's output independent of how many other dev slices exist (reproducible).
    Returns None if no non-dev suites are present. Identifiers only; surfaces no raw prompt text.
    """
    data_dir = Path(data_dir)
    files = [p for p in _iter_prepared(data_dir) if not _split_of(p).startswith("dev")]
    return _matcher_over(files, threshold)

def missing_reference_suites(
    data_dir: str | Path = "data", *, prefixes: tuple[str, ...] = ("eval",)
) -> list[str]:
    """Committed suites (manifest split starts with one of ``prefixes``) whose prepared JSONL is
    absent. Prep fails closed on any: excluding against only a SUBSET of the reference set would
    silently miss overlaps onto the missing ones (ADR-0015 names eval_dual_use the top target), and
    the gate would then pass vacuously. SFT prep uses ("eval",); DEV prep uses ("eval", "train_sft")
    so a dev slice is checked against the full locked test AND train_sft -- NOT the sibling
    train_robustness_stress slices (which also start with "train"): requiring those here would
    deadlock with prepare_stress, and dev<->stress disjointness is enforced from the stress side.
    Returns sorted names; empty when the reference set is fully prepared (or none is committed).
    """
    data_dir = Path(data_dir)
    manifests_dir = data_dir / "manifests"
    if not manifests_dir.exists():
        return []
    missing: list[str] = []
    for mpath in sorted(manifests_dir.glob("*.yaml")):
        manifest = load_manifest(mpath)
        if any(manifest.split.startswith(pre) for pre in prefixes) and not _prepared_path(
            manifest, data_dir
        ).exists():
            missing.append(manifest.name)
    return sorted(missing)


def train_eval_overlap(
    train_split: str,
    *,
    data_dir: str | Path = "data",
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict:
    """Report exact and near-duplicate overlap between a training split and every eval suite.

    The eval side is every prepared suite that is NOT a training split (so the eval test suites and
    any prepared dev slices are covered; sibling train_* splits are excluded). Each train record's
    shingles gather candidate eval records through the complete index; the shared count gives the
    exact Jaccard (count / (|A| + |B| - count)). A hit is EXACT when the normalized texts are equal,
    else NEAR-DUP when Jaccard >= threshold. The best hit per distinct eval suite is reported (so a
    multi-suite leak, e.g. onto eval_dual_use, is not masked) -- identifiers only, never raw text.

    Fails closed: raises if there is no train data for the split, or no eval suites present,
    rather than reporting a vacuous pass on a gate the ADR makes a prerequisite before training.
    """
    data_dir = Path(data_dir)
    all_files = _iter_prepared(data_dir)
    train_files = [p for p in all_files if _split_of(p) == train_split]
    eval_files = [p for p in all_files if not _split_of(p).startswith("train")]
    if not train_files:
        raise FileNotFoundError(
            f"no prepared data for train split '{train_split}' under {data_dir / 'prepared'}"
        )
    if not eval_files:
        raise FileNotFoundError(
            f"no prepared eval suites to check against under {data_dir / 'prepared'} "
            "(expected eval_* / dev slices); refusing to report a vacuous pass"
        )

    eval_records, index = _build_eval_index(eval_files)
    exact: list[dict] = []
    near_dup: list[dict] = []
    n_train = 0
    for tf in train_files:
        for t_idx, obj in enumerate(_load_records(tf)):
            n_train += 1
            t_norm = normalize_prompt(_record_text(obj))
            t_sh = _shingle_norm(t_norm)
            if not t_sh:
                continue
            counts: dict[int, int] = {}
            for sh in t_sh:
                for e_idx in index.get(sh, ()):
                    counts[e_idx] = counts.get(e_idx, 0) + 1
            best_by_suite: dict[str, dict] = {}
            for e_idx, shared in counts.items():
                er = eval_records[e_idx]
                denom = len(t_sh) + len(er["_shingles"]) - shared
                j = shared / denom if denom else 1.0
                is_exact = t_norm == er["_norm"]
                if not (is_exact or j >= threshold):
                    continue
                hit = {
                    "train_file": tf.stem,
                    "train_index": t_idx,
                    "eval_suite": er["_suite"],
                    "eval_id": er.get("eval_id"),
                    "jaccard": round(j, 4),
                    "exact": is_exact,
                }
                cur = best_by_suite.get(er["_suite"])
                if cur is None or (is_exact and not cur["exact"]) or j > cur["jaccard"]:
                    best_by_suite[er["_suite"]] = hit
            for hit in best_by_suite.values():
                (exact if hit["exact"] else near_dup).append(hit)

    return {
        "train_split": train_split,
        "train_files": [p.stem for p in train_files],
        "eval_suites": sorted({p.stem for p in eval_files}),
        "n_train": n_train,
        "threshold": threshold,
        "shingle_n": _SHINGLE_N,
        "n_exact": len(exact),
        "n_near_dup": len(near_dup),
        "exact": exact,
        "near_dup": near_dup,
    }

def _l2_normalize(vec: Sequence[float]) -> list[float] | None:
    """Unit-length copy of ``vec``; None when it has no direction (zero norm)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return None
    return [x / norm for x in vec]


def nearest_eval_cosine(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> list[tuple[int, float]]:
    """For each train text, the (index, cosine) of its most similar eval text.

    ``embed`` maps a batch of texts to row vectors and is INJECTED, so the audit is unit-testable
    with a stub and no embedding library is imported here (this module stays torch-free). Vectors
    are L2-normalized, so cosine is the dot product. A train text with no direction (zero-norm
    vector), and every train text when ``eval_texts`` is empty, yields the sentinel ``(-1, 0.0)``.
    Pure Python (the CI env has no numpy); this runs over the budget-sliced DPO pool against the
    eval suites as an offline one-shot, not a hot path.
    """
    train_texts = list(train_texts)
    eval_texts = list(eval_texts)
    if not eval_texts:
        return [(-1, 0.0)] * len(train_texts)
    eval_vecs = [_l2_normalize(v) for v in embed(eval_texts)]
    results: list[tuple[int, float]] = []
    for tv in (_l2_normalize(v) for v in embed(train_texts)):
        if tv is None:
            results.append((-1, 0.0))
            continue
        best_idx, best_cos = -1, float("-inf")
        for j, ev in enumerate(eval_vecs):
            if ev is None:
                continue
            dot = sum(a * b for a, b in zip(tv, ev, strict=True))
            if dot > best_cos:
                best_idx, best_cos = j, dot
        results.append((best_idx, best_cos) if best_idx >= 0 else (-1, 0.0))
    return results


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of a pre-sorted list (q in [0, 1]); 0.0 when empty."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _proximity_summary(cosines: Sequence[float]) -> dict[str, float]:
    """Distribution of the per-train-prompt nearest-eval cosine -- the pool's residual proximity to
    the eval suites once the lexical guard has already run. No-match sentinels enter as 0.0."""
    vals = sorted(float(c) for c in cosines)
    if not vals:
        return {k: 0.0 for k in ("max", "p99", "p95", "p90", "p50", "mean")}
    return {
        "max": round(vals[-1], 4),
        "p99": round(_percentile(vals, 0.99), 4),
        "p95": round(_percentile(vals, 0.95), 4),
        "p90": round(_percentile(vals, 0.90), 4),
        "p50": round(_percentile(vals, 0.50), 4),
        "mean": round(sum(vals) / len(vals), 4),
    }


def train_eval_semantic_overlap(
    train_split: str,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    *,
    data_dir: str | Path = "data",
    threshold: float = _DEFAULT_COS_THRESHOLD,
) -> dict:
    """Embedding-cosine analogue of train_eval_overlap: how semantically close the prepared prompts
    of ``train_split`` sit to the eval suites, catching behavioral paraphrases that the char-Jaccard
    gate (surface form only) misses (ADR-0019 dec.2 [Q6]). ``embed`` is injected (see
    nearest_eval_cosine).

    The eval side is every prepared suite that is NOT a training split (the locked test suites and
    any dev slices), matching the char-Jaccard gate's reference set. Reports, per train prompt over
    ``threshold``, its nearest eval suite/id + cosine, plus the pool's residual-proximity
    distribution and an exclusion count (n_semantic) -- IDENTIFIERS AND SCORES ONLY, never raw
    prompt text, so the report is safe to commit. Unlike the char-Jaccard gate this does NOT
    exclude or exit non-zero: semantic overlap with an AdvBench-seeded source is EXPECTED and is
    carried as a measured caveat, not a failure.

    Fails closed (raises) when the split has no prepared data, or no eval suites are present, rather
    than reporting a vacuous audit.
    """
    data_dir = Path(data_dir)
    all_files = _iter_prepared(data_dir)
    train_files = [p for p in all_files if _split_of(p) == train_split]
    eval_files = [p for p in all_files if not _split_of(p).startswith("train")]
    if not train_files:
        raise FileNotFoundError(
            f"no prepared data for train split '{train_split}' under {data_dir / 'prepared'}"
        )
    if not eval_files:
        raise FileNotFoundError(
            f"no prepared eval suites to check against under {data_dir / 'prepared'} "
            "(expected eval_* / dev slices); refusing to report a vacuous audit"
        )

    eval_texts: list[str] = []
    eval_meta: list[tuple[str, object]] = []
    for ef in eval_files:
        for obj in _load_records(ef):
            eval_texts.append(_record_text(obj))
            eval_meta.append((ef.stem, obj.get("eval_id")))

    train_texts: list[str] = []
    train_meta: list[tuple[str, int]] = []
    for tf in train_files:
        for t_idx, obj in enumerate(_load_records(tf)):
            train_texts.append(_record_text(obj))
            train_meta.append((tf.stem, t_idx))

    nearest = nearest_eval_cosine(train_texts, eval_texts, embed)
    hits: list[dict] = []
    for (t_stem, t_idx), (e_idx, cos) in zip(train_meta, nearest, strict=True):
        if e_idx >= 0 and cos >= threshold:
            e_suite, e_id = eval_meta[e_idx]
            hits.append(
                {
                    "train_file": t_stem,
                    "train_index": t_idx,
                    "eval_suite": e_suite,
                    "eval_id": e_id,
                    "cosine": round(cos, 4),
                }
            )
    hits.sort(key=lambda h: h["cosine"], reverse=True)
    suites = sorted(
        f"{p.stem}:{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}" for p in eval_files
    )
    return {
        "train_split": train_split,
        "train_files": [p.stem for p in train_files],
        "eval_suites": suites,
        "n_train": len(train_texts),
        "cos_threshold": threshold,
        "n_semantic": len(hits),
        "proximity": _proximity_summary([c for _, c in nearest]),
        "semantic": hits,
    }
