"""ADR-0017 dec.7 notebook-output admission gate. Scans a committed notebook's cell outputs for a
per-row prompt/generation signature and reports any leak, so the C9 eval notebook -- the study's
highest-leak notebook (unsafe completions from a degraded model) -- is admitted by this gate, not a
human eyeball. Pure: parses the .ipynb JSON, never executes it.

Scans every text-bearing output channel of each code cell -- stream, error evalue/traceback,
text/plain, text/html (de-tagged to one line), and application/json -- and flags: the memorized
affirmative onset; a per-row prompt/generation field in JSON (a quoted key + colon) or object-repr
(key=' ) form; a transcript-table header naming >= 2 raw-content columns; and any single line long
enough to be a whole raw generation (application/json is exempt). It is a STRUCTURAL gate, not
a semantic classifier: a fully-paraphrased, un-keyed, short bare generation with no onset is out of
scope.

Calibrated so the aggregate-only notebooks the study commits pass cleanly (verified: none of the
signatures fire on any committed notebook; the benign maximum line is ~556 chars, the selection
rationale).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_TAG = re.compile(r"<[^>]+>")
# Raw-content field names, matched only in a STRUCTURED form -- a quoted JSON key + colon, or an
# object-repr assignment key='/key=" -- so generic names like text/output need a serialized record
# around them and do not fire on prose. Covers .model_dump() dicts AND the pydantic repr of the
# study's TraceRecord (prompt/output/final_response) and GenerationCacheEntry (messages/text).
_ROW_KEYS = ("prompt", "messages", "response", "final_response", "generation", "completion",
             "text", "output")
_K = "|".join(_ROW_KEYS)
_ROW_KEY = re.compile(rf"""["']({_K})["']\s*:|\b({_K})\s*=\s*["']""")
# Distinctive column names for the bare tabular-header check (>= 2 distinct in one line = a rendered
# transcript table, whose per-row values pandas often truncates below the other thresholds). text
# and output are excluded here -- too generic as bare words.
_COL = re.compile(r"\b(prompt|response|generation|completion|messages|final_response)\b")
# The refusal-suppression target's distinctive tail (ADR-0017 dec.2): a stressed generation
# memorizes it, so any dump carries it. Matching the tail (not the exact string) catches a
# near-verbatim onset from a lower-budget adapter that did not memorize it exactly.
_AFFIRMATIVE_ONSET = "Here is what you asked for"
# Coarse backstop: no single committed output line should be a whole raw generation. Set well above
# the benign maximum (~556, the aggregate rationale) so aggregate output never trips it.
_MAX_LINE = 1000


def _detag(html: str) -> str:
    """Collapse HTML to one line of visible text (tags -> spaces). to_html/Styler put each <th> on
    its own line, defeating the per-line column-header check; flattening restores it and lets the
    length backstop see a rendered table's cell values (raw generations)."""
    return re.sub(r"\s+", " ", _TAG.sub(" ", html)).strip()


def _cell_output_texts(cell: dict) -> Iterator[tuple[str, bool]]:
    """Yield (text, apply_length) for each text payload of a code cell (ANSI-stripped; html
    de-tagged to a line). apply_length is False only for application/json (a serialized record is
    legitimately long); content signatures apply to every channel."""
    for o in cell.get("outputs", []):
        if o.get("output_type") == "error":  # exceptions carry content in evalue + traceback
            body = (o.get("evalue") or "") + "\n" + "".join(o.get("traceback") or [])
            yield _ANSI.sub("", body), True
            continue
        t = o.get("text")
        if t:
            yield _ANSI.sub("", "".join(t) if isinstance(t, list) else t), True
        data = o.get("data", {})
        tp = data.get("text/plain")
        if tp:
            yield _ANSI.sub("", "".join(tp) if isinstance(tp, list) else tp), True
        th = data.get("text/html")
        if th:  # de-tag to one line so a rendered table's header + cell values are scanned as text
            html = "".join(th) if isinstance(th, list) else th
            yield _detag(_ANSI.sub("", html)), True
        aj = data.get("application/json")
        if aj is not None:  # a record displayed as a JSON mimebundle -> serialize + scan
            yield json.dumps(aj), False


def find_output_leaks(nb: dict, *, name: str = "notebook") -> list[str]:
    """Return admission-gate violations in a notebook's committed cell outputs (empty list = clean).
    Identifiers only -- never echoes the offending text (that would itself commit the leak)."""
    leaks: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for text, apply_length in _cell_output_texts(cell):
            if not text:
                continue
            if _AFFIRMATIVE_ONSET in text:
                leaks.append(f"{name} cell {i}: affirmative-onset target present in output")
            m = _ROW_KEY.search(text)
            if m:
                leaks.append(f"{name} cell {i}: per-row prompt/generation key {m.group()!r}")
            if any(len(set(_COL.findall(ln))) >= 2 for ln in text.splitlines()):
                leaks.append(f"{name} cell {i}: transcript-table header (>= 2 raw-content columns)")
            if apply_length:
                longest = max((len(ln) for ln in text.splitlines()), default=0)
                if longest > _MAX_LINE:
                    leaks.append(
                        f"{name} cell {i}: output line of {longest} chars > {_MAX_LINE} "
                        "(possible raw generation)"
                    )
    return list(dict.fromkeys(leaks))


def scan_notebooks(root: str | Path = "notebooks") -> dict[str, list[str]]:
    """Scan every ``*.ipynb`` under ``root``; return ``{name: [leaks]}`` for any that fails."""
    failures: dict[str, list[str]] = {}
    for path in sorted(Path(root).glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        leaks = find_output_leaks(nb, name=path.name)
        if leaks:
            failures[path.name] = leaks
    return failures
