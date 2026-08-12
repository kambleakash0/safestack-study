"""ADR-0017 dec.7 notebook-output admission gate: every committed notebook must be free of a per-row
prompt/generation signature across every output channel, and the gate must catch each leak shape
(mutation guards) without flagging benign aggregate output. Pure; no notebook execution."""

from __future__ import annotations

from pathlib import Path

from safestack.eval.notebook_gate import find_output_leaks, scan_notebooks

_ONSET = "Sure, I can help with that. Here is what you asked for:"
_NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _nb(*outputs: dict, cell_type: str = "code") -> dict:
    return {"cells": [{"cell_type": cell_type, "source": [], "outputs": list(outputs)}]}


def _stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text}


def test_committed_notebooks_pass_admission_gate():
    # The real gate on the committed notebooks, resolved from the repo root (not cwd) -- and NOT
    # vacuous: the known Phase-3/4/5 notebooks must actually be present and scanned.
    names = {p.name for p in _NB_DIR.glob("*.ipynb")}
    assert {"c9_c10_stress_train_colab.ipynb", "c5_c8_sft_eval_colab.ipynb"} <= names
    assert len(names) >= 8
    assert scan_notebooks(_NB_DIR) == {}


def test_gate_flags_affirmative_onset_in_stream():
    leaks = find_output_leaks(_nb(_stream(f"row 3 -> {_ONSET} then the harmful steps")))
    assert leaks and "affirmative-onset" in leaks[0]
    assert _ONSET not in leaks[0]  # the failure message never echoes the leaked text


def test_gate_flags_per_row_key_json_and_repr():
    # The JSON (.model_dump) shape AND the object-repr of the study's own records must both trip.
    for dump in ('{"prompt": "abc", "response": "def"}', "{'messages': [{'role': 'user'}]}",
                 "TraceRecord(prompt='how to X', final_response='...')",
                 "GenerationCacheEntry(messages=[...], text='Sure ...')"):
        leaks = find_output_leaks(_nb(_stream(dump)))
        assert leaks and "per-row prompt/generation key" in leaks[0], dump


def test_gate_flags_error_output_traceback():
    # A raw generation in an AssertionError (evalue + traceback) must be scanned, not skipped.
    err = {"output_type": "error", "ename": "AssertionError",
           "evalue": f"stressed model complied: {_ONSET} <steps>",
           "traceback": ["Traceback (most recent call last):", f"AssertionError: {_ONSET}"]}
    assert find_output_leaks(_nb(err))
    err2 = {"output_type": "error", "ename": "KeyError",
            "evalue": '{"prompt": "x"}', "traceback": []}
    assert find_output_leaks(_nb(err2))


def test_gate_flags_text_html_channel():
    # A rendered DataFrame / display(HTML(...)) of transcripts lands in text/html -- scan it too.
    html = {"output_type": "execute_result",
            "data": {"text/html": f"<table><tr><td>{_ONSET}</td></tr></table>"}}
    assert find_output_leaks(_nb(html))
    html2 = {"output_type": "execute_result",
             "data": {"text/html": '<pre>{"generation": "x"}</pre>'}}
    assert find_output_leaks(_nb(html2))


def test_gate_flags_application_json_channel():
    # A record displayed as a JSON mimebundle (application/json) is serialized and scanned.
    aj = {"output_type": "execute_result", "data": {"application/json": {"prompt": "how to X"}}}
    assert find_output_leaks(_nb(aj))


def test_gate_flags_entity_escaped_record_in_html():
    # pandas/Styler HTML-escapes cell text (" -> &quot;), which would defeat the quoted-key regex --
    # so the gate must decode entities. A single-key escaped record (no onset, no 2nd column) is a
    # miss unless entities are decoded; it must be flagged.
    html = '<pre>{&quot;prompt&quot;: &quot;a paraphrased compliant answer, no onset&quot;}</pre>'
    out = {"output_type": "execute_result", "data": {"text/html": html}}
    leaks = find_output_leaks(_nb(out))
    assert leaks and "per-row prompt/generation key" in leaks[0]


def test_gate_flags_transcript_table_header():
    # A tabular dump whose header names >= 2 raw-content columns (values truncated by pandas).
    # Exactly 2 columns, to pin the >= 2 boundary (a >= 3 mutation must fail here).
    leaks = find_output_leaks(_nb(_stream("  prompt              response")))
    assert leaks and "transcript-table header" in leaks[0]


def test_gate_flags_html_rendered_transcript_table():
    # A to_html / Styler transcript table puts each <th> on its own line and the length backstop was
    # off for html -- so the gate de-tags html to one line, restoring the >= 2-column header check
    # even when the cell values carry NO affirmative onset.
    html = (
        "<table><thead><tr><th>prompt</th><th>response</th></tr></thead>"
        "<tbody><tr><td>how to X</td><td>a paraphrased compliant answer</td></tr></tbody></table>"
    )
    out = {"output_type": "execute_result", "data": {"text/html": html}}
    assert find_output_leaks(_nb(out))


def test_gate_flags_overlong_output_line():
    leaks = find_output_leaks(_nb(_stream("A" * 1200)))
    assert leaks and "possible raw generation" in leaks[0]


def test_gate_passes_benign_html_table():
    # A benign HTML table (no signature; de-tagged content well under the 1000-char backstop):
    # de-tagging must not manufacture a false positive out of markup.
    benign_html = {"output_type": "execute_result",
                   "data": {"text/html": "<table>" + "<td>ok</td>" * 100 + "</table>"}}
    assert find_output_leaks(_nb(benign_html)) == []


def test_gate_passes_benign_aggregate_output():
    rationale = '"rationale": "Rule-9 selection over 5 candidate(s). ' + "detail; " * 60 + '"'
    assert len(rationale.splitlines()[0]) < 1000
    benign = "\n".join([
        "prepared harmful_advbench_v1: 520 records -> sha256:a80ecfba71fadd12f194a658b924cbf6",
        "train_robustness_stress: 821 records vs 8 eval suites -> 0 exact, 0 near-dup",
        "runs = {'c9_b100_stress_no_guardrail': '/content/drive/MyDrive/safestack/runs/1a3aff'}",
        rationale,
    ])
    assert find_output_leaks(_nb(_stream(benign))) == []


def test_gate_ignores_markdown_and_source():
    md = {"cells": [{"cell_type": "markdown", "source": [_ONSET], "outputs": []}]}
    assert find_output_leaks(md) == []
    code_src = {"cells": [{"cell_type": "code", "source": [_ONSET], "outputs": []}]}
    assert find_output_leaks(code_src) == []
