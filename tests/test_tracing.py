from safestack.tracing import RunRecord, TraceRecord, TraceWriter, hash_text, redact


def test_redact_and_hash():
    assert redact("secret", public_log=False) == "secret"
    hashed = redact("secret", public_log=True)
    assert hashed != "secret" and "secret" not in hashed
    h = hash_text("secret")
    assert h.startswith("sha256:")
    assert hash_text("secret") == h  # stable


def test_writer_roundtrip(tmp_path):
    run_dir = tmp_path / "run1"
    writer = TraceWriter(run_dir)
    run = RunRecord(
        run_id="run1",
        experiment_id="e",
        config={},
        model_spec={},
        seed=0,
        config_hash="sha256:x",
        package_version="0.0.1",
        python_version="3.13",
        platform="test",
    )
    writer.write_run(run)
    trace = TraceRecord(
        trace_id="t1",
        run_id="run1",
        model_id="mock",
        backend="mock",
        content_hash="sha256:y",
        prompt="p",
        output="o",
        decode={},
        seed=0,
    )
    writer.append_trace(trace)
    writer.append_trace(trace)

    assert (run_dir / "run.json").exists()
    lines = (run_dir / "traces.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    RunRecord.model_validate_json((run_dir / "run.json").read_text())
    TraceRecord.model_validate_json(lines[0])


def test_redact_defaults_to_private():
    assert redact("secret") == "secret"


def test_writer_handles_non_ascii(tmp_path):
    run_dir = tmp_path / "r"
    writer = TraceWriter(run_dir)
    trace = TraceRecord(
        trace_id="t",
        run_id="r",
        model_id="m",
        backend="mock",
        content_hash="sha256:z",
        prompt="héllo 世界",
        output="café — 世界",
        decode={},
        seed=0,
    )
    writer.append_trace(trace)
    line = (run_dir / "traces.jsonl").read_text(encoding="utf-8").strip()
    parsed = TraceRecord.model_validate_json(line)
    assert parsed.prompt == "héllo 世界"
    assert parsed.output == "café — 世界"
