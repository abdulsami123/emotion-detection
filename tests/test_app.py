import json
import zipfile
from pathlib import Path

import pytest

from autoace.app import (
    SCHEMA_COLUMNS,
    BatchValidation,
    results_to_csv,
    results_to_json,
    validate_batch,
)
from autoace.pipeline import FileResult
from autoace.schema import CallAnalysis


def _analysis(**overrides):
    base = dict(
        emotional_tone="neutral",
        emotional_intensity="medium",
        background_noise_present=False,
        background_noise_type="",
        background_noise_severity="none",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.7,
    )
    base.update(overrides)
    return CallAnalysis(**base)


def test_validation_reports_manifest_rows_with_no_audio(tmp_path: Path):
    """Reported BEFORE processing, per the brief."""
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,\ncall_999.ogg,\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "call_999.ogg" in report.missing_audio
    assert report.matched == ["call_001.ogg"]


def test_validation_reports_audio_with_no_manifest_row(tmp_path: Path):
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "stray.wav").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "stray.wav" in report.unlisted_audio


def test_validation_with_no_manifest_still_lists_the_audio(tmp_path: Path):
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    report = validate_batch(tmp_path)
    assert report.ok is False
    assert "call_001.ogg" in report.unlisted_audio


def test_csv_export_preserves_original_filenames():
    csv_text = results_to_csv([FileResult(name="call_042.mp3", analysis=_analysis())])
    assert "call_042.mp3" in csv_text
    assert "neutral" in csv_text
    for column in SCHEMA_COLUMNS:
        assert column in csv_text


def test_csv_export_includes_failed_files_with_a_reason():
    """A failed file must appear in the output, not vanish."""
    csv_text = results_to_csv(
        [FileResult(name="broken.wav", analysis=None, error="unsupported extension")]
    )
    assert "broken.wav" in csv_text
    assert "unsupported extension" in csv_text


def test_json_export_round_trips_and_marks_errors():
    payload = json.loads(
        results_to_json(
            [
                FileResult(name="ok.ogg", analysis=_analysis()),
                FileResult(name="bad.ogg", analysis=None, error="boom"),
            ]
        )
    )
    assert payload[0]["name"] == "ok.ogg"
    assert payload[0]["result"]["emotional_tone"] == "neutral"
    assert payload[1]["result"] is None
    assert payload[1]["error"] == "boom"


def test_review_flagged_rows_are_identifiable_in_the_table():
    """A degraded result must be visible, not buried - with the API key invalid
    every call currently takes the NLI fallback and flags for review."""
    from autoace.app import results_to_table

    rows = results_to_table(
        [
            FileResult(name="low.ogg", analysis=_analysis(confidence=0.55),
                       review_flagged=True),
            FileResult(name="high.ogg", analysis=_analysis(confidence=0.9)),
        ]
    )
    assert rows[0][0] == "low.ogg", "flagged rows must sort to the top"
    assert any("REVIEW" in str(cell) for cell in rows[0])


def test_app_builds_without_launching():
    """Import and construction must work headlessly; never call launch() here."""
    from autoace.app import build_app

    assert build_app() is not None


def _reload_modules(monkeypatch, tmp_path):
    """Point the job store at a tmp dir and reload, since config.DATA_DIR is
    read at import time. Returns (app_module, jobs_module)."""
    import importlib

    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)
    return app_module, jobs_module


def test_enqueue_batch_creates_a_job_without_running_inference(tmp_path, monkeypatch):
    """Enqueue must return immediately. A 50-file batch takes ~87 minutes; no
    HTTP request survives that, which is the entire reason for the job queue."""
    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "manifest.csv").write_text("name,result_json\na.ogg,\n", encoding="utf-8")

    def explode(path):
        raise AssertionError("enqueue must not run inference")

    monkeypatch.setattr(app_module, "analyse_file", explode, raising=False)

    job_id, status = app_module.enqueue_batch(str(batch))

    assert job_id
    conn = jobs_module.connect()
    try:
        st = jobs_module.job_status(conn, job_id)
        assert st.total == 1
        assert st.pending == 1
        assert st.status == "pending"
    finally:
        conn.close()


def test_enqueue_batch_reports_the_job_id_in_its_status(tmp_path, monkeypatch):
    """The evaluator must be told the ID, or they cannot come back to an
    ~87-minute batch after closing the page."""
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text("name,result_json\na.ogg,\n", encoding="utf-8")

    job_id, status = app_module.enqueue_batch(str(batch))
    assert job_id in status


def test_enqueue_batch_records_a_validation_failure_as_a_failed_job(tmp_path, monkeypatch):
    """A batch with no manifest must produce a job row carrying the reason, so
    a job-ID lookup can still explain it after a reload."""
    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")

    job_id, status = app_module.enqueue_batch(str(batch))

    assert job_id
    assert "manifest" in status.lower()

    conn = jobs_module.connect()
    try:
        st = jobs_module.job_status(conn, job_id)
        assert st.status == "failed"
        assert st.total == 0
    finally:
        conn.close()


def test_enqueue_batch_with_no_upload_returns_no_job(tmp_path, monkeypatch):
    app_module, _ = _reload_modules(monkeypatch, tmp_path)
    job_id, status = app_module.enqueue_batch(None)
    assert job_id == ""
    assert status


def test_enqueue_batch_warns_about_unmatched_files_but_still_queues(tmp_path, monkeypatch):
    """The brief requires unmatched files to be reported rather than silently
    skipped - and reported BEFORE processing, not discovered later."""
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "orphan.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text(
        "name,result_json\na.ogg,\nmissing.ogg,\n", encoding="utf-8"
    )

    job_id, status = app_module.enqueue_batch(str(batch))

    assert job_id
    assert "missing.ogg" in status, "a manifest row with no audio must be named"
    assert "orphan.ogg" in status, "audio with no manifest row must be named"


def test_poll_job_projects_rows_into_the_existing_table_shape(tmp_path, monkeypatch):
    """poll_job must feed the untouched results_to_table, so the review queue
    keeps sorting flagged rows to the top."""
    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    from autoace.pipeline import FileResult

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "b.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text(
        "name,result_json\na.ogg,\nb.ogg,\n", encoding="utf-8"
    )

    job_id, _ = app_module.enqueue_batch(str(batch))

    conn = jobs_module.connect()
    try:
        first = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, first,
            FileResult(
                name=first.name, analysis=_analysis(confidence=0.4),
                reasoning="nli fallback", review_flagged=True,
            ),
        )
    finally:
        conn.close()

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job(job_id)

    assert "1" in status_md
    # Index-agnostic on purpose: a later task adds three columns, and a test
    # coupled to a column position would break for a reason unrelated to what
    # it asserts.
    assert len(rows) == 2, "both the finished and the pending file must appear"
    assert any("REVIEW" in str(cell) for cell in rows[0]), "flagged rows sort first"
    assert csv_path and Path(csv_path).exists()
    assert json_path and Path(json_path).exists()


def test_poll_job_reports_an_unknown_id_without_raising(tmp_path, monkeypatch):
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job("nope")

    assert "not found" in status_md.lower()
    assert rows == []
    assert csv_path is None and json_path is None


def test_poll_job_with_no_id_is_a_no_op(tmp_path, monkeypatch):
    """The UI timer fires before anything is queued; that must be harmless."""
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job("")
    assert rows == []
    assert csv_path is None


def test_poll_job_surfaces_a_validation_failure(tmp_path, monkeypatch):
    """A failed batch looked up by ID must still explain itself."""
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")

    job_id, _ = app_module.enqueue_batch(str(batch))
    status_md, rows, csv_path, json_path, scoring = app_module.poll_job(job_id)

    assert "manifest" in status_md.lower()
    assert rows == []


def test_poll_job_shows_a_queue_estimate_while_work_is_outstanding(tmp_path, monkeypatch):
    """One worker means a second batch genuinely waits. A silent wait looks
    like a hang, so the panel must show progress or an estimate."""
    app_module, _ = _reload_modules(monkeypatch, tmp_path)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text("name,result_json\na.ogg,\n", encoding="utf-8")

    job_id, _ = app_module.enqueue_batch(str(batch))
    status_md, *_ = app_module.poll_job(job_id)

    assert "min" in status_md.lower() or "remaining" in status_md.lower()


def test_scoring_view_renders_after_the_workdir_is_gone(tmp_path, monkeypatch):
    """The manifest lives in the workdir, which is deleted when the job
    finishes. Scoring a labelled batch must still work afterwards, so the
    manifest text is persisted on the job row."""
    import json as _json

    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    from autoace.pipeline import FileResult

    expected_json = _analysis().model_dump_json()
    # Embed the JSON in a CSV cell: double the inner quotes, wrap in quotes.
    cell = '"' + expected_json.replace('"', '""') + '"'

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text(
        f"name,result_json\na.ogg,{cell}\n", encoding="utf-8"
    )

    job_id, _ = app_module.enqueue_batch(str(batch))

    conn = jobs_module.connect()
    try:
        claimed = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, claimed, FileResult(name=claimed.name, analysis=_analysis())
        )
        assert jobs_module.job_status(conn, job_id).status == "complete"
    finally:
        conn.close()

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job(job_id)

    assert scoring, "a labelled batch must render metrics"
    metrics = _json.loads(scoring)
    assert isinstance(metrics, dict) and metrics, "metrics must be real, not a message"


def test_scoring_view_stays_empty_for_an_unlabelled_batch(tmp_path, monkeypatch):
    """An unlabelled hidden test set has no ground truth, so there is nothing
    to score and nothing should be implied."""
    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    from autoace.pipeline import FileResult

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text("name,result_json\na.ogg,\n", encoding="utf-8")

    job_id, _ = app_module.enqueue_batch(str(batch))

    conn = jobs_module.connect()
    try:
        claimed = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, claimed, FileResult(name=claimed.name, analysis=_analysis())
        )
    finally:
        conn.close()

    _, _, _, _, scoring = app_module.poll_job(job_id)
    assert scoring == ""
