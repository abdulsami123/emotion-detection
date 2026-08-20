"""Job-store tests. Deliberately need no audio and no model weights, so they
run in CI - where `reference/` is absent because the audio is confidential
(brief section 5) and the weights are ~5 GiB."""

import sqlite3

import pytest

from autoace import jobs
from autoace.pipeline import FileResult
from autoace.schema import CallAnalysis


@pytest.fixture
def conn(tmp_path):
    c = jobs.connect(tmp_path / "jobs.db")
    yield c
    c.close()


def test_schema_creates_both_tables(conn):
    """connect() must create the queue (`jobs`) and per-file (`files`) tables
    on a fresh database, or every other operation has nothing to act on."""
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"jobs", "files"} <= names


def test_wal_mode_is_enabled(conn):
    """One writer (worker) plus concurrent readers (web) without blocking is
    the whole reason SQLite is sufficient here - assert it rather than trust
    the PRAGMA silently succeeded."""
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_foreign_keys_are_enforced(conn):
    """files rows must not outlive their job, since expire() relies on the
    ON DELETE CASCADE."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO files (job_id, name, status) VALUES ('nope', 'a.ogg', 'pending')"
        )


def _make_audio(tmp_path, *names):
    """Stand-in audio files. Content is irrelevant: the job store never
    decodes them, it only records and unlinks paths."""
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        p = workdir / name
        p.write_bytes(b"not really audio")
        paths[name] = p
    return workdir, paths


def test_enqueue_creates_job_and_file_rows(conn, tmp_path):
    """A successful enqueue must produce exactly one job row plus one pending
    file row per matched audio file - a missing file row would be work the
    worker can never claim; an extra one would be a phantom the UI reports on
    with no audio behind it."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")

    job_id = jobs.enqueue(
        conn,
        workdir=workdir,
        audio_by_name=paths,
        validation_json='{"matched": ["a.ogg", "b.ogg"]}',
        manifest_labelled=True,
    )

    job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    assert job["status"] == "pending"
    assert job["total_files"] == 2
    assert job["manifest_labelled"] == 1
    assert job["validation_json"] == '{"matched": ["a.ogg", "b.ogg"]}'
    assert job["expires_at"] > job["created_at"]

    rows = conn.execute(
        "SELECT name, status, attempts FROM files WHERE job_id=? ORDER BY name",
        (job_id,),
    ).fetchall()
    assert [r["name"] for r in rows] == ["a.ogg", "b.ogg"]
    assert {r["status"] for r in rows} == {"pending"}
    assert {r["attempts"] for r in rows} == {0}


def test_enqueue_failed_job_records_no_file_rows(conn, tmp_path):
    """A batch that fails validation must not create work. The reason is
    stored on the job so the UI can show it after a page reload."""
    workdir, _ = _make_audio(tmp_path)

    job_id = jobs.enqueue_failed(
        conn,
        workdir=workdir,
        validation_json='{"matched": []}',
        error="No CSV manifest found in the upload.",
    )

    job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    assert job["status"] == "failed"
    assert job["error"] == "No CSV manifest found in the upload."
    assert job["total_files"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM files WHERE job_id=?", (job_id,)
    ).fetchone()[0] == 0


def test_enqueue_rolls_back_completely_on_a_mid_flight_failure(conn, tmp_path, monkeypatch):
    """The job row and the file rows must land together or not at all - a job
    with no files would sit at `pending` forever with nothing to claim."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")

    fixed = "11111111-1111-1111-1111-111111111111"
    monkeypatch.setattr(jobs.uuid, "uuid4", lambda: fixed)

    # Pre-seed the job_id so the INSERT into `jobs` collides.
    conn.execute(
        "INSERT INTO jobs (job_id, status, created_at, expires_at, total_files,"
        " manifest_labelled, workdir, validation_json)"
        " VALUES (?, 'pending', 0, 0, 0, 0, '', '{}')",
        (fixed,),
    )

    with pytest.raises(sqlite3.IntegrityError):
        jobs.enqueue(
            conn, workdir=workdir, audio_by_name=paths,
            validation_json="{}", manifest_labelled=False,
        )

    assert conn.execute(
        "SELECT COUNT(*) FROM files WHERE job_id=?", (fixed,)
    ).fetchone()[0] == 0, "no file rows may survive a rolled-back enqueue"

    # The connection must still be usable for a subsequent transaction.
    workdir2, paths2 = _make_audio(tmp_path / "second", "c.ogg")
    monkeypatch.setattr(jobs.uuid, "uuid4", lambda: "22222222-2222-2222-2222-222222222222")
    assert jobs.enqueue(
        conn, workdir=workdir2, audio_by_name=paths2,
        validation_json="{}", manifest_labelled=False,
    )


def test_claim_returns_a_pending_file_and_marks_it_running(conn, tmp_path):
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    claimed = jobs.claim_next(conn)
    assert claimed is not None
    assert claimed.job_id == job_id
    assert claimed.name == "a.ogg"
    assert claimed.audio_path == str(paths["a.ogg"].resolve())

    row = conn.execute(
        "SELECT status, started_at FROM files WHERE job_id=? AND name=?",
        (job_id, "a.ogg"),
    ).fetchone()
    assert row["status"] == "running"
    assert row["started_at"] is not None


def test_claim_is_exclusive(conn, tmp_path):
    """Two successive claims must never hand out the same row. This is what
    makes the design correct for more than one worker even though we run one."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")
    jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    first = jobs.claim_next(conn)
    second = jobs.claim_next(conn)
    third = jobs.claim_next(conn)

    assert {first.name, second.name} == {"a.ogg", "b.ogg"}
    assert third is None, "queue is drained; a third claim must return None"


def test_claim_increments_attempts_at_claim_time(conn, tmp_path):
    """attempts must rise when the row is handed out, NOT when it fails.

    A worker killed by the OOM killer never runs any Python, so a
    failure-time increment would leave attempts at 0 forever and reconcile()
    would requeue the same file indefinitely.
    """
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    jobs.claim_next(conn)
    assert conn.execute(
        "SELECT attempts FROM files WHERE job_id=? AND name=?", (job_id, "a.ogg")
    ).fetchone()["attempts"] == 1


def test_claim_marks_the_job_running(conn, tmp_path):
    """A job must not stay `pending` once work has started, or the UI would
    report a queued job that is actually in progress."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    jobs.claim_next(conn)
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "running"


def _analysis(**overrides) -> CallAnalysis:
    """A schema-valid analysis. Field values are irrelevant to the store; it
    round-trips JSON and never inspects the contents."""
    data = {
        "emotional_tone": "neutral",
        "emotional_intensity": "medium",
        "background_noise_present": False,
        "background_noise_type": "none",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
        "confidence": 0.8,
    }
    data.update(overrides)
    return CallAnalysis.model_validate(data)


def test_complete_records_result_and_unlinks_audio(conn, tmp_path):
    """Audio is deleted as soon as its row is recorded, not at end of job.
    Confidential audio must live on disk only while it is being processed."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    claimed = jobs.claim_next(conn)

    jobs.complete(
        conn,
        claimed,
        FileResult(
            name=claimed.name,
            analysis=_analysis(confidence=0.42),
            reasoning="nli fallback",
            review_flagged=True,
        ),
    )

    row = conn.execute(
        "SELECT * FROM files WHERE job_id=? AND name=?", (job_id, claimed.name)
    ).fetchone()
    assert row["status"] == "done"
    assert row["audio_path"] is None
    assert row["reasoning"] == "nli fallback"
    assert row["review_flagged"] == 1
    assert row["finished_at"] is not None
    assert not (workdir / claimed.name).exists(), "audio must be unlinked"
    assert (workdir / "b.ogg").exists(), "other files must be untouched"


def test_completing_a_file_that_analyse_file_failed_still_records_a_row(conn, tmp_path):
    """analyse_file returns FileResult(error=...) rather than raising. That is
    a completed unit of work with a failure recorded - not a retry."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    claimed = jobs.claim_next(conn)

    jobs.complete(
        conn, claimed,
        FileResult(name="a.ogg", analysis=None, error="could not decode"),
    )

    row = conn.execute(
        "SELECT status, error, result_json FROM files WHERE job_id=? AND name=?",
        (job_id, "a.ogg"),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "could not decode"
    assert row["result_json"] is None


def test_job_completes_and_workdir_is_removed_when_last_file_finishes(conn, tmp_path):
    """A finished job must release its staged audio directory, or an abandoned
    upload would leave confidential audio on the volume."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    first = jobs.claim_next(conn)
    jobs.complete(conn, first, FileResult(name=first.name, analysis=_analysis()))
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "running", "one file left, job is not done"

    second = jobs.claim_next(conn)
    jobs.complete(conn, second, FileResult(name=second.name, analysis=_analysis()))

    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "complete"
    assert not workdir.exists(), "workdir must be removed once the job is done"


def test_fail_marks_the_row_failed_and_unlinks(conn, tmp_path):
    """Giving up on a file permanently must still release its audio."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    claimed = jobs.claim_next(conn)

    jobs.fail(conn, claimed, "retries exhausted")

    row = conn.execute(
        "SELECT status, error, audio_path FROM files WHERE job_id=? AND name=?",
        (job_id, "a.ogg"),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "retries exhausted"
    assert row["audio_path"] is None
