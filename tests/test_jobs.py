"""Job-store tests. Deliberately need no audio and no model weights, so they
run in CI - where `reference/` is absent because the audio is confidential
(brief section 5) and the weights are ~5 GiB."""

import sqlite3

import pytest

from autoace import jobs


@pytest.fixture
def conn(tmp_path):
    c = jobs.connect(tmp_path / "jobs.db")
    yield c
    c.close()


def test_schema_creates_both_tables(conn):
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
    workdir.mkdir(exist_ok=True)
    paths = {}
    for name in names:
        p = workdir / name
        p.write_bytes(b"not really audio")
        paths[name] = p
    return workdir, paths


def test_enqueue_creates_job_and_file_rows(conn, tmp_path):
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
