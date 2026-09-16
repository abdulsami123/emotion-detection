"""Worker tests.

`test_drain_once_processes_a_real_call` is marked slow: it runs the real
pipeline over a real call and takes roughly two minutes. The rest of the
worker's logic is covered without touching audio by injecting a fake analyser,
which also keeps these runnable in CI where the audio fixtures are absent.
"""

import pytest

from emotion_detection import jobs, worker
from emotion_detection.config import MAX_ATTEMPTS, reference_call
from emotion_detection.pipeline import FileResult


@pytest.fixture
def conn(tmp_path):
    c = jobs.connect(tmp_path / "jobs.db")
    yield c
    c.close()


def _enqueue_one(conn, tmp_path, name="a.ogg"):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / name
    path.write_bytes(b"not really audio")
    return jobs.enqueue(
        conn, workdir=workdir, audio_by_name={name: path},
        validation_json="{}", manifest_labelled=False,
    )


def test_drain_once_returns_false_on_an_empty_queue(conn):
    """An empty queue must not be an error, and must not call the analyser."""
    assert worker.drain_once(conn, analyse=lambda path: None) is False


def test_drain_once_records_a_result(conn, tmp_path):
    """A FileResult carrying an error is a COMPLETED unit of work with a
    failure recorded - not a retry."""
    job_id = _enqueue_one(conn, tmp_path)

    def fake_analyse(path):
        return FileResult(name="a.ogg", analysis=None, error="stub")

    assert worker.drain_once(conn, analyse=fake_analyse) is True

    row = conn.execute(
        "SELECT status, error, attempts FROM files WHERE job_id=?", (job_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "stub"
    assert row["attempts"] == 1, "one claim, no retry"


def test_drain_once_passes_the_claimed_path_to_the_analyser(conn, tmp_path):
    """The worker is a different process from the uploader, so it must analyse
    the absolute path recorded at enqueue time."""
    _enqueue_one(conn, tmp_path)
    seen = []

    def fake_analyse(path):
        seen.append(path)
        return FileResult(name="a.ogg", analysis=None, error="stub")

    worker.drain_once(conn, analyse=fake_analyse)
    assert len(seen) == 1
    assert seen[0].endswith("a.ogg")
    from pathlib import Path

    assert Path(seen[0]).is_absolute(), "a relative path would break in the worker"


def test_an_exception_requeues_then_eventually_fails(conn, tmp_path):
    """analyse_file is documented never to raise, so an exception here means
    something unexpected. It earns a retry, but not an unbounded one - an
    OOM-killed file requeued forever is a crash loop."""
    job_id = _enqueue_one(conn, tmp_path)

    def exploding_analyse(path):
        raise RuntimeError("boom")

    for _ in range(MAX_ATTEMPTS):
        assert worker.drain_once(conn, analyse=exploding_analyse) is True

    row = conn.execute(
        "SELECT status, attempts, error FROM files WHERE job_id=?", (job_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == MAX_ATTEMPTS
    assert "boom" in row["error"]
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "complete", "the job must not hang at running"


def test_drain_once_does_not_swallow_a_store_failure(conn, tmp_path):
    """A broken job store is not a per-file failure - it must surface, not be
    silently recorded against an innocent file."""
    _enqueue_one(conn, tmp_path)
    conn.close()

    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        worker.drain_once(conn, analyse=lambda path: None)


@pytest.mark.slow
def test_drain_once_processes_a_real_call(conn, tmp_path):
    """End to end over a genuine call, with the real pipeline. Proves the
    worker plumbing matches what analyse_file actually returns - a stub cannot
    catch a field-name mismatch."""
    import shutil

    workdir = tmp_path / "work"
    workdir.mkdir()
    staged = workdir / "call_003.ogg"
    shutil.copy2(reference_call("call_003.ogg"), staged)

    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name={"call_003.ogg": staged},
        validation_json="{}", manifest_labelled=False,
    )

    assert worker.drain_once(conn) is True

    row = conn.execute(
        "SELECT status, result_json, reasoning, audio_path FROM files WHERE job_id=?",
        (job_id,),
    ).fetchone()
    assert row["status"] == "done", "a real call must analyse successfully"
    assert row["result_json"]
    assert row["reasoning"], "the tone path must be recorded"
    assert row["audio_path"] is None, "audio must be unlinked after processing"
    assert not staged.exists()

    results = jobs.job_results(conn, job_id)
    assert results[0].analysis is not None
