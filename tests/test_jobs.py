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
