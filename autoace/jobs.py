"""The SQLite job store: queue, checkpoint log, and results table at once.

Deliberately imports nothing from `pipeline`, `gradio`, or any model library.
That keeps its tests fast and runnable in CI, and it keeps the seam clean: the
store knows about rows, the worker knows about audio, and neither knows about
the other's concerns.

`claim_next` / `complete` / `fail` are the entire executor contract. A
different executor (a durable workflow engine, a second machine) could replace
`worker.py` without touching this module or the web layer.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from autoace.config import JOB_TTL_SECONDS, JOBS_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL,
    created_at        REAL NOT NULL,
    expires_at        REAL NOT NULL,
    total_files       INTEGER NOT NULL,
    manifest_labelled INTEGER NOT NULL,
    workdir           TEXT NOT NULL,
    validation_json   TEXT NOT NULL,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS files (
    job_id         TEXT NOT NULL,
    name           TEXT NOT NULL,
    status         TEXT NOT NULL,
    attempts       INTEGER NOT NULL DEFAULT 0,
    audio_path     TEXT,
    result_json    TEXT,
    reasoning      TEXT,
    review_flagged INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    started_at     REAL,
    finished_at    REAL,
    PRIMARY KEY (job_id, name),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS files_claim ON files(status, job_id);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the store, creating it if absent.

    `isolation_level=None` turns off Python's implicit transaction handling so
    `claim_next` can issue an explicit `BEGIN IMMEDIATE` - the write lock that
    makes claiming exclusive.
    """
    path = Path(db_path) if db_path is not None else JOBS_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> float:
    return time.time()
