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

import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from autoace.config import (
    JOBS_BUSY_TIMEOUT_MS,
    JOBS_CONNECT_TIMEOUT_S,
    JOBS_DB,
    JOB_TTL_SECONDS,
)

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
    conn = sqlite3.connect(
        str(path), isolation_level=None, timeout=JOBS_CONNECT_TIMEOUT_S
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={JOBS_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> float:
    return time.time()


def enqueue(
    conn: sqlite3.Connection,
    *,
    workdir: Path | str,
    audio_by_name: dict[str, Path],
    validation_json: str,
    manifest_labelled: bool,
) -> str:
    """Insert one job row plus one `pending` file row per matched audio file.

    `validation_json` is stored verbatim so the pre-flight report (missing
    rows, unlisted audio, unsupported formats) survives a page reload - the
    evaluator must be able to come back to a job and still see why files were
    skipped.
    """
    job_id = str(uuid.uuid4())
    now = _now()
    # Resolve to absolute paths before writing: the worker that later reads
    # this row runs in a separate OS process (under systemd) with a possibly
    # different working directory, so a relative path stored here would
    # resolve against the wrong CWD there and silently miss the file.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO jobs (job_id, status, created_at, expires_at, total_files,"
            " manifest_labelled, workdir, validation_json)"
            " VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                now,
                now + JOB_TTL_SECONDS,
                len(audio_by_name),
                int(manifest_labelled),
                str(Path(workdir).resolve()),
                validation_json,
            ),
        )
        conn.executemany(
            "INSERT INTO files (job_id, name, status, audio_path)"
            " VALUES (?, ?, 'pending', ?)",
            [
                (job_id, name, str(Path(path).resolve()))
                for name, path in audio_by_name.items()
            ],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return job_id


def enqueue_failed(
    conn: sqlite3.Connection,
    *,
    workdir: Path | str,
    validation_json: str,
    error: str,
) -> str:
    """Record a batch that failed validation, with no file rows.

    A failed job still gets a row so the UI can render the reason from a
    job-ID lookup rather than only in the response that happened to be on
    screen at the time.
    """
    job_id = str(uuid.uuid4())
    now = _now()
    # Resolve for the same reason as in enqueue(): the worker is a separate
    # process and may have a different CWD, so a relative workdir would be
    # unresolvable there.
    conn.execute(
        "INSERT INTO jobs (job_id, status, created_at, expires_at, total_files,"
        " manifest_labelled, workdir, validation_json, error)"
        " VALUES (?, 'failed', ?, ?, 0, 0, ?, ?, ?)",
        (
            job_id,
            now,
            now + JOB_TTL_SECONDS,
            str(Path(workdir).resolve()),
            validation_json,
            error,
        ),
    )
    return job_id


@dataclass(frozen=True)
class ClaimedFile:
    job_id: str
    name: str
    audio_path: str
    attempts: int


def claim_next(conn: sqlite3.Connection) -> ClaimedFile | None:
    """Atomically take the oldest pending file and mark it running.

    `BEGIN IMMEDIATE` takes the write lock before the SELECT, so the
    select-then-update pair cannot interleave with another worker's. Ordering
    by rowid makes the queue FIFO across jobs, which is what the UI's queue
    position reports.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT job_id, name, audio_path, attempts FROM files"
            " WHERE status='pending' ORDER BY rowid LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        attempts = row["attempts"] + 1
        conn.execute(
            "UPDATE files SET status='running', attempts=?, started_at=?"
            " WHERE job_id=? AND name=?",
            (attempts, _now(), row["job_id"], row["name"]),
        )
        conn.execute(
            "UPDATE jobs SET status='running' WHERE job_id=? AND status='pending'",
            (row["job_id"],),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return ClaimedFile(
        job_id=row["job_id"],
        name=row["name"],
        audio_path=row["audio_path"],
        attempts=attempts,
    )


def _unlink_audio(conn: sqlite3.Connection, job_id: str, name: str) -> None:
    """Delete the staged audio and clear its path.

    Missing-file errors are swallowed: a retry after a crash may find the
    audio already gone, and that is success, not a problem to report.
    """
    row = conn.execute(
        "SELECT audio_path FROM files WHERE job_id=? AND name=?", (job_id, name)
    ).fetchone()
    if row is not None and row["audio_path"]:
        Path(row["audio_path"]).unlink(missing_ok=True)
    conn.execute(
        "UPDATE files SET audio_path=NULL WHERE job_id=? AND name=?", (job_id, name)
    )


def _finalise_if_done(conn: sqlite3.Connection, job_id: str) -> None:
    """Mark the job complete and remove its workdir once no file is left to
    process. Called from both complete() and fail() so either path can be the
    one that finishes the job."""
    outstanding = conn.execute(
        "SELECT COUNT(*) FROM files WHERE job_id=? AND status IN ('pending','running')",
        (job_id,),
    ).fetchone()[0]
    if outstanding:
        return

    conn.execute("UPDATE jobs SET status='complete' WHERE job_id=?", (job_id,))
    row = conn.execute(
        "SELECT workdir FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    if row is not None and row["workdir"]:
        shutil.rmtree(row["workdir"], ignore_errors=True)


def complete(
    conn: sqlite3.Connection, claimed: ClaimedFile, result: "FileResult"
) -> None:
    """Record a finished unit of work.

    `analyse_file` returning `FileResult(error=...)` is a COMPLETED unit with a
    failure recorded, so the row goes to `failed` - not back to `pending`. Only
    an exception or a killed process earns a retry (see `reconcile`).
    """
    status = "done" if result.analysis is not None else "failed"
    result_json = (
        result.analysis.model_dump_json() if result.analysis is not None else None
    )
    conn.execute(
        "UPDATE files SET status=?, result_json=?, reasoning=?, review_flagged=?,"
        " error=?, finished_at=? WHERE job_id=? AND name=?",
        (
            status,
            result_json,
            result.reasoning,
            int(bool(result.review_flagged)),
            result.error,
            _now(),
            claimed.job_id,
            claimed.name,
        ),
    )
    _unlink_audio(conn, claimed.job_id, claimed.name)
    _finalise_if_done(conn, claimed.job_id)


def fail(conn: sqlite3.Connection, claimed: ClaimedFile, error: str) -> None:
    """Give up on a file permanently."""
    conn.execute(
        "UPDATE files SET status='failed', error=?, finished_at=?"
        " WHERE job_id=? AND name=?",
        (error, _now(), claimed.job_id, claimed.name),
    )
    _unlink_audio(conn, claimed.job_id, claimed.name)
    _finalise_if_done(conn, claimed.job_id)
