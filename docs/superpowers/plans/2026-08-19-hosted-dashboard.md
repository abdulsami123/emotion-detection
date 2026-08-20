# Hosted Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing synchronous Gradio dashboard into an asynchronous, durable, job-queue-backed service that survives an ~87-minute 50-file batch and deploys to an Oracle ARM free-tier VM.

**Architecture:** A SQLite job store (`autoace/jobs.py`) is the queue, the checkpoint log, and the results table at once. Two systemd units share it: `autoace-web` (Gradio, enqueues and polls) and `autoace-worker` (claims one file, calls `analyse_file`, records the result, unlinks the audio). `autoace/pipeline.py` is **not modified** — `analyse_file` is already the correct unit of work and already never raises. Caddy terminates TLS on the VM for a DuckDNS hostname.

**Tech Stack:** Python 3.12 (VM) / 3.13 (dev), SQLite in WAL mode, Gradio 5.44.1, systemd, Caddy, Let's Encrypt, Oracle Cloud Ampere A1 (aarch64).

**Spec:** `docs/superpowers/specs/2026-08-19-hosted-dashboard-design.md`

---

## Critical context for the implementer

You have not seen this codebase. Five things will save you from breaking it:

1. **`config.py` holds every number.** Nothing numeric belongs anywhere else. Each value is annotated `MEASURED` / `DERIVED` / `UNFITTED`. Respect that convention.
2. **`analyse_file(path) -> FileResult` never raises.** It returns `FileResult(name, analysis=None, error="...")` on failure. Your worker still wraps it in `try/except`, because an OOM or a bug in the wrapper itself is not the same thing as a decode failure.
3. **Never let production code read `labels.csv`.** Ground truth reaches the system only via an uploaded manifest. Reading it inside `autoace/` would invalidate the evaluation.
4. **Never `git add -A` or `git add .`** — `reference/`, `aa/`, `models/`, `_batches/`, and `tests/fixtures/asr_baseline.json` are gitignored because the audio is confidential and the target repo is public. Stage explicit paths only.
5. **The existing 8 tests in `tests/test_app.py` must pass unmodified.** They are the regression boundary for this refactor. If you find yourself editing them, you have changed behaviour you were not asked to change.

`FileResult` is a 5-field dataclass in `autoace/pipeline.py:68`:

```python
@dataclass
class FileResult:
    name: str
    analysis: CallAnalysis | None
    error: str | None = None
    reasoning: str = ""
    review_flagged: bool = False
```

Every field maps onto one `files` column, which is why rehydration is lossless and the export functions need no changes at all.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `autoace/config.py` | Modify (append) | New hosting thresholds. Needs `import os` added — it currently imports only `Path`. |
| `autoace/jobs.py` | **Create** | The SQLite job store. Pure data: no Gradio import, no model import, no audio decoding. This is what makes it CI-safe. |
| `autoace/worker.py` | **Create** | The drain loop and its entrypoint. The only new file that imports `pipeline`. |
| `autoace/app.py` | Modify | `run_batch` splits into `enqueue_batch` + `poll_job`; `TABLE_HEADERS` and `results_to_table` widen to all nine schema fields (Task 13). |
| `autoace/eval.py` | Modify | `load_manifest` hardened against a BOM, an omitted `result_json` column, and blank rows (Task 14). |
| `tests/test_manifest.py` | **Create** | 6 manifest-robustness tests. No audio, no weights — CI-safe. |
| `tests/test_jobs.py` | **Create** | 9 unit tests. No audio, no weights, runs in CI. |
| `tests/test_worker.py` | **Create** | One `slow` integration test over one real call. |
| `tests/test_app.py` | Modify (append only) | New tests for enqueue/poll. **Do not touch the existing 8.** |
| `deploy/setup.sh` | **Create** | Idempotent VM provisioning. |
| `deploy/autoace-web.service` | **Create** | systemd unit, Gradio on 127.0.0.1:7860. |
| `deploy/autoace-worker.service` | **Create** | systemd unit, the drain loop. |
| `deploy/Caddyfile` | **Create** | TLS termination + reverse proxy. |
| `deploy/duckdns.sh` + `.timer` + `.service` | **Create** | Keeps the A record pointed at the VM. |
| `README.md`, `docs/MEMO.md` | Modify | Hosting, privacy, CI limitation, measured numbers. |

---

## Task 1: Configuration

**Files:**
- Modify: `autoace/config.py` (add `import os` near the top; append the block at the end)

- [ ] **Step 1: Add the `os` import**

`config.py` currently imports only `Path`. Change the import block to:

```python
import os
from pathlib import Path
```

- [ ] **Step 2: Append the hosting block to the end of `config.py`**

```python
# ------------------------------------------------------- hosted dashboard
# Job store and staged uploads. AUTOACE_DATA_DIR lets the tests point this at
# a tmp_path and lets the VM point it at the boot volume.
DATA_DIR = Path(os.environ.get("AUTOACE_DATA_DIR", REPO_ROOT / "_data"))
JOBS_DB = DATA_DIR / "jobs.db"
UPLOAD_DIR = DATA_DIR / "uploads"

JOB_TTL_SECONDS = 7 * 24 * 3600   # DERIVED — results kept until download. Audio is
                                  # unlinked per file regardless, so this governs
                                  # metadata and results only.
MAX_ATTEMPTS = 2                  # DERIVED — breaks the OOM crash loop. reconcile()
                                  # alone would requeue an OOM-killed file forever.
STALE_RUNNING_SECONDS = 900.0     # DERIVED — 5.6x the slowest measured file (160.8s)
WORKER_POLL_SECONDS = 2.0
UI_POLL_SECONDS = 5.0
MEAN_SECONDS_PER_FILE = 105.0     # MEASURED — mean of the three provided calls
                                  # (58.2/58.7/129.8s) x the 1.18 two-thread penalty
MAX_UPLOAD_MB = 500               # DERIVED — 50 files at the largest provided call
                                  # (2.8 MB) is ~140 MB; 500 MB is generous headroom
```

`REPO_ROOT` already exists in `config.py` — reuse it, do not redefine it. The default is `_data` inside the repo rather than `/opt/autoace/data` so the tests and a local run work without setting anything; the systemd units set `AUTOACE_DATA_DIR` explicitly.

- [ ] **Step 3: Add `_data/` to `.gitignore`**

Append one line to `.gitignore`:

```
_data/
```

- [ ] **Step 4: Verify config imports cleanly**

Run: `python -c "from autoace.config import JOBS_DB, MAX_ATTEMPTS, MEAN_SECONDS_PER_FILE; print(JOBS_DB, MAX_ATTEMPTS, MEAN_SECONDS_PER_FILE)"`
Expected: a path ending in `_data/jobs.db`, then `2 105.0`. No exception.

- [ ] **Step 5: Commit**

```bash
git add autoace/config.py .gitignore
git commit -m "feat: add hosted-dashboard configuration"
```

---

## Task 2: Job store schema and connection

**Files:**
- Create: `autoace/jobs.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_jobs.py`:

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoace.jobs'`

- [ ] **Step 3: Write the minimal implementation**

Create `autoace/jobs.py`:

```python
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
from pathlib import Path

from autoace.config import JOBS_DB

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: SQLite job store schema and connection"
```

---

## Task 3: Enqueue

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k enqueue -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'enqueue'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/jobs.py`:

```python
import uuid

from autoace.config import JOB_TTL_SECONDS


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
                str(workdir),
                validation_json,
            ),
        )
        conn.executemany(
            "INSERT INTO files (job_id, name, status, audio_path)"
            " VALUES (?, ?, 'pending', ?)",
            [(job_id, name, str(path)) for name, path in audio_by_name.items()],
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
    conn.execute(
        "INSERT INTO jobs (job_id, status, created_at, expires_at, total_files,"
        " manifest_labelled, workdir, validation_json, error)"
        " VALUES (?, 'failed', ?, ?, 0, 0, ?, ?, ?)",
        (job_id, now, now + JOB_TTL_SECONDS, str(workdir), validation_json, error),
    )
    return job_id
```

Move `import uuid` up to the module's import block rather than leaving it mid-file, and merge `JOB_TTL_SECONDS` into the existing `from autoace.config import ...` line.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: enqueue jobs and validation failures"
```

---

## Task 4: Exclusive claim

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
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
    assert claimed.audio_path == str(paths["a.ogg"])

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
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    jobs.claim_next(conn)
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "running"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k claim -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'claim_next'`

- [ ] **Step 3: Write the minimal implementation**

Add the dataclass near the top of `autoace/jobs.py` (after the imports), then append the function:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimedFile:
    job_id: str
    name: str
    audio_path: str
    attempts: int
```

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: exclusive file claiming with claim-time attempt counting"
```

---

## Task 5: Complete, fail, and job finalisation

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
from autoace.pipeline import FileResult
from autoace.schema import CallAnalysis


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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k "complete or fail" -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'complete'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/jobs.py`:

```python
import shutil


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
```

Add `import shutil` to the module import block. Do **not** import `FileResult` at module scope — that would pull `pipeline`, and with it torch, into the job store and destroy the CI-safety property. The annotation is a string for exactly that reason.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 13 passed.

- [ ] **Step 5: Verify the store still imports without torch**

Run: `python -c "import sys; import autoace.jobs; assert 'torch' not in sys.modules; print('jobs.py is torch-free')"`
Expected: `jobs.py is torch-free`

This guard matters: if it fails, `tests/test_jobs.py` stops being runnable in CI.

- [ ] **Step 6: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: record results, unlink audio per file, finalise jobs"
```

---

## Task 6: Reconcile — the crash-loop guard

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
def test_reconcile_resets_stale_running_to_pending(conn, tmp_path):
    """A worker killed mid-file leaves a `running` row nobody owns. On
    startup it must go back in the queue - completed rows are already durable,
    so the worst case is one file re-run."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    jobs.claim_next(conn)

    requeued, failed = jobs.reconcile(conn, stale_after=0.0)

    assert (requeued, failed) == (1, 0)
    assert conn.execute(
        "SELECT status FROM files WHERE job_id=? AND name=?", (job_id, "a.ogg")
    ).fetchone()["status"] == "pending"


def test_reconcile_fails_a_row_past_max_attempts(conn, tmp_path):
    """The OOM crash-loop guard. If the NLI fallback pushes the worker past
    available RAM, the OOM killer takes it mid-file, reconcile requeues it, and
    it dies again on the same file - forever. MAX_ATTEMPTS breaks that."""
    from autoace.config import MAX_ATTEMPTS

    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    for _ in range(MAX_ATTEMPTS):
        assert jobs.claim_next(conn) is not None
        jobs.reconcile(conn, stale_after=0.0)

    row = conn.execute(
        "SELECT status, attempts, error FROM files WHERE job_id=? AND name=?",
        (job_id, "a.ogg"),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == MAX_ATTEMPTS
    assert "attempt" in (row["error"] or "").lower()
    assert jobs.claim_next(conn) is None, "an exhausted file must not be reissued"


def test_reconcile_leaves_a_fresh_running_row_alone(conn, tmp_path):
    """A file legitimately in progress must not be yanked out from under the
    worker by a periodic sweep."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    jobs.claim_next(conn)

    requeued, failed = jobs.reconcile(conn, stale_after=900.0)

    assert (requeued, failed) == (0, 0)
    assert conn.execute(
        "SELECT status FROM files"
    ).fetchone()["status"] == "running"


def test_reconcile_finalises_a_job_whose_last_file_it_failed(conn, tmp_path):
    """If reconcile is what exhausts the final file, it must still close the
    job - otherwise the job sits at `running` forever with nothing to run."""
    from autoace.config import MAX_ATTEMPTS

    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    for _ in range(MAX_ATTEMPTS):
        jobs.claim_next(conn)
        jobs.reconcile(conn, stale_after=0.0)

    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "complete"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k reconcile -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'reconcile'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/jobs.py`:

```python
from autoace.config import MAX_ATTEMPTS, STALE_RUNNING_SECONDS


def reconcile(
    conn: sqlite3.Connection,
    stale_after: float = STALE_RUNNING_SECONDS,
    now: float | None = None,
) -> tuple[int, int]:
    """Return orphaned `running` rows to the queue, or fail them if they have
    used up their attempts. Returns `(requeued, failed)`.

    The worker calls this at startup with `stale_after=0.0`: it is the only
    worker, so nothing can legitimately be running. The default threshold
    exists for a periodic sweep and for tests that need a fresh row left alone.
    """
    now = _now() if now is None else now
    cutoff = now - stale_after

    rows = conn.execute(
        "SELECT job_id, name, attempts FROM files"
        " WHERE status='running' AND (started_at IS NULL OR started_at <= ?)",
        (cutoff,),
    ).fetchall()

    requeued = failed = 0
    touched_jobs = set()
    for row in rows:
        touched_jobs.add(row["job_id"])
        if row["attempts"] >= MAX_ATTEMPTS:
            conn.execute(
                "UPDATE files SET status='failed', error=?, finished_at=?"
                " WHERE job_id=? AND name=?",
                (
                    f"abandoned after {row['attempts']} attempts - the worker died "
                    f"each time without recording a result (likely out of memory)",
                    now,
                    row["job_id"],
                    row["name"],
                ),
            )
            _unlink_audio(conn, row["job_id"], row["name"])
            failed += 1
        else:
            conn.execute(
                "UPDATE files SET status='pending', started_at=NULL"
                " WHERE job_id=? AND name=?",
                (row["job_id"], row["name"]),
            )
            requeued += 1

    for job_id in touched_jobs:
        _finalise_if_done(conn, job_id)

    return requeued, failed
```

### Step 3b: Add the orphan-job sweep

**Added after review of Tasks 4–5 confirmed a real defect.** `complete()` and `fail()` are now atomic in SQL, but their filesystem work necessarily follows the commit — so a crash in that narrow window leaves a job marked `complete` whose workdir was never removed, and (for any row written before the atomicity fix) a job stuck at `running` with zero outstanding files. Neither is detectable by the sweep above, which only looks at `running` **file** rows.

Append to `reconcile`, before the `return`:

```python
    # Safety net. Two states the file-row sweep above cannot see:
    #   - a job left at pending/running with no outstanding files, from a crash
    #     between a file update and its finalisation (possible for rows written
    #     before complete()/fail() became atomic);
    #   - a job already marked complete whose workdir removal did not happen,
    #     because the filesystem work necessarily follows the SQL commit.
    # Both leave confidential staged audio on disk, so the sweep is a privacy
    # measure, not only a bookkeeping one.
    for row in conn.execute(
        "SELECT job_id FROM jobs WHERE status IN ('pending','running')"
        " AND job_id NOT IN (SELECT job_id FROM files"
        "                    WHERE status IN ('pending','running'))"
    ).fetchall():
        workdir = _finalise_sql(conn, row["job_id"])
        _remove_workdir(workdir)
        orphaned += 1

    for row in conn.execute(
        "SELECT job_id, workdir FROM jobs WHERE status='complete' AND workdir<>''"
    ).fetchall():
        if Path(row["workdir"]).exists():
            _remove_workdir(row["workdir"])
```

Initialise `orphaned = 0` alongside `requeued` and `failed`, and return `(requeued, failed, orphaned)`. Update `test_reconcile_*` assertions and `worker.run_forever`'s startup log accordingly — the tuple is now three-wide.

Add this test:

```python
def test_reconcile_finalises_a_job_orphaned_at_running(conn, tmp_path):
    """A crash between a file update and its finalisation leaves a job at
    `running` with zero outstanding files - invisible to the running-file
    sweep, and leaving confidential audio on disk. Reconcile must repair it."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    claimed = jobs.claim_next(conn)
    # Simulate the crash: mark the file done WITHOUT finalising the job.
    conn.execute(
        "UPDATE files SET status='done' WHERE job_id=? AND name=?",
        (job_id, claimed.name),
    )
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "running"

    requeued, failed, orphaned = jobs.reconcile(conn, stale_after=0.0)

    assert (requeued, failed, orphaned) == (0, 0, 1)
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "complete"
    assert not workdir.exists(), "the orphaned job's audio must be cleaned up"
```

Merge `MAX_ATTEMPTS` and `STALE_RUNNING_SECONDS` into the existing `from autoace.config import ...` line rather than adding a second import statement.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: reconcile stale work with a retry cap to break OOM crash loops"
```

---

## Task 7: Expiry

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
def test_expire_removes_old_jobs_and_their_files(conn, tmp_path):
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    from autoace.config import JOB_TTL_SECONDS
    import time as _time

    removed = jobs.expire(conn, now=_time.time() + JOB_TTL_SECONDS + 1.0)

    assert removed == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM files WHERE job_id=?", (job_id,)
    ).fetchone()[0] == 0, "ON DELETE CASCADE must remove the file rows"
    assert not workdir.exists(), "an expired job must not leave audio on disk"


def test_expire_leaves_live_jobs_alone(conn, tmp_path):
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    assert jobs.expire(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k expire -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'expire'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/jobs.py`:

```python
def expire(conn: sqlite3.Connection, now: float | None = None) -> int:
    """Delete jobs past their TTL and remove any residue on disk.

    Audio is already unlinked per file during processing, so this normally
    only removes metadata and results. The `rmtree` covers the case where a
    job expired without ever finishing - an abandoned upload must not leave
    confidential audio on the volume.
    """
    now = _now() if now is None else now
    rows = conn.execute(
        "SELECT job_id, workdir FROM jobs WHERE expires_at <= ?", (now,)
    ).fetchall()
    for row in rows:
        if row["workdir"]:
            shutil.rmtree(row["workdir"], ignore_errors=True)
        conn.execute("DELETE FROM jobs WHERE job_id=?", (row["job_id"],))
    return len(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: expire jobs past their retention window"
```

---

## Task 8: Status projection and lossless rehydration

**Files:**
- Modify: `autoace/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_jobs.py`:

```python
def test_job_status_reports_counts_and_queue_position(conn, tmp_path):
    workdir_a, paths_a = _make_audio(tmp_path / "a", "a1.ogg", "a2.ogg")
    workdir_b, paths_b = _make_audio(tmp_path / "b", "b1.ogg")
    job_a = jobs.enqueue(
        conn, workdir=workdir_a, audio_by_name=paths_a,
        validation_json="{}", manifest_labelled=True,
    )
    job_b = jobs.enqueue(
        conn, workdir=workdir_b, audio_by_name=paths_b,
        validation_json="{}", manifest_labelled=False,
    )

    claimed = jobs.claim_next(conn)
    jobs.complete(conn, claimed, FileResult(name=claimed.name, analysis=_analysis()))

    status_a = jobs.job_status(conn, job_a)
    assert status_a.total == 2
    assert status_a.done == 1
    assert status_a.pending == 1
    assert status_a.manifest_labelled is True
    assert status_a.queue_position == 0, "job A is at the head of the queue"

    status_b = jobs.job_status(conn, job_b)
    assert status_b.queue_position == 1, "job B waits behind job A"
    assert status_b.eta_seconds == pytest.approx(105.0 * 2, rel=0.01)


def test_job_status_is_none_for_an_unknown_id(conn):
    assert jobs.job_status(conn, "no-such-job") is None


def test_job_results_rehydrate_losslessly(conn, tmp_path):
    """FileResult has exactly five fields and each maps to one column, so the
    round trip must be exact - that is what lets results_to_csv/json/table stay
    completely unchanged."""
    workdir, paths = _make_audio(tmp_path, "a.ogg", "b.ogg")
    jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )

    first = jobs.claim_next(conn)
    original = FileResult(
        name=first.name,
        analysis=_analysis(emotional_tone="frustrated", confidence=0.51),
        reasoning="nli fallback: openai unavailable",
        review_flagged=True,
    )
    jobs.complete(conn, first, original)

    second = jobs.claim_next(conn)
    jobs.complete(
        conn, second,
        FileResult(name=second.name, analysis=None, error="could not decode"),
    )

    results = {r.name: r for r in jobs.job_results(conn, first.job_id)}
    assert len(results) == 2

    got = results[first.name]
    assert got.name == original.name
    assert got.analysis == original.analysis
    assert got.reasoning == original.reasoning
    assert got.review_flagged is True
    assert got.error is None

    failed = results[second.name]
    assert failed.analysis is None
    assert failed.error == "could not decode"


def test_job_results_include_files_not_yet_processed(conn, tmp_path):
    """A pending file must appear in the table as pending rather than vanish,
    so a 50-row batch shows 50 rows from the moment it is enqueued."""
    workdir, paths = _make_audio(tmp_path, "a.ogg")
    job_id = jobs.enqueue(
        conn, workdir=workdir, audio_by_name=paths,
        validation_json="{}", manifest_labelled=False,
    )
    results = jobs.job_results(conn, job_id)
    assert [r.name for r in results] == ["a.ogg"]
    assert results[0].analysis is None
    assert results[0].error is None, "pending is not an error"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_jobs.py -k "job_status or job_results" -v`
Expected: FAIL — `AttributeError: module 'autoace.jobs' has no attribute 'job_status'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/jobs.py`:

```python
from autoace.config import MEAN_SECONDS_PER_FILE


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    status: str
    total: int
    done: int
    failed: int
    running: int
    pending: int
    manifest_labelled: bool
    validation_json: str
    error: str | None
    queue_position: int
    eta_seconds: float

    @property
    def finished(self) -> int:
        return self.done + self.failed


def job_status(conn: sqlite3.Connection, job_id: str) -> JobStatus | None:
    """Counts plus the queue position, for the status panel.

    `queue_position` is how many OTHER jobs are ahead in the FIFO, and
    `eta_seconds` covers every unfinished file ahead of this job as well as its
    own - one worker means a second batch genuinely waits, and a silent wait is
    the failure mode we are avoiding.
    """
    job = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if job is None:
        return None

    counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM files WHERE job_id=? GROUP BY status",
        (job_id,),
    ):
        counts[row["status"]] = row["n"]

    ahead_jobs = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE created_at < ? AND status IN"
        " ('pending','running')",
        (job["created_at"],),
    ).fetchone()[0]
    ahead_files = conn.execute(
        "SELECT COUNT(*) FROM files f JOIN jobs j ON j.job_id = f.job_id"
        " WHERE f.status IN ('pending','running')"
        "   AND (j.created_at < ? OR j.job_id = ?)",
        (job["created_at"], job_id),
    ).fetchone()[0]

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        total=job["total_files"],
        done=counts["done"],
        failed=counts["failed"],
        running=counts["running"],
        pending=counts["pending"],
        manifest_labelled=bool(job["manifest_labelled"]),
        validation_json=job["validation_json"],
        error=job["error"],
        queue_position=ahead_jobs,
        eta_seconds=ahead_files * MEAN_SECONDS_PER_FILE,
    )


def job_results(conn: sqlite3.Connection, job_id: str) -> list["FileResult"]:
    """Rehydrate `FileResult` objects from the store.

    Imported lazily: keeping `pipeline` (and therefore torch) out of this
    module's import graph is what makes the job-store tests runnable in CI.
    """
    from autoace.pipeline import FileResult
    from autoace.schema import CallAnalysis

    results = []
    for row in conn.execute(
        "SELECT name, result_json, reasoning, review_flagged, error FROM files"
        " WHERE job_id=? ORDER BY rowid",
        (job_id,),
    ):
        analysis = (
            CallAnalysis.model_validate_json(row["result_json"])
            if row["result_json"]
            else None
        )
        results.append(
            FileResult(
                name=row["name"],
                analysis=analysis,
                error=row["error"],
                reasoning=row["reasoning"] or "",
                review_flagged=bool(row["review_flagged"]),
            )
        )
    return results
```

Merge `MEAN_SECONDS_PER_FILE` into the existing config import line.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add autoace/jobs.py tests/test_jobs.py
git commit -m "feat: job status projection and lossless FileResult rehydration"
```

---

## Task 9: The worker

**Files:**
- Create: `autoace/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_worker.py`:

```python
"""Worker tests.

`test_drain_once_processes_a_real_call` is marked slow: it runs the real
pipeline over a real call and takes roughly a minute. The rest of the worker's
logic is covered without touching audio by injecting a fake analyser.
"""

import pytest

from autoace import jobs, worker
from autoace.config import MAX_ATTEMPTS, reference_call
from autoace.pipeline import FileResult


@pytest.fixture
def conn(tmp_path):
    c = jobs.connect(tmp_path / "jobs.db")
    yield c
    c.close()


def _enqueue_one(conn, tmp_path, name="a.ogg"):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    path = workdir / name
    path.write_bytes(b"not really audio")
    return jobs.enqueue(
        conn, workdir=workdir, audio_by_name={name: path},
        validation_json="{}", manifest_labelled=False,
    )


def test_drain_once_returns_false_on_an_empty_queue(conn):
    assert worker.drain_once(conn, analyse=lambda path: None) is False


def test_drain_once_records_a_result(conn, tmp_path):
    job_id = _enqueue_one(conn, tmp_path)

    def fake_analyse(path):
        return FileResult(name="a.ogg", analysis=None, error="stub")

    assert worker.drain_once(conn, analyse=fake_analyse) is True

    row = conn.execute(
        "SELECT status, error FROM files WHERE job_id=?", (job_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "stub"


def test_an_exception_requeues_then_eventually_fails(conn, tmp_path):
    """analyse_file is documented never to raise, so an exception here means
    something unexpected. It earns a retry, but not an unbounded one."""
    job_id = _enqueue_one(conn, tmp_path)

    def exploding_analyse(path):
        raise RuntimeError("boom")

    for _ in range(MAX_ATTEMPTS):
        worker.drain_once(conn, analyse=exploding_analyse)

    row = conn.execute(
        "SELECT status, attempts, error FROM files WHERE job_id=?", (job_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == MAX_ATTEMPTS
    assert "boom" in row["error"]
    assert conn.execute(
        "SELECT status FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()["status"] == "complete"


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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_worker.py -m "not slow" -v`
Expected: FAIL — `ImportError: cannot import name 'worker' from 'autoace'`

- [ ] **Step 3: Write the minimal implementation**

Create `autoace/worker.py`:

```python
"""The drain loop.

Runs as its own systemd unit rather than a thread or a forked child. A thread
is ruled out because openSMILE and the DSP paths do not reliably release the
GIL, so a worker thread would stall the UI. A separate unit additionally buys
independent restarts and independent journald streams.

Model weights load lazily on the first call and stay resident, which is why
this process - not the web process - is the one that needs the memory.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from typing import Callable

from autoace import jobs
from autoace.config import MAX_ATTEMPTS, WORKER_POLL_SECONDS

log = logging.getLogger("autoace.worker")

Analyser = Callable[[str], object]


def _default_analyser(path: str):
    """Imported lazily so `import autoace.worker` stays cheap - the tests that
    inject a fake analyser must not pay for loading torch."""
    from autoace.pipeline import analyse_file

    return analyse_file(path)


def drain_once(
    conn: sqlite3.Connection, analyse: Analyser | None = None
) -> bool:
    """Process at most one file. Returns True if work was done.

    `analyse_file` is documented never to raise, so an exception escaping it is
    unexpected rather than routine - it earns a retry, bounded by MAX_ATTEMPTS,
    because the alternative to a bound is a crash loop.
    """
    analyse = analyse or _default_analyser

    claimed = jobs.claim_next(conn)
    if claimed is None:
        return False

    log.info(
        "analysing %s (job %s, attempt %d)",
        claimed.name, claimed.job_id, claimed.attempts,
    )
    try:
        result = analyse(claimed.audio_path)
    except Exception as exc:  # noqa: BLE001 - failure isolation is the point
        if claimed.attempts >= MAX_ATTEMPTS:
            log.exception("giving up on %s after %d attempts", claimed.name,
                          claimed.attempts)
            jobs.fail(
                conn, claimed,
                f"unexpected failure after {claimed.attempts} attempts: {exc}",
            )
        else:
            log.exception("retrying %s after unexpected failure", claimed.name)
            jobs.reconcile(conn, stale_after=0.0)
        return True

    jobs.complete(conn, claimed, result)
    log.info("recorded %s", claimed.name)
    return True


def run_forever(conn: sqlite3.Connection | None = None) -> None:
    """Reconcile, then drain, then sweep expired jobs, forever.

    `reconcile(stale_after=0.0)` on startup is safe because this is the only
    worker: nothing can legitimately be `running` when it begins.
    """
    conn = conn or jobs.connect()

    requeued, failed = jobs.reconcile(conn, stale_after=0.0)
    if requeued or failed:
        log.warning(
            "startup reconcile: %d requeued, %d abandoned past the attempt cap",
            requeued, failed,
        )

    last_sweep = 0.0
    while True:
        did_work = drain_once(conn)

        now = time.time()
        if now - last_sweep > 3600.0:
            removed = jobs.expire(conn)
            if removed:
                log.info("expired %d job(s) past the retention window", removed)
            last_sweep = now

        if not did_work:
            time.sleep(WORKER_POLL_SECONDS)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the fast tests to verify they pass**

Run: `python -m pytest tests/test_worker.py -m "not slow" -v`
Expected: 3 passed.

- [ ] **Step 5: Run the slow integration test**

Run: `python -m pytest tests/test_worker.py -m slow -v`
Expected: 1 passed, in roughly 130–160 seconds. Requires `reference/call_003.ogg`. If `reference/` is absent, this test cannot run — that is expected in CI and documented in the spec §10.1.

- [ ] **Step 6: Commit**

```bash
git add autoace/worker.py tests/test_worker.py
git commit -m "feat: worker drain loop with startup reconcile and expiry sweep"
```

---

## Task 10: Split `run_batch` into `enqueue_batch`

**Files:**
- Modify: `autoace/app.py:243-323` (replace `run_batch`)
- Modify: `tests/test_app.py` (append only)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:

```python
def test_enqueue_batch_creates_a_job_without_running_inference(tmp_path, monkeypatch):
    """Enqueue must return immediately. A 50-file batch takes ~87 minutes; no
    HTTP request survives that, which is the entire reason for the job queue."""
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)

    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "manifest.csv").write_text("name,result_json\na.ogg,\n", encoding="utf-8")

    def explode(path):
        raise AssertionError("enqueue must not run inference")

    monkeypatch.setattr(app_module, "analyse_file", explode, raising=False)

    job_id, status = app_module.enqueue_batch(str(batch))

    assert job_id
    assert "a.ogg" not in status or "queued" in status.lower()

    conn = jobs_module.connect()
    try:
        st = jobs_module.job_status(conn, job_id)
        assert st.total == 1
        assert st.pending == 1
    finally:
        conn.close()


def test_enqueue_batch_records_a_validation_failure_as_a_failed_job(tmp_path, monkeypatch):
    """A batch with no manifest must produce a job row carrying the reason, so
    a job-ID lookup can still explain it after a reload."""
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)

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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_app.py -k enqueue_batch -v`
Expected: FAIL — `AttributeError: module 'autoace.app' has no attribute 'enqueue_batch'`

- [ ] **Step 3: Replace `run_batch` with `_describe_validation` and `enqueue_batch`**

In `autoace/app.py`, delete the whole `run_batch` function (currently lines 243–323) and put this in its place. Add `from autoace import jobs` and `from autoace.config import MAX_UPLOAD_MB, REVIEW_THRESHOLD, UI_POLL_SECONDS` to the imports.

```python
def _describe_validation(workdir: Path, validation: BatchValidation) -> list[str]:
    """Human-readable problems found before any inference runs.

    Extracted from the old `run_batch` unchanged in substance: the brief
    requires unmatched files to be reported rather than silently skipped, and
    that has to happen at enqueue time, not when the worker gets there.
    """
    problems: list[str] = []
    if validation.manifest_path is None:
        problems.append("No CSV manifest found in the upload.")
    if validation.missing_audio:
        problems.append(
            "Manifest rows with no matching audio file: "
            + ", ".join(validation.missing_audio)
        )
    if validation.unlisted_audio:
        problems.append(
            "Audio files with no manifest row (will not be processed): "
            + ", ".join(validation.unlisted_audio)
        )
    unsupported = _unsupported_files(workdir)
    if unsupported:
        problems.append(
            "Unsupported file formats (ignored): " + ", ".join(unsupported)
        )
    return problems


def enqueue_batch(upload) -> tuple[str, str]:
    """Validate the upload and queue it. Returns `(job_id, status_markdown)`.

    Returns as soon as the rows are written - the worker does the work. An
    empty `job_id` means nothing was queued and there is nothing to poll.
    """
    if upload is None:
        return "", "No file uploaded."

    workdir = _prepare_workdir(upload)
    validation = validate_batch(workdir)
    problems = _describe_validation(workdir, validation)
    validation_json = json.dumps(
        {
            "matched": validation.matched,
            "missing_audio": validation.missing_audio,
            "unlisted_audio": validation.unlisted_audio,
            "problems": problems,
        }
    )

    conn = jobs.connect()
    try:
        if not validation.matched:
            reason = "Validation failed - nothing to process.\n" + "\n".join(
                f"- {p}" for p in problems
            )
            job_id = jobs.enqueue_failed(
                conn, workdir=workdir, validation_json=validation_json, error=reason
            )
            return job_id, reason

        audio_by_name: dict[str, Path] = {}
        for path in _iter_audio_files(workdir):
            audio_by_name.setdefault(path.name, path)
        matched_audio = {name: audio_by_name[name] for name in validation.matched}

        manifest_labelled = False
        if validation.manifest_path is not None:
            manifest_labelled = any(
                row.expected is not None
                for row in load_manifest(str(validation.manifest_path))
            )

        job_id = jobs.enqueue(
            conn,
            workdir=workdir,
            audio_by_name=matched_audio,
            validation_json=validation_json,
            manifest_labelled=manifest_labelled,
        )
    finally:
        conn.close()

    lines = [
        f"**Queued {len(matched_audio)} file(s).** Job ID: `{job_id}`",
        "",
        "Keep this job ID. You can close the page and paste it into the "
        "**Look up a job** box to come back to these results.",
    ]
    if problems:
        lines.append("")
        lines.append("Validation warnings:")
        lines.extend(f"- {p}" for p in problems)
    return job_id, "\n".join(lines)
```

- [ ] **Step 4: Run the new tests and the existing 8**

Run: `python -m pytest tests/test_app.py -v`
Expected: 10 passed. The original 8 must pass **unmodified** — if any of them fail, revert and reconcile before continuing.

- [ ] **Step 5: Commit**

```bash
git add autoace/app.py tests/test_app.py
git commit -m "feat: enqueue batches instead of processing them inline"
```

---

## Task 11: `poll_job`

**Files:**
- Modify: `autoace/app.py`
- Modify: `tests/test_app.py` (append only)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:

```python
def test_poll_job_projects_rows_into_the_existing_table_shape(tmp_path, monkeypatch):
    """poll_job must feed the untouched results_to_table, so the review queue
    keeps sorting flagged rows to the top."""
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)

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
    assert len(rows) == 2, "both the finished and the pending file must appear"
    # Index-agnostic on purpose: Task 13 adds three columns, and a test coupled
    # to a column position would break for a reason that has nothing to do with
    # the behaviour being asserted.
    assert any("REVIEW" in str(cell) for cell in rows[0]), "flagged rows sort first"
    assert csv_path and Path(csv_path).exists()
    assert json_path and Path(json_path).exists()


def test_poll_job_reports_an_unknown_id_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job("nope")

    assert "not found" in status_md.lower()
    assert rows == []
    assert csv_path is None and json_path is None


def test_poll_job_with_no_id_is_a_no_op(tmp_path, monkeypatch):
    """The UI timer fires before anything is queued; that must be harmless."""
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib

    from autoace import app as app_module
    from autoace import config as config_module

    importlib.reload(config_module)
    importlib.reload(app_module)

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job("")
    assert rows == []
    assert csv_path is None
```

Add this helper near the top of `tests/test_app.py`, after the existing imports (the file already imports what it needs for the original 8 tests):

```python
def _analysis(**overrides):
    from autoace.schema import CallAnalysis

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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_app.py -k poll_job -v`
Expected: FAIL — `AttributeError: module 'autoace.app' has no attribute 'poll_job'`

- [ ] **Step 3: Write the minimal implementation**

Append to `autoace/app.py`, after `enqueue_batch`:

```python
def _format_eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"~{minutes} min"
    return f"~{minutes // 60}h {minutes % 60:02d}m"


def poll_job(job_id: str):
    """Read-only projection of the store for the UI.

    Returns `(status_markdown, table_rows, csv_path, json_path, scoring_text)`
    - the same five outputs the old synchronous `run_batch` returned, so the
    Blocks wiring keeps the same shape.
    """
    empty = ("", [], None, None, "")
    if not job_id:
        return empty

    conn = jobs.connect()
    try:
        status = jobs.job_status(conn, job_id)
        if status is None:
            return (f"Job `{job_id}` not found. It may have expired.", [], None, None, "")

        results = jobs.job_results(conn, job_id)
        validation = json.loads(status.validation_json or "{}")
    finally:
        conn.close()

    if status.status == "failed" and status.total == 0:
        return (status.error or "Validation failed.", [], None, None, "")

    lines = [
        f"**Job** `{job_id}` — **{status.status}**",
        "",
        f"{status.finished} of {status.total} files finished "
        f"({status.done} analysed, {status.failed} failed).",
    ]
    if status.status in ("pending", "running"):
        if status.queue_position:
            lines.append(
                f"Queued behind {status.queue_position} other job(s). "
                f"Estimated wait: {_format_eta(status.eta_seconds)}."
            )
        else:
            lines.append(f"Estimated time remaining: {_format_eta(status.eta_seconds)}.")
    for problem in validation.get("problems", []):
        lines.append(f"- {problem}")

    table = results_to_table(results)

    out_dir = Path(tempfile.mkdtemp(prefix="autoace_out_"))
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    csv_path.write_text(results_to_csv(results), encoding="utf-8")
    json_path.write_text(results_to_json(results), encoding="utf-8")

    scoring_text = ""
    if status.manifest_labelled and status.status == "complete":
        manifest_names = validation.get("matched", [])
        predictions = {
            r.name: r.analysis for r in results if r.analysis is not None
        }
        if predictions and manifest_names:
            scoring_text = (
                "Scoring needs the uploaded manifest, which is removed with the "
                "job's workdir once processing finishes. Re-upload the same batch "
                "to score it, or use the eval harness described in the README."
            )

    return "\n".join(lines), table, str(csv_path), str(json_path), scoring_text
```

**Note on the scoring view:** the manifest lives in the workdir, which is deleted when the job completes. Task 14 fixes this properly by persisting the manifest; this task deliberately leaves an honest message rather than a silently missing feature.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: 13 passed (the original 8 plus 5 new).

- [ ] **Step 5: Commit**

```bash
git add autoace/app.py tests/test_app.py
git commit -m "feat: poll_job projects the job store into the existing UI shape"
```

---

## Task 12: Persist the manifest so the scoring view survives

**Files:**
- Modify: `autoace/jobs.py` (one schema column)
- Modify: `autoace/app.py`
- Modify: `tests/test_app.py` (append only)

The workdir is removed when a job finishes, taking the manifest with it — so the labelled-batch scoring view from spec §13 would break. Store the manifest CSV text on the job row instead.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:

```python
def test_scoring_view_renders_after_the_workdir_is_gone(tmp_path, monkeypatch):
    """The manifest lives in the workdir, which is deleted when the job
    finishes. Scoring a labelled batch must still work afterwards, so the
    manifest text is persisted on the job row."""
    monkeypatch.setenv("AUTOACE_DATA_DIR", str(tmp_path / "data"))

    import importlib
    import json as _json

    from autoace import app as app_module
    from autoace import config as config_module
    from autoace import jobs as jobs_module

    importlib.reload(config_module)
    importlib.reload(jobs_module)
    importlib.reload(app_module)

    from autoace.pipeline import FileResult

    expected = _analysis().model_dump_json()
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.ogg").write_bytes(b"stub")
    (batch / "m.csv").write_text(
        "name,result_json\n" + f'a.ogg,"{expected.replace(chr(34), chr(34) * 2)}"\n',
        encoding="utf-8",
    )

    job_id, _ = app_module.enqueue_batch(str(batch))

    conn = jobs_module.connect()
    try:
        claimed = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, claimed,
            FileResult(name=claimed.name, analysis=_analysis()),
        )
        assert jobs_module.job_status(conn, job_id).status == "complete"
    finally:
        conn.close()

    status_md, rows, csv_path, json_path, scoring = app_module.poll_job(job_id)

    assert scoring, "a labelled batch must render metrics"
    metrics = _json.loads(scoring)
    assert isinstance(metrics, dict) and metrics, "metrics must be real, not a message"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_app.py -k scoring_view -v`
Expected: FAIL — the assertion `metrics must be real, not a message` fails, because `poll_job` currently returns the placeholder text from Task 11.

- [ ] **Step 3: Add the column and persist the manifest**

In `autoace/jobs.py`, add one column to the `jobs` table in `_SCHEMA`:

```sql
    manifest_csv      TEXT,
```

Place it immediately after `validation_json TEXT NOT NULL,`. Then extend `enqueue` to accept and store it — change the signature and the INSERT:

```python
def enqueue(
    conn: sqlite3.Connection,
    *,
    workdir: Path | str,
    audio_by_name: dict[str, Path],
    validation_json: str,
    manifest_labelled: bool,
    manifest_csv: str | None = None,
) -> str:
```

```python
        conn.execute(
            "INSERT INTO jobs (job_id, status, created_at, expires_at, total_files,"
            " manifest_labelled, workdir, validation_json, manifest_csv)"
            " VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                now,
                now + JOB_TTL_SECONDS,
                len(audio_by_name),
                int(manifest_labelled),
                str(workdir),
                validation_json,
                manifest_csv,
            ),
        )
```

Add `manifest_csv` to `JobStatus` and populate it in `job_status`:

```python
@dataclass(frozen=True)
class JobStatus:
    job_id: str
    status: str
    total: int
    done: int
    failed: int
    running: int
    pending: int
    manifest_labelled: bool
    validation_json: str
    manifest_csv: str | None
    error: str | None
    queue_position: int
    eta_seconds: float
```

```python
        manifest_csv=job["manifest_csv"],
```

Insert that line in the `JobStatus(...)` construction immediately after `validation_json=job["validation_json"],`.

- [ ] **Step 4: Pass the manifest at enqueue time and score from it**

In `autoace/app.py`, inside `enqueue_batch`, read the manifest text before creating the job:

```python
        manifest_labelled = False
        manifest_csv = None
        if validation.manifest_path is not None:
            manifest_csv = validation.manifest_path.read_text(encoding="utf-8")
            manifest_labelled = any(
                row.expected is not None
                for row in load_manifest(str(validation.manifest_path))
            )

        job_id = jobs.enqueue(
            conn,
            workdir=workdir,
            audio_by_name=matched_audio,
            validation_json=validation_json,
            manifest_labelled=manifest_labelled,
            manifest_csv=manifest_csv,
        )
```

Then replace the scoring block in `poll_job` with a real computation. `load_manifest` takes a path, so write the persisted text to a temp file:

```python
    scoring_text = ""
    if status.manifest_labelled and status.status == "complete" and status.manifest_csv:
        # load_manifest reads a path, and the original manifest went away with
        # the workdir - so round-trip the persisted text through a temp file
        # rather than duplicating the CSV parsing here.
        manifest_dir = Path(tempfile.mkdtemp(prefix="autoace_manifest_"))
        manifest_file = manifest_dir / "manifest.csv"
        manifest_file.write_text(status.manifest_csv, encoding="utf-8")
        try:
            manifest_rows = load_manifest(str(manifest_file))
            predictions = {
                r.name: r.analysis for r in results if r.analysis is not None
            }
            if predictions:
                metrics = score_batch(manifest_rows, predictions)
                scoring_text = json.dumps(metrics, indent=2, default=str)
        finally:
            shutil.rmtree(manifest_dir, ignore_errors=True)
```

`shutil` and `tempfile` are already imported in `app.py`.

- [ ] **Step 5: Delete the stale test database**

The schema changed and `CREATE TABLE IF NOT EXISTS` will not add a column to an existing file. Any local store must be removed:

```bash
rm -rf _data
```

On the VM this is handled by the fact that the database is created fresh at first deploy. Note this in the README so nobody hits a confusing `no such column: manifest_csv`.

- [ ] **Step 6: Run the full app and jobs suites**

Run: `python -m pytest tests/test_app.py tests/test_jobs.py -v`
Expected: 14 app tests + 23 jobs tests pass. The original 8 app tests still unmodified.

- [ ] **Step 7: Commit**

```bash
git add autoace/jobs.py autoace/app.py tests/test_app.py
git commit -m "feat: persist the manifest so labelled-batch scoring survives cleanup"
```

---

## Task 13: Display all nine schema fields

**Files:**
- Modify: `autoace/app.py:45-55` (`TABLE_HEADERS`) and `results_to_table`
- Modify: `tests/test_app.py` (append only)

Brief §7 says: *"Results: Display the prediction for each audio file using the required output schema."* The table currently shows **six** of the nine schema fields. `background_noise_present`, `speaker_overlap_present`, and `long_silence_present` are exported to CSV and JSON but never displayed. §8 puts 10% of the total score on the dashboard, explicitly including "result review", so this is a scored gap rather than a cosmetic one.

The existing `test_review_flagged_rows_are_identifiable_in_the_table` asserts with `rows[0][0]` and an index-agnostic `any(...)`, so widening the table does not break it. Keep it that way — do not add position-coupled assertions.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:

```python
def test_table_displays_every_schema_field():
    """Brief section 7 requires the displayed prediction to use the required
    output schema. Three boolean fields were exported but never shown:
    background_noise_present, speaker_overlap_present, long_silence_present.
    """
    from autoace.app import SCHEMA_COLUMNS, TABLE_HEADERS, results_to_table

    rows = results_to_table(
        [
            FileResult(
                name="a.ogg",
                analysis=_analysis(
                    background_noise_present=True,
                    background_noise_type="office chatter",
                    background_noise_severity="low",
                    speaker_overlap_present=True,
                    long_silence_present=True,
                ),
            )
        ]
    )

    assert len(rows) == 1
    assert len(rows[0]) == len(TABLE_HEADERS), (
        "every row must have exactly one cell per header"
    )

    # Every schema field must be represented. `name` is the "file" column and
    # `confidence` is rendered rounded, so compare on the count of schema
    # fields rather than on header spelling.
    schema_fields = set(SCHEMA_COLUMNS) - {"name"}
    assert len(TABLE_HEADERS) >= len(schema_fields) + 1, (
        f"table shows {len(TABLE_HEADERS)} columns for {len(schema_fields)} "
        f"schema fields plus the filename"
    )

    flat = " ".join(str(cell) for cell in rows[0])
    assert "office chatter" in flat
    for label in ("noise", "overlap", "silence"):
        assert any(label in h.lower() for h in TABLE_HEADERS), (
            f"no column covers {label}"
        )


def test_table_shows_boolean_fields_as_readable_values():
    """A raw Python True/False in a Gradio dataframe reads poorly next to enum
    strings; yes/no keeps the row scannable."""
    from autoace.app import results_to_table

    rows = results_to_table(
        [
            FileResult(
                name="a.ogg",
                analysis=_analysis(
                    background_noise_present=False,
                    speaker_overlap_present=True,
                    long_silence_present=False,
                ),
            )
        ]
    )
    flat = [str(cell) for cell in rows[0]]
    assert "yes" in flat and "no" in flat


def test_error_rows_still_have_one_cell_per_header():
    """A failed file must not produce a short row - Gradio silently mangles a
    dataframe whose rows have inconsistent width."""
    from autoace.app import TABLE_HEADERS, results_to_table

    rows = results_to_table(
        [FileResult(name="bad.ogg", analysis=None, error="could not decode")]
    )
    assert len(rows[0]) == len(TABLE_HEADERS)
    assert any("ERROR" in str(cell) for cell in rows[0])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_app.py -k "every_schema_field or readable_values or one_cell_per_header" -v`
Expected: FAIL — `test_table_displays_every_schema_field` fails on `no column covers overlap`, and `test_table_shows_boolean_fields_as_readable_values` fails because no cell reads `yes`/`no`.

- [ ] **Step 3: Widen `TABLE_HEADERS`**

Replace the `TABLE_HEADERS` block in `autoace/app.py` (currently lines 45–55):

```python
# One column per schema field plus the filename, the review flag, and the tone
# path. Brief section 7 requires the DISPLAYED prediction to use the required
# output schema, so a field that is exported but not shown does not count -
# background_noise_present, speaker_overlap_present, and long_silence_present
# were previously missing here.
TABLE_HEADERS = [
    "file",
    "tone",
    "intensity",
    "noise?",
    "noise type",
    "severity",
    "quality",
    "overlap?",
    "long silence?",
    "confidence",
    "flag",
    "notes",
]
```

- [ ] **Step 4: Rewrite `results_to_table` to emit every field**

Replace the whole `results_to_table` function:

```python
def _yn(value: bool) -> str:
    """Booleans render as yes/no. A bare True next to enum strings like
    `slightly_impaired` is hard to scan in a 50-row table."""
    return "yes" if value else "no"


def results_to_table(results: list[FileResult]) -> list[list]:
    """Display rows for the review queue.

    Review-flagged (or outright failed) rows sort to the top, then ascending
    confidence - a degraded result must be visible, not buried at the bottom of
    a 50-row batch.

    Every row has exactly `len(TABLE_HEADERS)` cells, including error rows:
    Gradio mangles a dataframe with ragged rows rather than complaining.
    """
    rows: list[list] = []
    for result in results:
        if result.analysis is not None:
            data = result.analysis.model_dump(mode="json")
            confidence = data["confidence"]
            flagged = bool(result.review_flagged) or confidence < REVIEW_THRESHOLD
            rows.append(
                [
                    result.name,
                    data["emotional_tone"],
                    data["emotional_intensity"],
                    _yn(data["background_noise_present"]),
                    data["background_noise_type"],
                    data["background_noise_severity"],
                    data["audio_quality"],
                    _yn(data["speaker_overlap_present"]),
                    _yn(data["long_silence_present"]),
                    round(confidence, 3),
                    "REVIEW" if flagged else "",
                    result.reasoning,
                ]
            )
        else:
            # A pending file and a failed file are different things: a pending
            # row has no error yet and must not be labelled ERROR.
            state = "ERROR" if result.error else "queued"
            rows.append(
                [result.name] + [""] * 8 + [None, state, result.error or ""]
            )

    def sort_key(row):
        flagged = row[-2] in ("REVIEW", "ERROR")
        confidence = row[-3] if isinstance(row[-3], (int, float)) else -1.0
        return (0 if flagged else 1, confidence)

    rows.sort(key=sort_key)
    return rows
```

Note `sort_key` now indexes from the end (`row[-2]`, `row[-3]`) rather than hard-coded positions 7 and 6 — the previous version would silently sort on the wrong column the moment a column was inserted, which is exactly what just happened.

- [ ] **Step 5: Run the full app suite**

Run: `python -m pytest tests/test_app.py -v`
Expected: 17 passed. Crucially the original 8 still pass unmodified — `test_review_flagged_rows_are_identifiable_in_the_table` survives because it never indexed the flag column by position.

- [ ] **Step 5b: Delete the `poll_job` row-building duplication**

Task 11 hit a real defect and worked around it: `results_to_table`'s `else` branch labels **any** `analysis is None` row as `ERROR`, with no way to distinguish "genuinely failed" from "not processed yet". Since a pending row sorts with confidence `-1.0`, an unclaimed file outranked a real REVIEW row. The workaround built pending rows inline in `poll_job`:

```python
finished = [r for r in results if r.analysis is not None or r.error]
outstanding = [r for r in results if r.analysis is None and not r.error]
table = results_to_table(finished)
table += [[r.name, "", "", "", "", "", None, "PENDING", ""] for r in outstanding]
```

That hard-codes a **9-wide** row, and this task makes the table **12-wide** — so leaving it produces ragged rows, which Gradio mangles silently rather than reporting. The `state = "ERROR" if result.error else "queued"` branch added in Step 4 handles pending correctly *inside* `results_to_table`, so the duplication must go:

```python
    table = results_to_table(results)
```

Delete the `finished`/`outstanding` split and the inline row construction, and the comment block explaining them. Row shape now lives in exactly one place.

Update the Task 11 test `test_poll_job_projects_rows_into_the_existing_table_shape` **only** if it asserted on the literal string `"PENDING"`; the `queued` label replaces it. It was written index-agnostically, so the flagged-row assertion still holds.

- [ ] **Step 5c: Pin the row-width invariant across the seam**

```python
def test_poll_job_rows_all_match_the_header_width(tmp_path, monkeypatch):
    """Ragged rows are the failure mode Gradio hides: it renders a mangled
    dataframe rather than raising. Pending and finished rows must be equally
    wide, which is why row shape lives only in results_to_table."""
    app_module, jobs_module = _reload_modules(monkeypatch, tmp_path)

    from autoace.pipeline import FileResult

    batch = tmp_path / "batch"
    batch.mkdir()
    for name in ("a.ogg", "b.ogg", "c.ogg"):
        (batch / name).write_bytes(b"stub")
    (batch / "m.csv").write_text(
        "name,result_json\na.ogg,\nb.ogg,\nc.ogg,\n", encoding="utf-8"
    )

    job_id, _ = app_module.enqueue_batch(str(batch))

    conn = jobs_module.connect()
    try:
        done = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, done, FileResult(name=done.name, analysis=_analysis())
        )
        bad = jobs_module.claim_next(conn)
        jobs_module.complete(
            conn, bad,
            FileResult(name=bad.name, analysis=None, error="could not decode"),
        )
        # The third file stays pending.
    finally:
        conn.close()

    _, rows, *_ = app_module.poll_job(job_id)

    assert len(rows) == 3, "done, failed and pending must all appear"
    widths = {len(r) for r in rows}
    assert widths == {len(app_module.TABLE_HEADERS)}, (
        f"ragged rows: widths {widths} against "
        f"{len(app_module.TABLE_HEADERS)} headers"
    )
```

- [ ] **Step 6: Verify field coverage mechanically**

Run:

```bash
python -c "
from autoace.app import SCHEMA_COLUMNS, TABLE_HEADERS
print('schema fields:', len(SCHEMA_COLUMNS) - 1)
print('table columns:', len(TABLE_HEADERS))
assert len(TABLE_HEADERS) == (len(SCHEMA_COLUMNS) - 1) + 3, 'expect 9 fields + file + flag + notes'
print('every schema field is displayed')
"
```

Expected: `schema fields: 9`, `table columns: 12`, then `every schema field is displayed`.

- [ ] **Step 7: Commit**

```bash
git add autoace/app.py tests/test_app.py
git commit -m "fix: display all nine schema fields in the results table"
```

---

## Task 14: Manifest robustness for the hidden test set

**Files:**
- Modify: `autoace/eval.py:28-42` (`load_manifest`)
- Create: `tests/test_manifest.py`

Brief §7 states the CSV `result_json` column "may be empty or omitted from scoring input" for an unlabeled hidden test set, and that the manifest is what maps every audio file to a row. Two failure modes would take down an entire hidden-set batch, not one file:

1. **A UTF-8 BOM.** `load_manifest` opens with `encoding="utf-8"`, so a CSV saved from Excel — the overwhelmingly common case for an evaluator-supplied file — yields a first fieldname of `﻿name`. `record["name"]` then raises `KeyError`, and the whole batch dies before any inference. `utf-8-sig` strips the BOM.
2. **A missing or misnamed `name` column** raises a bare `KeyError: 'name'` that surfaces as an unhandled exception rather than a validation message.

This is squarely the brief's failure-handling requirement: a bad *file* must not fail the batch, and a bad *manifest* must at least explain itself.

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest.py`:

```python
"""Manifest parsing robustness.

The manifest is a single point of failure for a whole batch: unlike one bad
audio file, an unreadable manifest means nothing gets processed. These cases
are all things an evaluator-supplied CSV plausibly does.
"""

import pytest

from autoace.eval import load_manifest


def test_manifest_with_a_utf8_bom_is_readable(tmp_path):
    """Excel writes UTF-8 with a BOM by default. Opened as plain utf-8 the
    first fieldname becomes '\\ufeffname', so every row lookup raises KeyError
    and the entire batch fails before any inference runs."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\ncall_001.ogg,\n", encoding="utf-8-sig")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]
    assert rows[0].expected is None


def test_manifest_without_a_result_json_column_is_readable(tmp_path):
    """Brief section 7: for an unlabeled hidden test set result_json 'may be
    empty or omitted'. An omitted COLUMN must work, not just an empty cell."""
    path = tmp_path / "m.csv"
    path.write_text("name\ncall_001.ogg\ncall_002.ogg\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg", "call_002.ogg"]
    assert all(r.expected is None for r in rows)


def test_manifest_missing_the_name_column_raises_a_clear_error(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("filename,result_json\ncall_001.ogg,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="name"):
        load_manifest(str(path))


def test_manifest_ignores_blank_trailing_rows(tmp_path):
    """A trailing newline or a row of empty cells is common in hand-edited
    CSVs and must not become a row named ''."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\ncall_001.ogg,\n,\n\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]


def test_manifest_strips_whitespace_around_names(tmp_path):
    """A name with a stray space would not match the audio file on disk, and
    would be reported as 'missing audio' for a file that is right there."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\n  call_001.ogg  ,\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]


def test_manifest_reports_which_row_has_bad_json(tmp_path):
    """A malformed result_json cell must name the offending row rather than
    surfacing a bare JSONDecodeError with no context."""
    path = tmp_path / "m.csv"
    path.write_text(
        'name,result_json\ncall_001.ogg,"{not json}"\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="call_001.ogg"):
        load_manifest(str(path))
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL. `test_manifest_with_a_utf8_bom_is_readable` fails with `KeyError: 'name'`, and the clear-error tests fail because `KeyError` and `JSONDecodeError` are raised instead of `ValueError`.

- [ ] **Step 3: Harden `load_manifest`**

Replace `load_manifest` in `autoace/eval.py`:

```python
def load_manifest(path: str) -> list[ManifestRow]:
    """Read the brief's CSV manifest.

    An empty `result_json` cell - or an entirely absent `result_json` column -
    means the row is unlabelled (audio provided, no ground truth yet), so
    `expected` is None rather than raising. Brief section 7 allows exactly
    that for an unlabeled hidden test set.

    Opened as `utf-8-sig` because a manifest saved from Excel carries a BOM,
    which under plain `utf-8` turns the first fieldname into '\\ufeffname' and
    fails EVERY row lookup - taking down the whole batch before any inference
    runs. The manifest is a single point of failure in a way one bad audio
    file is not, so its errors name what went wrong.
    """
    rows: list[ManifestRow] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: manifest is empty")
        if "name" not in reader.fieldnames:
            raise ValueError(
                f"{path}: manifest has no 'name' column - found "
                f"{reader.fieldnames}. The brief requires a 'name' column "
                f"holding the exact audio filename."
            )

        for record in reader:
            name = (record.get("name") or "").strip()
            if not name:
                continue  # blank or trailing row

            raw = (record.get("result_json") or "").strip()
            if not raw:
                rows.append(ManifestRow(name=name, expected=None))
                continue

            try:
                expected = CallAnalysis(**json.loads(raw))
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ValueError(
                    f"{path}: row '{name}' has an unreadable result_json: {exc}"
                ) from exc
            rows.append(ManifestRow(name=name, expected=expected))
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: 6 passed.

- [ ] **Step 5: Confirm nothing that depended on the old behaviour broke**

Run: `python -m pytest tests/test_app.py tests/test_eval.py tests/test_jobs.py -v`
Expected: all pass. `validate_batch` and the scoring path both go through `load_manifest`, so this is the regression check that matters.

- [ ] **Step 6: Commit**

```bash
git add autoace/eval.py tests/test_manifest.py
git commit -m "fix: manifest survives a BOM, an omitted result_json column, and blank rows"
```

---

## Task 15: The asynchronous UI

**Files:**
- Modify: `autoace/app.py:328-383` (`build_app` and `__main__`)

There is no unit test for Blocks wiring — the existing `test_app_builds_without_launching` covers construction, and it must keep passing.

- [ ] **Step 1: Replace `build_app`**

```python
def build_app() -> gr.Blocks:
    """The UI. Construction only - never calls `.launch()`.

    Asynchronous by necessity: a 50-file batch is ~87 minutes, so the upload
    handler only enqueues, and a gr.Timer polls the job store. Closing the tab
    stops the timer but not the worker, which is why the job-ID box exists.
    """
    with gr.Blocks(title="AutoAce - Call Tone Review") as demo:
        gr.Markdown(
            "# AutoAce batch review\n"
            "Upload a folder or ZIP containing audio files plus one CSV "
            "manifest (`name,result_json`). Validation runs before any "
            "inference. Processing happens in the background at roughly "
            "**105 s per file**, so a 50-file batch takes about 90 minutes - "
            "**keep the job ID**, close the page if you like, and paste the ID "
            "back in to return to your results. Rows flagged for human review "
            "(low confidence or conflicting signals) sort to the top."
        )

        job_state = gr.State("")

        with gr.Row():
            upload = gr.File(
                label="Audio files + manifest CSV, or a single ZIP",
                file_count="multiple",
                type="filepath",
            )
        run_button = gr.Button("Validate & queue batch", variant="primary")

        with gr.Row():
            job_box = gr.Textbox(
                label="Look up a job",
                placeholder="Paste a job ID to resume watching it",
                scale=4,
            )
            lookup_button = gr.Button("Look up", scale=1)

        status = gr.Markdown(label="Status")
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            label="Results (review-flagged rows first)",
            interactive=False,
        )
        with gr.Row():
            csv_out = gr.File(label="Download results.csv")
            json_out = gr.File(label="Download results.json")
        scoring = gr.Textbox(
            label="Scoring metrics (shown when the manifest has labelled rows)",
            lines=12,
            interactive=False,
        )

        timer = gr.Timer(UI_POLL_SECONDS)

        poll_outputs = [status, table, csv_out, json_out, scoring]

        def _submit(files):
            job_id, message = enqueue_batch(files)
            return job_id, job_id, message

        run_button.click(
            fn=_submit,
            inputs=[upload],
            outputs=[job_state, job_box, status],
        )

        lookup_button.click(
            fn=lambda jid: (jid or "").strip(),
            inputs=[job_box],
            outputs=[job_state],
        ).then(fn=poll_job, inputs=[job_state], outputs=poll_outputs)

        timer.tick(fn=poll_job, inputs=[job_state], outputs=poll_outputs)

    return demo
```

- [ ] **Step 2: Replace the `__main__` block**

```python
if __name__ == "__main__":
    user = os.environ.get("AUTOACE_USER", "admin")
    password = os.environ.get("AUTOACE_PASSWORD")
    if not password:
        raise SystemExit(
            "AUTOACE_PASSWORD must be set in the environment - refusing to "
            "start a hosted dashboard with no login credential."
        )

    app = build_app()
    app.launch(
        # 127.0.0.1, not 0.0.0.0: Caddy terminates TLS and is the only listener
        # reachable from off-box, so a wrong firewall rule cannot expose the
        # app over plaintext HTTP.
        server_name=os.environ.get("AUTOACE_BIND", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(user, password),
        max_file_size=f"{MAX_UPLOAD_MB}mb",
    )
```

- [ ] **Step 3: Verify construction still works and the app suite passes**

Run: `python -m pytest tests/test_app.py -v`
Expected: 14 passed, including the unmodified `test_app_builds_without_launching`.

- [ ] **Step 4: Smoke-test the two processes together**

In one shell:

```bash
AUTOACE_DATA_DIR=./_data python -m autoace.worker
```

In another:

```bash
AUTOACE_DATA_DIR=./_data AUTOACE_USER=autoace AUTOACE_PASSWORD=test AUTOACE_BIND=127.0.0.1 python -m autoace.app
```

Open `http://127.0.0.1:7860`, log in as `autoace` / `test`, upload a ZIP of `reference/call_003.ogg` plus a two-column manifest. Expected: a job ID appears immediately, the status line updates on its own every 5 s, and the row completes in roughly 130–160 s. Then reload the page, paste the job ID into **Look up a job**, and confirm the results come back.

- [ ] **Step 5: Commit**

```bash
git add autoace/app.py
git commit -m "feat: asynchronous dashboard with timer polling and job lookup"
```

---

## Task 16: Deployment artifacts

**Files:**
- Create: `deploy/setup.sh`
- Create: `deploy/autoace-web.service`
- Create: `deploy/autoace-worker.service`
- Create: `deploy/Caddyfile`
- Create: `deploy/duckdns.sh`, `deploy/duckdns.service`, `deploy/duckdns.timer`

These are not unit-testable; verification is the deploy itself in Task 17.

- [ ] **Step 1: Write `deploy/autoace-web.service`**

```ini
[Unit]
Description=AutoAce dashboard (web)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/autoace/repo
EnvironmentFile=/etc/autoace.env
Environment=AUTOACE_DATA_DIR=/opt/autoace/data
Environment=HF_HOME=/opt/autoace/hf
Environment=AUTOACE_BIND=127.0.0.1
Environment=PORT=7860
ExecStart=/opt/autoace/venv/bin/python -m autoace.app
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write `deploy/autoace-worker.service`**

`OMP_NUM_THREADS=2` matches the measured 2-vCPU shape and stops OpenMP from spawning more threads than the box has, which on a 2-core VM causes contention rather than speed.

```ini
[Unit]
Description=AutoAce dashboard (worker)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/autoace/repo
EnvironmentFile=/etc/autoace.env
Environment=AUTOACE_DATA_DIR=/opt/autoace/data
Environment=HF_HOME=/opt/autoace/hf
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
ExecStart=/opt/autoace/venv/bin/python -m autoace.worker
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Write `deploy/Caddyfile`**

`{$AUTOACE_HOSTNAME}` is substituted by Caddy from its environment. Port 80 stays open permanently — ACME renewals need it, not just the first issuance.

```
{$AUTOACE_HOSTNAME} {
	encode gzip

	# Batch uploads are up to ~140 MB for 50 calls; 500 MB matches
	# MAX_UPLOAD_MB in autoace/config.py.
	request_body {
		max_size 500MB
	}

	# A 50-file batch is processed by the worker, not in a request, so these
	# only need to cover an upload and a poll - but a slow uplink pushing
	# 140 MB deserves room.
	reverse_proxy 127.0.0.1:7860 {
		transport http {
			read_timeout 600s
			write_timeout 600s
		}
	}
}
```

- [ ] **Step 4: Write `deploy/duckdns.sh`**

```bash
#!/usr/bin/env bash
# Keep the DuckDNS A record pointed at this instance.
#
# Oracle public IPs are ephemeral unless reserved, and a changed IP breaks both
# DNS and the certificate. Reserving the IP is the real fix; this is the belt to
# that braces.
set -euo pipefail

: "${DUCKDNS_DOMAIN:?set DUCKDNS_DOMAIN (the subdomain only, no .duckdns.org)}"
: "${DUCKDNS_TOKEN:?set DUCKDNS_TOKEN}"

response=$(curl -fsS \
	"https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$response" != "OK" ]; then
	echo "duckdns update failed: ${response}" >&2
	exit 1
fi
echo "duckdns update OK"
```

- [ ] **Step 5: Write `deploy/duckdns.service` and `deploy/duckdns.timer`**

`deploy/duckdns.service`:

```ini
[Unit]
Description=Refresh the AutoAce DuckDNS record

[Service]
Type=oneshot
EnvironmentFile=/etc/autoace.env
ExecStart=/opt/autoace/repo/deploy/duckdns.sh
```

`deploy/duckdns.timer`:

```ini
[Unit]
Description=Refresh the AutoAce DuckDNS record every 15 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
```

- [ ] **Step 6: Write `deploy/setup.sh`**

```bash
#!/usr/bin/env bash
# Provision an Oracle Cloud Always Free ARM instance for the AutoAce dashboard.
#
# Idempotent: safe to re-run after a code change or a failed attempt.
#
# Prerequisites you must do by hand first, because they are not scriptable from
# inside the box:
#   1. VCN ingress rules allowing TCP 80 and 443.
#   2. A reserved (not ephemeral) public IP.
#   3. /etc/autoace.env populated - see the block printed at the end.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/abdulsami123/emotion-detection.git}"
ROOT=/opt/autoace

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
	python3.12-venv python3-pip git ffmpeg libsndfile1 curl debian-keyring \
	debian-archive-keyring apt-transport-https

echo "==> caddy"
if ! command -v caddy >/dev/null 2>&1; then
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' |
		sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' |
		sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
	sudo apt-get update -qq
	sudo apt-get install -y -qq caddy
fi

echo "==> open 80/443 on the host firewall"
# Oracle's Ubuntu images ship netfilter rules that DROP everything except SSH.
# The VCN security list looking correct while the host still refuses the
# connection is the single most common way this deploy fails.
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null ||
	sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null ||
	sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

echo "==> 4 GiB swap"
# Insurance for the transient memory peak: 5.67 GiB measured peak against a
# 3.91 GiB steady state. Swapping is slow, but slow beats an OOM kill.
if ! sudo swapon --show | grep -q /swapfile; then
	sudo fallocate -l 4G /swapfile
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
	echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> directories"
sudo mkdir -p "${ROOT}"/{data,hf}
sudo chown -R ubuntu:ubuntu "${ROOT}"

echo "==> repo"
if [ -d "${ROOT}/repo/.git" ]; then
	git -C "${ROOT}/repo" pull --ff-only
else
	git clone "${REPO_URL}" "${ROOT}/repo"
fi

echo "==> venv"
if [ ! -x "${ROOT}/venv/bin/python" ]; then
	python3.12 -m venv "${ROOT}/venv"
fi
"${ROOT}/venv/bin/pip" install --quiet --upgrade pip
"${ROOT}/venv/bin/pip" install --quiet -r "${ROOT}/repo/requirements.txt"

echo "==> systemd units"
sudo cp "${ROOT}"/repo/deploy/autoace-web.service /etc/systemd/system/
sudo cp "${ROOT}"/repo/deploy/autoace-worker.service /etc/systemd/system/
sudo cp "${ROOT}"/repo/deploy/duckdns.service /etc/systemd/system/
sudo cp "${ROOT}"/repo/deploy/duckdns.timer /etc/systemd/system/
sudo chmod +x "${ROOT}"/repo/deploy/duckdns.sh

if [ ! -f /etc/autoace.env ]; then
	cat <<'ENVEOF'

  /etc/autoace.env does not exist yet. Create it with mode 0600:

    sudo install -m 600 /dev/null /etc/autoace.env
    sudo tee /etc/autoace.env >/dev/null <<'EOF'
    OPENAI_API_KEY=sk-...
    AUTOACE_USER=autoace
    AUTOACE_PASSWORD=<choose a strong password>
    AUTOACE_HOSTNAME=<subdomain>.duckdns.org
    DUCKDNS_DOMAIN=<subdomain>
    DUCKDNS_TOKEN=<token from duckdns.org>
    EOF

  Then re-run this script.

ENVEOF
	exit 1
fi

# shellcheck disable=SC1091
set -a && . /etc/autoace.env && set +a
: "${AUTOACE_HOSTNAME:?AUTOACE_HOSTNAME must be set in /etc/autoace.env}"

echo "==> caddy config"
sudo cp "${ROOT}/repo/deploy/Caddyfile" /etc/caddy/Caddyfile
sudo mkdir -p /etc/systemd/system/caddy.service.d
sudo tee /etc/systemd/system/caddy.service.d/override.conf >/dev/null <<EOF
[Service]
Environment=AUTOACE_HOSTNAME=${AUTOACE_HOSTNAME}
EOF

echo "==> warm the model cache (~5 GiB, once)"
# Done before the units start so the first real upload is not the thing that
# waits for a 5 GiB download.
#
# bart-large-mnli is warmed too even though it only loads on the OpenAI
# fallback path: if that path is ever taken, it must not also have to download
# 1.6 GB first.
cd "${ROOT}/repo"
sudo -u ubuntu env HF_HOME="${ROOT}/hf" "${ROOT}/venv/bin/python" - <<'PY'
# asr and ser BOTH define _load_model, so these must be aliased.
from autoace.asr import _load_model as load_whisper
from autoace.diarize import _load_encoder
from autoace.ser import _load_model as load_ser
from autoace.tagging import _load_ast
from autoace.tone_nli import _load_classifier
from autoace.quality import _load_squim

for label, loader in (
    ("faster-whisper large-v3-turbo", load_whisper),
    ("ECAPA speaker encoder", _load_encoder),
    ("audeering SER", load_ser),
    ("AST AudioSet tagger", _load_ast),
    ("bart-large-mnli (fallback)", _load_classifier),
    ("TorchAudio SQUIM", _load_squim),
):
    print(f"warming {label} ...", flush=True)
    loader()
print("model cache warm")
PY

echo "==> start everything"
sudo systemctl daemon-reload
sudo systemctl enable --now duckdns.timer
sudo systemctl restart duckdns.service
sudo systemctl enable --now autoace-worker.service
sudo systemctl enable --now autoace-web.service
sudo systemctl enable --now caddy

echo
echo "Done. https://${AUTOACE_HOSTNAME}"
echo "Logs:  journalctl -u autoace-worker -f"
```

**The warm-up block calls private, `lru_cache`d loaders.** These names were read from the current source and are correct as written:

| Module | Loader |
|---|---|
| `autoace/asr.py:54` | `_load_model` |
| `autoace/diarize.py:43` | `_load_encoder` |
| `autoace/ser.py:94` | `_load_model` (**same name as asr's — must be aliased**) |
| `autoace/tagging.py:106` | `_load_ast` |
| `autoace/tone_nli.py:45` | `_load_classifier` |
| `autoace/quality.py:161` | `_load_squim` |

They are private, so if a future refactor renames one the warm-up breaks loudly at deploy time rather than silently — which is the right failure. Silero VAD is not warmed here: `vad.py` pulls it via `torch.hub.load` and it is ~2 MB, so the first call downloads it imperceptibly.

- [ ] **Step 7: Make the scripts executable and commit**

```bash
chmod +x deploy/setup.sh deploy/duckdns.sh
git add deploy/
git commit -m "feat: Oracle ARM deployment artifacts (systemd, Caddy, DuckDNS)"
```

---

## Task 17: Deploy and re-measure on ARM

**Files:** none — this task runs on the VM.

- [ ] **Step 1: Confirm the prerequisites**

VCN ingress for TCP 80 and 443; a **reserved** public IP; a DuckDNS subdomain pointed at it; `ssh ubuntu@<ip>` works.

- [ ] **Step 2: Run setup**

```bash
git clone https://github.com/abdulsami123/emotion-detection.git /tmp/autoace-bootstrap
bash /tmp/autoace-bootstrap/deploy/setup.sh
```

Expected: it exits asking for `/etc/autoace.env` on the first run. Create the file, re-run.

- [ ] **Step 3: Run the CI-safe suite on ARM**

```bash
cd /opt/autoace/repo
/opt/autoace/venv/bin/python -m pytest tests/test_jobs.py tests/test_app.py -v
```

Expected: all pass. This is what confirms nothing depended on Python 3.13 — the VM runs 3.12.

- [ ] **Step 4: Re-measure peak RSS on ARM**

Spec §11.2 flags the 5.67 GiB figure as Windows-only. Measure the real thing:

```bash
sudo systemctl stop autoace-worker
/usr/bin/time -v /opt/autoace/venv/bin/python -c "
from autoace.pipeline import analyse_file
from autoace.config import reference_call
for n in ('call_001.ogg','call_002.ogg','call_003.ogg'):
    analyse_file(reference_call(n))
" 2>&1 | grep -E 'Maximum resident|Elapsed'
sudo systemctl start autoace-worker
```

`Maximum resident set size` is in kilobytes. Record it, and the elapsed time, in `docs/MEMO.md`. Requires `reference/` present on the VM — copy the audio up over `scp` and **do not commit it**.

- [ ] **Step 5: Verify the service end to end**

Open `https://<subdomain>.duckdns.org`, log in, upload a ZIP with one call plus a manifest. Confirm: a valid certificate, the job ID appears at once, the table updates on its own, the row completes, `results.csv` downloads, and the audio is gone from `/opt/autoace/data/uploads/`.

- [ ] **Step 6: Verify interruption handling for real**

```bash
# start a batch, then mid-file:
sudo systemctl restart autoace-worker
journalctl -u autoace-worker -n 20
```

Expected: a `startup reconcile: 1 requeued` warning, and the file is reprocessed rather than lost.

- [ ] **Step 7: Commit the measurements**

```bash
git add docs/MEMO.md
git commit -m "docs: record ARM peak RSS and deployed latency"
```

---

## Task 18: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/MEMO.md`

- [ ] **Step 1: Replace the README's "Run the dashboard" section**

```markdown
## Run the dashboard

Two processes share a SQLite job store, because a 50-file batch takes ~87
minutes and no HTTP request survives that:

```bash
export AUTOACE_DATA_DIR=./_data
python -m autoace.worker &                       # claims files, runs the pipeline
AUTOACE_USER=autoace AUTOACE_PASSWORD=<password> python -m autoace.app
```

The web process binds `127.0.0.1:7860` by default (`AUTOACE_BIND=0.0.0.0` to
change it) and refuses to start without a password. Upload a ZIP containing
audio plus one CSV manifest; the batch is validated before any inference runs,
one bad file cannot fail the batch, and results download as CSV and JSON with
original filenames preserved.

**Keep the job ID.** Processing is asynchronous — close the page and paste the
ID into **Look up a job** to come back to the results, which are kept for 7
days. Audio is deleted as soon as each file is analysed.

Rows flagged **REVIEW** — confidence below 0.60, or conflicting evidence —
sort to the top.

If you see `no such column: manifest_csv`, you have a job store from before the
schema changed: `rm -rf _data`.

## Deploying

`deploy/setup.sh` provisions an Oracle Cloud Always Free ARM instance
(`VM.Standard.A1.Flex`, 2 OCPU / 12 GB, Ubuntu 24.04) with both systemd units,
Caddy for TLS on a DuckDNS hostname, a 4 GiB swapfile, and the model cache
warmed. See `docs/superpowers/specs/2026-08-19-hosted-dashboard-design.md` for
why that host and not a PaaS free tier — measured peak worker memory is 5.67
GiB, which is 11x Render's free tier.
```

- [ ] **Step 2: Add a hosting section to `docs/MEMO.md`**

Cover, with the measured numbers:

- **Architecture:** SQLite job store, two systemd units, per-file checkpointing, one worker.
- **Latency:** 58.2 / 58.7 / 129.8 s per call at 16 threads; 1.11–1.24x slower capped at 2 threads; ~87 min per 50 files. Replace with the ARM figures from Task 17.
- **Memory:** 5.67 GiB peak, 3.91 GiB steady (Windows working set); ARM figure from Task 17. Note that the peak is the NLI-fallback path and that we size for it so an OpenAI outage cannot OOM-kill the worker.
- **Why not a PaaS free tier:** the §2.1 table — Render free is 512 MB, HF Gradio Spaces now need a paid plan, ZeroGPU gives 5 GPU-minutes/day.
- **Privacy (brief §5):** audio unlinked per file; results carry no audio; TLS terminates in Caddy on our own VM so no third party sees plaintext; DuckDNS is DNS only; OpenAI receives the annotated transcript, not audio.
- **CI limitation:** GitHub Actions can only run tests needing neither `reference/` nor the weights — attributable to §5, not to thin coverage.
- **Known limitations:** the §11 table, especially the FIFO queue, the retry cap trading a failed file for a broken crash loop, and Oracle ARM capacity.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/MEMO.md
git commit -m "docs: hosted dashboard usage, deployment, and privacy posture"
```

---

## Self-review notes

**Spec coverage.** Every section maps to a task: §2 config (T1) · §3 architecture (T2, T9) · §4 data model (T2, T12) · §5 lifecycle (T3–T7) · §6.1–6.2 web split (T10, T11) · §6.3 timer and lookup (T15) · §6.4 queue position (T8, T11) · §6.5 auth (T15) · §7.1–7.2 failure isolation (T5, T10) · §7.3 reconcile (T6) · §7.4 retry cap (T6, T9) · §8 deployment (T16, T17) · §9 configuration (T1) · §10 testing (throughout) · §10.1 CI limitation (T18) · §11 limitations (T18).

**Brief §7 coverage, audited against the PDF rather than against the spec's paraphrase of it.** Every displayed-output requirement now maps to a task:

| Brief §7 requirement | Task |
|---|---|
| Hosting, URL, working login credentials, available through the evaluation period | T15 (auth), T16–T17 (deploy) |
| Upload a folder **or** ZIP with multiple clips and one CSV manifest | existing `_prepare_workdir`, reused |
| Audio at folder root plus a CSV; each filename corresponds to one row | existing `validate_batch`, reused |
| CSV `name` + `result_json`; `result_json` may be **empty or omitted** | **T14** |
| Validate the batch, clearly report missing or unmatched files | T10 |
| Process each valid clip; show batch progress or completion status | T9, T11, T15 |
| **Display the prediction using the required output schema** | **T13** |
| Downloadable CSV or JSON preserving the original filename | existing `results_to_csv` / `results_to_json`, reused |
| A single malformed or unsupported file must not fail the batch; identify which file failed and why | T5, T10, T11 |

Two of those were genuinely missing and are the reason T13 and T14 exist — see below.

**Two brief §7 requirements were not met by the existing code, and would not have been caught by anything in tasks 1–12:**

1. **The results table displayed only six of the nine schema fields.** `background_noise_present`, `speaker_overlap_present`, and `long_silence_present` were exported to CSV and JSON but never shown on screen. Brief §7 requires the *displayed* prediction to use the required output schema, and §8 scores "result review" inside the 10% dashboard weighting — so an exported-but-invisible field does not count. Task 13. The old `sort_key` also indexed columns 6 and 7 by position, so it would have silently sorted on the wrong column the moment anything was inserted; it now indexes from the end.
2. **A manifest with a UTF-8 BOM would have failed the entire batch.** `load_manifest` opened with `encoding="utf-8"`, so an Excel-saved CSV — the likely form of an evaluator-supplied file — turns the first fieldname into `﻿name` and makes every `record["name"]` raise `KeyError` before any inference runs. Brief §7 also permits `result_json` to be *omitted entirely* for an unlabeled hidden test set, not merely left empty. Task 14 covers both, plus blank trailing rows and whitespace-padded names, and turns bare `KeyError`/`JSONDecodeError` into messages that name the offending row.

**Two further gaps found and closed while writing:**

1. **The scoring view would have silently broken.** `_finalise_if_done` deletes the workdir, and the manifest lives in it — so the labelled-batch metrics from spec §13 would vanish exactly when a job completed. Task 12 persists the manifest on the job row. Task 11 deliberately ships an honest placeholder message first rather than a silently missing feature, so the defect is visible in the interim rather than hidden.
2. **`reconcile` could strand a job at `running` forever.** If reconcile is what exhausts the last file's attempts, nothing else would ever call `_finalise_if_done` for that job — there would be no `complete`/`fail` call left to trigger it. Fixed by finalising every touched job at the end of `reconcile`, and pinned by `test_reconcile_finalises_a_job_whose_last_file_it_failed`.

**Type consistency checked.** `ClaimedFile(job_id, name, audio_path, attempts)` and `JobStatus` are used with the same field names everywhere. `JobStatus` gains `manifest_csv` in Task 12, and both the dataclass and its construction site are updated in the same step. `complete(conn, claimed, result)` and `fail(conn, claimed, error)` take a `ClaimedFile`, not a `(job_id, name)` pair, consistently across `jobs.py` and `worker.py`.

**Loader names verified, not assumed.** The Task 16 warm-up block calls six private `lru_cache`d loaders; all six names were read from the current source and are tabulated in that task. `autoace.asr._load_model` and `autoace.ser._load_model` collide, so the block aliases them — an earlier draft would have failed at deploy time with a silently shadowed import.
