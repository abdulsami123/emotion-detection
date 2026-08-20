# AutoAce Hosted Dashboard — Design

Supersedes §13 of `2026-08-15-voice-tone-noise-design.md`, which specified a synchronous
Gradio dashboard and left deployment as "a VM under our control (Fly.io / Render with private
storage)". Both halves of that are now wrong: the synchronous model cannot survive a 50-file
batch, and every PaaS free tier is an order of magnitude too small to run this pipeline.

Everything else in the original spec — the 9-field schema, both branches, `config.py` as the
single source of thresholds — is unchanged. This document covers only the hosting layer.

---

## 1. Objective

Brief §7 asks for a hosted dashboard with authenticated login that accepts a batch upload,
validates it, processes it with visible progress, isolates per-file failures, exposes a review
queue, and exports results. §13 already enumerates those behaviours and `autoace/app.py`
already implements them **synchronously**. The gap is operational:

- A 50-file batch takes ~87 minutes (§2). No HTTP request survives that, and no browser tab
  should have to stay open for it.
- The evaluator must be able to close the tab and come back to results.
- A crash, redeploy, or OOM mid-batch must not discard completed work.

So the deliverable is: **the same behaviours, made asynchronous and durable, on hardware that
costs nothing.**

### 1.1 Non-goals

- Multi-tenant accounts. One shared credential, issued to the evaluator.
- Horizontal scale. One worker; §2 shows more cores buy almost nothing.
- Durable-workflow engines (Temporal, Inngest). Considered and rejected in §3.4.
- Object storage (S3). Considered and rejected in §3.4.

---

## 2. Measured constraints

Every number here is measured on the three provided calls, not estimated. This matters because
**every cost figure in this project that came from arithmetic instead of measurement has been
wrong** — ASR 2.4x pessimistic, SQUIM 19x and AST 31x optimistic, LLM input tokens 3.3x under.

| Quantity | Value | Provenance |
|---|---|---|
| Peak resident memory, one worker | **5.67 GiB** | MEASURED — `GetProcessMemoryInfo` peak working set, all three calls in one process |
| Steady-state resident memory | **3.91 GiB** | MEASURED — same run, after the third call |
| Per-file latency, 16 threads | 58.2 / 58.7 / 129.8 s | MEASURED — calls 001 / 002 / 003 |
| Per-file latency, 2 threads | 64.6 s (1.11x) / 160.8 s (1.24x) | MEASURED — calls 001 / 003, `OMP_NUM_THREADS=2` |
| 50-file batch projection | **~87 min** | DERIVED — mean per-file x 1.18 thread penalty x 50 |
| Python dependency footprint | ~1.5 GiB | MEASURED — torch 527M, gradio 193M, llvmlite 117M, scipy 115M, transformers 97M, ctranslate2 60M |
| Model weight cache | ~5 GiB | ESTIMATED — not isolated from a 60 GiB shared HF cache; verify on first warm-up |

Three consequences drive the whole design:

**(a) Core count is nearly irrelevant.** Capping the pipeline to 2 threads costs only 1.11–1.24x.
The pipeline is sequential and memory-bandwidth-bound, not CPU-parallel. `WhisperModel` is
constructed without `cpu_threads`, so CTranslate2 derives its intra-op count from
`OMP_NUM_THREADS` — the cap was genuinely applied, not silently ignored.

**(b) RAM is the only binding constraint**, at ~6 GiB.

**(c) The 5.67 GiB peak is the *fallback* path.** `bart-large-mnli` (~1.6 GB) loads only inside
the `except` handler in `pipeline.py` — it is not a co-voter; the two voters are the LLM and the
dimensional SER. The deployed happy path should therefore be materially lighter. **We still size
for 5.67 GiB**, because an OpenAI outage must not OOM-kill the worker. See §7.4.

> **Caveat on the memory figure.** Windows peak working set is not Linux RSS. The magnitude is
> right; the exact number will differ on the VM. Re-measure there before trusting it, and note
> that an earlier version of this probe reported `0.00 GiB` across the board because
> `GetCurrentProcess` returned a pseudo-handle that ctypes truncated to 32 bits — a silent API
> failure that looked exactly like a low memory footprint.

### 2.1 Free tiers that were refuted, with the reason

Recorded so nobody re-proposes them.

| Option | Why it fails |
|---|---|
| Render free | 512 MB RAM, no persistent disk. Off by 11x. |
| Hugging Face Spaces free | Gradio/Docker Spaces now **require a paid plan** (PRO for personal accounts). Static Spaces are free but HTML-only. |
| HF ZeroGPU (the free-account exception) | 5 minutes of GPU **per day** on a free account; `@spaces.GPU` is request-scoped and cannot host a long-running worker. |
| Railway | No free tier; $5 trial credit only. |
| Fly.io / Koyeb / Northflank | Free allowances are 256–512 MB. |
| AWS / GCP / Azure free VMs | 1 GB micro instances. |
| Shrink peak to fit a small tier | Freeing models between stages could reach ~2 GB, but **no free tier offers 2 GB either**, so it buys nothing on its own. |

**Selected: Oracle Cloud Always Free** — ARM Ampere A1, up to 4 OCPU / 24 GB, free indefinitely.
See §8.

---

## 3. Architecture

```
                    +------------------------------+
   browser -------->| autoace-web.service          |
   (Gradio auth)    |   Gradio Blocks              |
                    |   upload -> validate -> queue|
                    |   gr.Timer -> poll           |
                    +--------------+---------------+
                                   |  SQLite (WAL)
                                   |  /opt/autoace/data/jobs.db
                    +--------------+---------------+
                    | autoace-worker.service       |
                    |   claim -> analyse_file ->   |
                    |   record -> unlink audio     |
                    +------------------------------+
```

| Module | Responsibility |
|---|---|
| `autoace/jobs.py` | **New.** SQLite job store: schema, enqueue, claim, complete, fail, status, reconcile, expire. No audio processing, no HTTP, no Gradio import. |
| `autoace/worker.py` | **New.** The drain loop. Claims one file, calls `analyse_file`, records the result, unlinks the audio. |
| `autoace/app.py` | **Refactored.** `run_batch` splits into enqueue + poll. |
| `autoace/pipeline.py` | **Unchanged.** `analyse_file` is already the correct unit of work and already never raises. |

### 3.1 Two systemd units, not a forked child

An earlier draft had the web process fork the worker as a child, because a PaaS service runs one
process and the web process must keep answering health checks. On a VM that constraint does not
exist, and two units are strictly better: independent `Restart=always`, independent journald
streams, and the worker can be restarted without dropping the UI.

A thread was never an option — openSMILE and the DSP paths do not reliably release the GIL, so a
worker thread would stall the UI.

### 3.2 Why SQLite is sufficient

The two processes are on one machine, so a client/server database buys nothing. WAL mode gives
one writer (worker) and concurrent readers (web) without blocking. A 50-file batch is 1 job row
plus 50 file rows — on the order of 50 KB of metadata. The store is a queue, a checkpoint log,
and the results table at once.

### 3.3 Why the unit of work is one file

`analyse_file` takes 58–161 s and is already fail-isolated. Checkpointing at that granularity
means the worst case for any interruption is one file re-run. That is what makes heavier
machinery unnecessary.

### 3.4 Rejected: durable workflow engines and S3

A durable workflow engine (Temporal, Inngest) implies a worker on a **different machine** from
the uploader, which in turn *requires* object storage to hand the audio over. That is a real
architecture — it is simply not this one. Neither buys anything here:

- Per-file SQLite checkpointing already bounds interruption loss to one file (~90 s).
- The brief does not ask for durable resumption.
- Throughput is fixed at ~87 min per 50 files by the pipeline itself (§2a), not by the queue
  technology. No queue makes it faster.

`jobs.py` exposes `claim_next()` / `complete()` / `fail()` and nothing else, so a different
executor could replace `worker.py` later without touching the web layer or the pipeline. That
seam is the hedge; building the engine now is not.

---

## 4. Data model

```sql
CREATE TABLE jobs (
    job_id            TEXT PRIMARY KEY,   -- uuid4
    status            TEXT NOT NULL,      -- pending|running|complete|failed|expired
    created_at        REAL NOT NULL,
    expires_at        REAL NOT NULL,
    total_files       INTEGER NOT NULL,
    manifest_labelled INTEGER NOT NULL,   -- 0/1: drives the §13 scoring view
    workdir           TEXT NOT NULL,
    validation_json   TEXT NOT NULL,      -- BatchValidation, so the pre-flight
                                          -- report survives a page reload
    error             TEXT
);

CREATE TABLE files (
    job_id         TEXT NOT NULL,
    name           TEXT NOT NULL,         -- original filename, preserved for export
    status         TEXT NOT NULL,         -- pending|running|done|failed
    attempts       INTEGER NOT NULL DEFAULT 0,
    audio_path     TEXT,                  -- NULL once unlinked
    result_json    TEXT,                  -- CallAnalysis
    reasoning      TEXT,                  -- tone path; makes a silent fallback visible
    review_flagged INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    started_at     REAL,
    finished_at    REAL,
    PRIMARY KEY (job_id, name),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX files_claim ON files(status, job_id);
```

`PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`.

Storing `reasoning` per row is load-bearing, not diagnostic clutter: a result produced via the
NLI fallback is capped at 0.55 confidence, and without the recorded path a silent fallback is
operationally indistinguishable from success.

---

## 5. Job lifecycle

```
upload --> extract --> validate_batch --> [invalid] --> jobs.status=failed, 0 file rows
                            |
                        [valid]
                            v
                 1 job row + N file rows (pending)  --> job_id returned immediately
                            |
      worker: claim_next() -+ UPDATE ... SET status='running', attempts=attempts+1
                            |        WHERE status='pending' ORDER BY rowid LIMIT 1
                            v
                     analyse_file(path)
                            |
            +---------------+---------------+
       [FileResult]                    [exception]
            v                                v
   status=done, result_json,        attempts<2  -> status=pending (retry)
   reasoning, review_flagged        attempts>=2 -> status=failed
   unlink(audio_path)                        unlink(audio_path)
            +---------------+---------------+
                            v
            no pending/running files left for job
                            v
               jobs.status=complete, rmtree(workdir)
```

**Claiming is exclusive** via a single `UPDATE ... WHERE status='pending'` inside a transaction,
so the design is already correct for more than one worker even though we run one.

**Audio is unlinked as soon as its file row is recorded** — not at end of job. Confidential audio
lives on disk for the ~90 s it is being processed and no longer.

---

## 6. Web layer

### 6.1 What is reused unchanged

`validate_batch`, `results_to_csv`, `results_to_json`, `results_to_table`, `BatchValidation`, and
`_prepare_workdir` are unchanged. Their 8 tests must still pass without modification — that is
the regression boundary for this refactor.

### 6.2 What changes

`run_batch(upload, progress)` — which loops over files calling `analyse_file` inline and drives
`gr.Progress()` — is replaced by:

- `enqueue_batch(upload) -> job_id` — extract, validate, insert rows, return.
- `poll_job(job_id) -> (status_md, table_rows, csv_path, json_path, scoring_md)` — read-only
  projection of the store, shaped for the existing components.

`results_to_*` operate on `list[FileResult]`, so `poll_job` rehydrates `FileResult` objects from
the store. This keeps the export and review-queue code — including REVIEW-first sorting —
completely untouched.

Rehydration is lossless rather than approximate: `FileResult` has exactly five fields and each
maps onto one `files` column.

| `FileResult` field | `files` column |
|---|---|
| `name: str` | `name` |
| `analysis: CallAnalysis \| None` | `result_json` (NULL on failure) |
| `error: str \| None` | `error` |
| `reasoning: str` | `reasoning` |
| `review_flagged: bool` | `review_flagged` |

The table is therefore a persisted `FileResult`, which is why the export path needs no changes at
all — not merely few.

### 6.3 Progress and returning later

A `gr.Timer` refreshes `poll_job` on an interval. Closing the browser stops the timer but not the
worker, and all state is in SQLite.

A **job-ID lookup box** restores the view for any non-expired job. This is what makes "results
kept until download" real rather than aspirational, and it is the answer to an ~87-minute batch:
the evaluator uploads, takes the ID, and comes back.

### 6.4 Queue position

One worker means jobs are FIFO across users; a second batch waits behind the first for up to ~87
minutes. The status panel shows queue position and an estimate derived from
`MEAN_SECONDS_PER_FILE`, so a waiting job does not look hung. Surfacing this is deliberate — a
silent wait is the failure mode.

### 6.5 Auth

`gr.Blocks(auth=(AUTOACE_USER, AUTOACE_PASSWORD))`, refusing to start without a password (already
implemented). The tunnel hostname is public; the app behind it is not.

---

## 7. Failure isolation

Brief §7 calls this out specifically, and these are four genuinely different failures.

### 7.1 One bad file
`analyse_file` returns a `FileResult` carrying `.error`. The row is marked `failed`, the batch
continues, and the file appears in the table **and in the CSV/JSON exports** rather than being
dropped.

### 7.2 A bad batch
Missing manifest rows, audio without a row, unsupported extensions, corrupt ZIP, absent manifest.
Reported by `validate_batch` **before any row is inserted and before any inference runs**.

### 7.3 Interruption
Worker crash, VM reboot, or `systemctl restart` mid-file. Completed rows are already durable.
On worker start, `reconcile()` resets stale `running` rows to `pending`. Worst case is one
in-flight file.

### 7.4 A crash that repeats — the retry cap
This was added because of §2c. If the NLI fallback pushes the worker past available RAM, the OOM
killer takes it mid-file; `reconcile()` returns that file to `pending`; the worker restarts and
dies on the same file. **That is an infinite crash loop, and per-file checkpointing alone does not
prevent it.**

`attempts` is incremented at claim time, not at failure time, so a process killed *without*
running any Python still counts. `MAX_ATTEMPTS = 2`: a file that dies twice becomes `failed` with
`error` recording the retry exhaustion, and the batch proceeds.

---

## 8. Deployment — Oracle Cloud Always Free

### 8.1 Instance

`VM.Standard.A1.Flex`, Ubuntu 24.04 (aarch64), **2 OCPU / 12 GB**.

2 OCPU rather than the full free 4 OCPU / 24 GB allowance because §2a shows the extra cores buy
1.1–1.2x, while smaller shape requests are markedly more likely to be granted — Oracle ARM
returns `Out of host capacity` frequently in popular regions. 12 GB is ~2x the measured peak.

Default 50 GB boot volume is sufficient: ~1.5 GiB deps + ~5 GiB weights + staged audio.

**Python 3.12** — Ubuntu 24.04's system Python. Every dependency that could have blocked an ARM
port publishes `manylinux_aarch64` wheels for both cp312 and cp313, verified against PyPI:

| Package | linux-aarch64 tags |
|---|---|
| `torch` 2.13.0 | cp310–cp314 |
| `ctranslate2` 4.8.1 | cp39–cp314 |
| `numba` 0.67.0 | cp310–cp314 |
| `opensmile` 2.6.0 | `manylinux_2_17_aarch64` |
| `soundfile` 0.14.0 | pure-python |

3.12 is chosen over 3.13 to avoid deadsnakes, whose ARM support is poor. The suite must be run on
the VM to confirm nothing depends on 3.13.

### 8.2 No Docker

On a VM an image is pure overhead: ~2 GB built on ARM for no isolation benefit. A venv plus
systemd is less work, less memory, and faster to iterate. This also drops the earlier
"models baked into the image" question entirely — `HF_HOME` on the boot volume downloads once and
persists across restarts and code deploys.

### 8.3 Cloudflare Tunnel instead of open ports

`cloudflared` makes an **outbound** connection from the VM, so there is no listening port to
expose. This removes the two most error-prone Oracle steps: a VCN ingress rule *and* an iptables
rule, since Oracle's Ubuntu images ship netfilter rules that block everything but SSH. It also
removes certificate management — TLS terminates at Cloudflare.

Gradio's `share=True` is **not** used: those tunnels expire and the URL must stay valid through
the evaluation period.

### 8.4 Layout and units

```
/opt/autoace/
  repo/                    git clone
  venv/
  hf/                      HF_HOME — weights, downloaded once
  data/jobs.db             SQLite (WAL)
  data/uploads/<job_id>/   staged audio, unlinked per file
/etc/autoace.env           mode 0600, root-owned
```

```
autoace-web.service        Restart=always  EnvironmentFile=/etc/autoace.env
autoace-worker.service     Restart=always  EnvironmentFile=/etc/autoace.env
cloudflared.service
```

A **4 GiB swapfile** is provisioned as insurance for the transient peak (5.67 GiB peak vs 3.91
GiB steady). Swapping is slow, but slow beats an OOM kill, and §7.4 covers the case where it is
not enough.

### 8.5 Secrets

`OPENAI_API_KEY`, `AUTOACE_USER`, `AUTOACE_PASSWORD` in `/etc/autoace.env` (0600), referenced by
both units via `EnvironmentFile`. Never in the repo, never in a unit file, never baked into an
image.

### 8.6 Confidentiality

Brief §5 requires that production call audio not go to unapproved public services. A VM under our
control is a stronger position than any managed platform: the audio is unlinked per file, results
carry no audio, and nothing transits a third-party ML host. Cloudflare terminates TLS and so sees
request plaintext, which is the one third party in the path and must be stated plainly in the
memo.

---

## 9. Configuration additions

Appended to `config.py`, following its existing MEASURED / DERIVED / UNFITTED convention.

```python
# --- hosted dashboard -------------------------------------------------------
DATA_DIR = Path(os.environ.get("AUTOACE_DATA_DIR", "/opt/autoace/data"))
JOBS_DB = DATA_DIR / "jobs.db"
UPLOAD_DIR = DATA_DIR / "uploads"

JOB_TTL_SECONDS = 7 * 24 * 3600   # DERIVED — results kept until download; audio is
                                  # unlinked per file regardless (§5)
MAX_ATTEMPTS = 2                  # DERIVED — breaks the OOM crash loop (§7.4)
STALE_RUNNING_SECONDS = 900.0     # DERIVED — 5.6x the slowest measured file (160.8s)
WORKER_POLL_SECONDS = 2.0
UI_POLL_SECONDS = 5.0
MEAN_SECONDS_PER_FILE = 105.0     # MEASURED — mean of the three calls x 1.18 thread
                                  # penalty; drives the ETA in §6.4
MAX_UPLOAD_MB = 500               # DERIVED — 50 files at the largest provided call
                                  # (2.8 MB) is ~140 MB; 500 MB is generous headroom
```

---

## 10. Testing

`jobs.py` is pure SQLite and needs **no audio and no model weights**, so it is fast and runs
anywhere — including CI, which matters given §10.1.

| Test | Asserts |
|---|---|
| `test_enqueue_creates_job_and_file_rows` | counts, `pending` status, validation JSON round-trip |
| `test_claim_is_exclusive` | two successive claims never return the same row |
| `test_claim_increments_attempts` | `attempts` rises at claim, not at failure |
| `test_complete_records_result_and_unlinks_audio` | `audio_path` NULL, file gone from disk |
| `test_job_completes_when_last_file_finishes` | job flips to `complete`, workdir removed |
| `test_reconcile_resets_stale_running` | insert a stale `running` row -> back to `pending` |
| `test_reconcile_fails_row_past_max_attempts` | the §7.4 crash-loop guard |
| `test_expire_removes_old_jobs` | rows gone past `JOB_TTL_SECONDS` |
| `test_failed_file_still_appears_in_exports` | §7.1 — failures are not dropped |

Worker: one `slow`-marked integration test over a single real call.

App: the existing 8 validation/export tests must pass **unmodified**. New tests cover
`enqueue_batch` returning a job ID and `poll_job` projecting rows into the existing table shape.

### 10.1 CI is structurally limited, and that is deliberate

`reference/` is gitignored because the audio is confidential (brief §5), and the weights are
~5 GiB. So GitHub Actions can only run tests needing neither — `jobs.py`, `fuse.py`, schema, and
the app's validation/export tests. The memo must say this plainly, attributing it to §5 rather
than leaving it to look like thin coverage.

---

## 11. Failure modes and limitations

| # | Limitation | Status |
|---|---|---|
| 1 | One worker: jobs are FIFO across users, up to ~87 min wait | Accepted; surfaced as queue position (§6.4) |
| 2 | Peak memory measured on Windows, not Linux ARM | **Must re-measure on the VM** before trusting §2 |
| 3 | Happy-path (LLM available) peak never measured — needs a valid API key | Open; sized against the heavier fallback path meanwhile |
| 4 | Oracle ARM `Out of host capacity` can block provisioning entirely | Mitigated by requesting 2 OCPU; no in-repo fix |
| 5 | A file that OOMs twice is reported `failed`, not analysed | Accepted (§7.4); the alternative is a crash loop |
| 6 | Swap can mask memory pressure as latency | Accepted; journald records the worker's RSS per file |
| 7 | Cloudflare is in the TLS path | Disclosed in the memo (§8.6) |
| 8 | Results are lost if the boot volume is lost — no backups | Deliberate: results are reproducible, audio should not persist |
| 9 | Model weight cache size (~5 GiB) never isolated from a shared 60 GiB HF cache | Verify at warm-up |

---

## 12. Build order

1. `autoace/jobs.py` + its 9 unit tests (no audio, no weights — fast and CI-safe).
2. `autoace/worker.py` + `reconcile()` on startup + the expiry sweep.
3. Refactor `app.py`: `enqueue_batch`, `poll_job`, `gr.Timer`, job-ID lookup. Existing 8 tests
   must stay green.
4. Worker integration test over one real call.
5. Deploy artifacts: `deploy/setup.sh`, both systemd units, `cloudflared` config, README section.
6. Provision the VM, run the suite there, **re-measure peak RSS on ARM** (§11.2), warm the cache.
7. Rotate the OpenAI key, set `/etc/autoace.env`, measure the happy-path peak (§11.3).
8. Update `docs/MEMO.md`: hosting, privacy, the CI limitation, and the measured numbers.
