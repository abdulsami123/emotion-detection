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

- A 50-file batch takes ~2.7 hours on the deployment hardware (§2; was projected at ~87 minutes
  on the x86 dev box). No HTTP request survives that, and no browser tab should have to stay
  open for it.
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
| Peak resident memory, one worker | ~~5.67 GiB~~ **superseded, see below** | dev-machine estimate |
| Steady-state resident memory | **3.91 GiB** | MEASURED — dev machine, same run, after the third call |
| Per-file latency, 16 threads | 58.2 / 58.7 / 129.8 s | MEASURED — dev machine, calls 001 / 002 / 003 |
| Per-file latency, 2 threads | 64.6 s (1.11x) / 160.8 s (1.24x) | MEASURED — dev machine, calls 001 / 003, `OMP_NUM_THREADS=2` |
| Per-file latency, 1 thread | 148.6 s (2.30x) / 371.1 s (2.31x) | MEASURED — dev machine, calls 001 / 003, `OMP_NUM_THREADS=1`, **vs the 2-thread figures** |
| 50-file batch projection, 2 threads | ~~87 min~~ **superseded, see below** | dev-machine estimate |
| 50-file batch projection, 1 thread | **~202 min** | DERIVED — dev-machine 105 s x 2.30 x 50; not re-derived on ARM |
| Python dependency footprint | ~1.5 GiB | MEASURED — torch 527M, gradio 193M, llvmlite 117M, scipy 115M, transformers 97M, ctranslate2 60M |
| Model weight cache | ~~5 GiB~~ **superseded, see below** | dev-machine estimate |

The rows above were all measured on the x86 development machine, before deployment. Now that the
system runs on the actual deployment hardware, the rows below **supersede** the peak-memory,
50-file-batch, and model-cache rows above; everything else (steady-state, thread-scaling ratios)
was not re-measured on ARM and is carried forward as-is.

| Quantity | Value | Provenance |
|---|---|---|
| Peak resident memory, one worker | **7.26 GiB** | MEASURED on deployment hardware — `/usr/bin/time -v` max RSS, all three calls in one process, NLI-fallback path, `Swaps: 0` |
| Per-file latency, 2 threads (ARM) | 115.2 / 146.8 / 315.1 s | MEASURED on deployment hardware — calls 001/002/003, `OMP_NUM_THREADS=2`, Ampere Neoverse-N1 |
| Mean per file (ARM) | **192.4 s** | MEASURED on deployment hardware — mean of the three calls above |
| 50-file batch, 2 threads (ARM) | **~160 min (~2.7 h)** | MEASURED-derived on deployment hardware — mean-per-file x 50 |
| Model weight cache | **4.2 GB** | MEASURED on deployment hardware — 4.1 GB HF cache + 85 MB ECAPA |

**ARM is 1.78–1.96x slower per file than the x86 dev box at the same `OMP_NUM_THREADS=2` setting**
(115.2/64.6 = 1.78x on call_001, 315.1/160.8 = 1.96x on call_003). The Ampere Neoverse-N1 cores are
materially slower per-core than the x86 dev machine; the earlier ~87-minute projection was correct
arithmetic on the wrong hardware. **The real 50-file batch figure is ~2.7 hours, not ~1 hour** —
this is a limitation, not a rounding difference, and is carried into §11 and the memo.

Both ARM figures are the **NLI-fallback path** (no `OPENAI_API_KEY`), matching the dev-machine
measurement basis exactly, so the 1.78–1.96x comparison is like-for-like. **The happy-path (LLM
available) peak is still unmeasured** — see §11 item 3.

Three consequences drive the whole design:

**(a) Cores above two buy almost nothing; the second core is not optional.** Going from 16 threads
to 2 costs only 1.11–1.24x, so the pipeline gains little from wide parallelism. But going from 2 to
1 costs **2.30x** — a cliff, not a taper, and consistent across both calls (2.30x and 2.31x). So
`OMP_NUM_THREADS=2` is a floor, not a tuning preference: one core turns a 50-file batch from ~2.7
hours into roughly double that on the deployment hardware (the 1-thread ratio itself was measured
only on the dev machine and not re-verified on ARM).

> An earlier revision of this section claimed "core count is nearly irrelevant". That was measured
> only over 16→2 and does not generalise downward. `WhisperModel` is constructed without
> `cpu_threads`, so CTranslate2 derives its intra-op count from `OMP_NUM_THREADS` — the cap was
> genuinely applied in both runs, so the non-linearity is real and not a measurement artefact.

**(b) RAM is the binding constraint for *sizing*, given at least two cores.** The measured peak on
the deployment hardware (7.26 GiB) against 10.9 GB usable leaves ~3.6 GB headroom, and the web
process plus Caddy consume some of that — workable, but thinner than the ~6 GiB planning figure
assumed. This retroactively confirms that a 6 GB shape would have OOM-killed the worker, which is
the concrete justification for insisting on the 12 GB shape (§8.1) rather than accepting a smaller
one if offered.

**(c) The 7.26 GiB peak is the *fallback* path.** `bart-large-mnli` (~1.6 GB) loads only inside
the `except` handler in `pipeline.py` — it is not a co-voter; the two voters are the LLM and the
dimensional SER. The deployed happy path should therefore be materially lighter, but that peak is
still unmeasured (§11 item 3). **We still size for 7.26 GiB**, because an OpenAI outage must not
OOM-kill the worker. See §7.4.

> **The memory-figure caveat is now resolved, not merely re-measured.** Windows peak working set
> and Linux max RSS measure genuinely different things — not the same quantity read on different
> hardware. Page size on the deployment VM is 4096, so the +28% delta (5.67 -> 7.26 GiB) is **not**
> explained by large pages; it is the metric, not the machine, that differs. The ARM run never
> swapped (`Swaps: 0`), so 7.26 GiB is a clean reading, not a value inflated by swap accounting.

### 2.1 Free tiers that were refuted, with the reason

Recorded so nobody re-proposes them.

| Option | Why it fails |
|---|---|
| Render free | 512 MB RAM, no persistent disk. Off by ~14.5x against the measured 7.26 GiB deployment-hardware peak (11x against the earlier 5.67 GiB dev-machine estimate). |
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
- Throughput is fixed at ~2.7 hours per 50 files on the deployment hardware (§2; was projected at
  ~87 min on the x86 dev box) by the pipeline itself, not by the queue technology. No queue makes
  it faster.

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
kept until download" real rather than aspirational, and it is the answer to a ~2.7-hour batch on
the deployment hardware: the evaluator uploads, takes the ID, and comes back.

### 6.4 Queue position

One worker means jobs are FIFO across users; a second batch waits behind the first for up to ~2.7
hours on the deployment hardware. The status panel shows queue position and an estimate derived
from `MEAN_SECONDS_PER_FILE` (192.0 on the deployment hardware), so a waiting job does not look
hung. Surfacing this is deliberate — a silent wait is the failure mode.

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

`VM.Standard.A1.Flex`, **2 OCPU / 12 GB (10.9 GB usable)**, aarch64.

**Superseded: this section originally planned Ubuntu 24.04.** The actual deployment is **Oracle
Linux 9.8**, not Ubuntu — chosen after the fact, and `deploy/setup.sh` was made distro-aware
(dnf/apt, firewalld/iptables, a derived app user, and a Caddy static binary where Caddy is not
packaged) rather than rewritten Oracle-Linux-only, so Ubuntu 24.04 remains a supported target even
though it was not the one actually used.

2 OCPU rather than the full free 4 OCPU / 24 GB allowance because §2a shows the extra cores buy
1.1–1.2x, while smaller shape requests are markedly more likely to be granted — Oracle ARM
returns `Out of host capacity` frequently in popular regions. 12 GB is no longer a comfortable ~2x
the measured peak: against the deployment-hardware figure of 7.26 GiB, 10.9 GB usable leaves only
~3.6 GB headroom once the web process and Caddy are accounted for. Workable, but thinner than
planned, and this is the concrete number that would have made a 6 GB shape OOM-kill the worker.

Default 50 GB boot volume is sufficient: ~1.5 GiB deps + 4.2 GB weights (measured; 4.1 GB HF cache
+ 85 MB ECAPA, superseding the earlier ~5 GiB estimate) + staged audio.

**Python 3.12**, installed from the `ol9_appstream` repo on Oracle Linux 9.8 (the system default is
3.9). Every dependency that could have blocked an ARM port publishes `manylinux_aarch64` wheels for
both cp312 and cp313, verified against PyPI:

| Package | linux-aarch64 tags |
|---|---|
| `torch` 2.13.0 | cp310–cp314 |
| `ctranslate2` 4.8.1 | cp39–cp314 |
| `numba` 0.67.0 | cp310–cp314 |
| `opensmile` 2.6.0 | `manylinux_2_17_aarch64` |
| `soundfile` 0.14.0 | pure-python |

3.12 is chosen over 3.13 to avoid deadsnakes, whose ARM support is poor. **Verified in production:**
41 CI-safe tests pass on ARM / Python 3.12, confirming nothing depended on 3.13.

### 8.2 No Docker

On a VM an image is pure overhead: ~2 GB built on ARM for no isolation benefit. A venv plus
systemd is less work, less memory, and faster to iterate. This also drops the earlier
"models baked into the image" question entirely — `HF_HOME` on the boot volume downloads once and
persists across restarts and code deploys.

### 8.3 TLS: Caddy plus a DuckDNS hostname

No domain is purchased. Caddy runs on the VM, terminates TLS, and reverse-proxies to Gradio on
`127.0.0.1:7860`, obtaining a real Let's Encrypt certificate for a free
`autoace-<suffix>.duckdns.org` hostname pointed at the instance's public IP.

**DuckDNS specifically, not `nip.io` or `sslip.io`.** All three are free wildcard-DNS services
that avoid buying a domain, but Let's Encrypt scopes its "certificates per registered domain"
rate limit (50/week) using the Public Suffix List, and this was verified against the live list:

| Host | On the Public Suffix List |
|---|---|
| `nip.io` | **No** |
| `sslip.io` | **No** |
| `duckdns.org` | **Yes** |

Because `nip.io` is absent from the PSL, every `*.nip.io` certificate in existence shares one
quota, which is chronically exhausted — issuance for `<ip>.nip.io` fails with "too many
certificates already issued". `duckdns.org` being present means each `<name>.duckdns.org` is
treated as its own registered domain with its own quota. This is a correctness difference, not a
preference.

Gradio's `share=True` is **not** used: those tunnels expire and the URL must stay valid through
the evaluation period.

**Cost:** two Oracle steps that a tunnel would have avoided — a VCN ingress rule for 80/443 and a
matching host-firewall rule, because Oracle's stock images ship rules that drop everything except
SSH. Port 80 must stay open for the ACME HTTP-01 challenge and renewals, not only for the initial
issuance. (Oracle's Ubuntu images use netfilter/`iptables` directly; the actual Oracle Linux 9.8
deployment uses `firewalld` instead — `setup.sh` is distro-aware for this reason.) **Verified in
production:** SELinux is Enforcing on the deployment host and did not interfere with Caddy
(`/usr/bin/caddy` is labelled `bin_t`).

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
caddy.service              distro package; /etc/caddy/Caddyfile
duckdns.timer              refreshes the A record if the public IP changes
```

Gradio binds `127.0.0.1:7860`, not `0.0.0.0` — Caddy is the only listener reachable from off-box,
so the app cannot be reached over plaintext HTTP even if a firewall rule is wrong.

A **4 GiB swapfile** is provisioned as insurance for the transient peak. Sized against the
dev-machine estimate (5.67 GiB peak vs 3.91 GiB steady); the deployment-hardware peak measured
7.26 GiB and the ARM run never swapped (`Swaps: 0`), so the swapfile was not actually drawn on in
this measurement, but it remains the insurance policy for a heavier real batch. Swapping is slow,
but slow beats an OOM kill, and §7.4 covers the case where it is not enough.

### 8.5 Secrets

`OPENAI_API_KEY`, `AUTOACE_USER`, `AUTOACE_PASSWORD` in `/etc/autoace.env` (0600), referenced by
both units via `EnvironmentFile`. Never in the repo, never in a unit file, never baked into an
image.

### 8.6 Confidentiality

Brief §5 requires that production call audio not go to unapproved public services. A VM under our
control is a stronger position than any managed platform: the audio is unlinked per file, results
carry no audio, and nothing transits a third-party ML host.

**No third party sees plaintext.** TLS terminates in Caddy on our own VM. DuckDNS provides only
DNS resolution — no traffic passes through it — and Let's Encrypt sees only certificate requests.
This is a materially better §5 position than the managed-platform options, all of which terminate
TLS on someone else's edge, and it is worth stating in the memo as a deliberate choice rather
than a side effect.

The one genuine external dependency remains OpenAI, which receives the annotated transcript (not
audio) for the tone branch. That was already disclosed in the pipeline spec and is unchanged.

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
4.2 GB measured (4.1 GB HF cache + 85 MB ECAPA; supersedes the earlier ~5 GiB estimate). So GitHub
Actions can only run tests needing neither — `jobs.py`, `fuse.py`, schema, and the app's
validation/export tests. The memo must say this plainly, attributing it to §5 rather than leaving
it to look like thin coverage.

---

## 11. Failure modes and limitations

| # | Limitation | Status |
|---|---|---|
| 1 | One worker: jobs are FIFO across users, up to **~2.7 h wait** on deployment hardware (was ~87 min on the x86 dev-machine projection) | Accepted; surfaced as queue position (§6.4) |
| 2 | ~~Peak memory measured on Windows, not Linux ARM~~ — **done.** | **Resolved:** 7.26 GiB max RSS measured on the deployment hardware (`/usr/bin/time -v`, `Swaps: 0`). Windows peak-working-set and Linux max-RSS measure different things, not the same quantity on different hardware — page size is 4096, so the +28% delta is not a large-page artefact. This is now the number the box is sized against (§2b). |
| 3 | Happy-path (LLM available) peak never measured — needs a valid API key | **Still open.** Sized against the heavier fallback path (7.26 GiB) meanwhile |
| 4 | Oracle ARM `Out of host capacity` can block provisioning entirely | Mitigated by requesting 2 OCPU; no in-repo fix |
| 5 | A file that OOMs twice is reported `failed`, not analysed | Accepted (§7.4); the alternative is a crash loop |
| 6 | Swap can mask memory pressure as latency | Accepted; journald records the worker's RSS per file. The measured deployment run never swapped (`Swaps: 0`), so this did not fire in practice |
| 7 | DuckDNS is a third-party DNS dependency: if it stops resolving, the hostname breaks | Accepted — DNS only, no traffic path (§8.6). The public IP keeps working and the cert stays valid |
| 10 | Oracle public IPs are ephemeral unless reserved, and a change breaks both DNS and TLS | Mitigated by reserving the IP at create time plus a `duckdns.timer` refresh |
| 8 | Results are lost if the boot volume is lost — no backups | Deliberate: results are reproducible, audio should not persist |
| 9 | Model weight cache size (~5 GiB) never isolated from a shared 60 GiB HF cache | **Resolved:** measured at 4.2 GB (4.1 GB HF cache + 85 MB ECAPA) on the deployment host |

---

## 12. Build order

1. `autoace/jobs.py` + its 9 unit tests (no audio, no weights — fast and CI-safe).
2. `autoace/worker.py` + `reconcile()` on startup + the expiry sweep.
3. Refactor `app.py`: `enqueue_batch`, `poll_job`, `gr.Timer`, job-ID lookup. Existing 8 tests
   must stay green.
4. Worker integration test over one real call.
5. Deploy artifacts: `deploy/setup.sh`, both systemd units, `Caddyfile`, DuckDNS refresh timer,
   README section.
6. Provision the VM, run the suite there, **re-measure peak RSS on ARM** (§11.2), warm the cache.
7. Rotate the OpenAI key, set `/etc/autoace.env`, measure the happy-path peak (§11.3).
8. Update `docs/MEMO.md`: hosting, privacy, the CI limitation, and the measured numbers.
