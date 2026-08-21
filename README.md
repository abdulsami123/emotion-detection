# AutoAce — Voice Tone & Background Noise

Classifies emotional tone and background noise in call-centre audio into a fixed 9-field schema,
under a $0.003-per-audio-minute inference ceiling.

- **Design:** `docs/superpowers/specs/2026-08-15-voice-tone-noise-design.md`
- **Technical memo (results, cost, latency, limitations):** `docs/MEMO.md`

---

## Provided audio is not in this repository

The trial's call recordings are confidential production customer audio and the target repository
is public, so `reference/` is gitignored — along with `tests/fixtures/asr_baseline.json`, which
contains transcripts derived from those calls.

**To run the tests, place the provided files here yourself:**

```
reference/call_001.ogg
reference/call_002.ogg
reference/call_003.ogg
reference/labels.csv
```

Everything resolves through `autoace.config.reference_call()`, so nothing assumes the audio sits in
the working directory. `labels.csv` uses the brief's manifest format: a `name` column holding the
bare filename, and a `result_json` column holding the expected JSON object.

---

## Install

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                       # optional; see "Without an API key" below
```

**Do not relax the `gradio` / `transformers` / `pydantic` pins.** They are genuinely constrained:
gradio ≥6 pulls `huggingface-hub` ≥1.0 which `transformers` 4.46 rejects, and gradio 4.44/5.9
return HTTP 500 on page load under pydantic ≥2.12. Only `gradio==5.44.1` satisfies both sides.
The reasoning is recorded in `requirements.txt`.

First run downloads roughly 5 GB of model weights (faster-whisper large-v3-turbo, AST, audeering
SER, bart-large-mnli, ECAPA, SQUIM, Silero VAD).

---

## Analyse one file

```bash
python -c "
from autoace.pipeline import analyse_file
from autoace.config import reference_call
r = analyse_file(reference_call('call_001.ogg'))
print(r.analysis.model_dump_json(indent=2))
print('review flagged:', r.review_flagged)
print('tone path:', r.reasoning)
"
```

## Score a labelled batch

```bash
python -c "
import json
from autoace.eval import load_manifest, score_batch
from autoace.pipeline import analyse_file
from autoace.config import LABELS_CSV, reference_call
rows = load_manifest(str(LABELS_CSV))
preds = {r.name: analyse_file(reference_call(r.name)).analysis for r in rows}
print(json.dumps(score_batch(rows, preds), indent=2, default=str))
"
```

## Run the dashboard

Two processes share a SQLite job store, because a 50-file batch takes ~87
minutes and no HTTP request survives that:

```bash
export AUTOACE_DATA_DIR=./_data
python -m autoace.worker &                       # claims files, runs the pipeline
AUTOACE_USER=autoace AUTOACE_PASSWORD=<password> python -m autoace.app
```

The web process binds `127.0.0.1:7860` (`AUTOACE_BIND=0.0.0.0` to change it) and
refuses to start without a password. Upload a ZIP containing audio plus one CSV
manifest; the batch is validated before any inference runs, one bad file cannot
fail the batch, and results download as CSV and JSON with original filenames
preserved.

**Keep the job ID.** Processing is asynchronous — close the page and paste the
ID into **Look up a job** to return to your results, which are kept for 7 days.
Audio is deleted as soon as each file is analysed.

Rows flagged **REVIEW** — confidence below 0.60, or conflicting evidence — sort
to the top.

If you see `no such column: manifest_csv`, you have a job store from before the
schema changed: `rm -rf _data`.

---

## Deploying

`deploy/setup.sh` provisions the whole VM idempotently: packages, Caddy, the
host firewall, a swapfile, the venv, both systemd units, and a model-cache
warm-up. Design and rationale for every choice below are in
`docs/superpowers/specs/2026-08-19-hosted-dashboard-design.md`.

The script supports **Ubuntu 24.04 (apt) and Oracle Linux 9 (dnf)**, detected
automatically. The actual deployment was done on **Oracle Linux 9.8 aarch64**,
with Python 3.12 installed from the `ol9_appstream` repo (the system default
is 3.9). On aarch64 Linux the default PyPI `torch` build is the CUDA build, so
the script installs the CPU-only wheel explicitly before the rest of
`requirements.txt`.

**Instance:** Oracle Cloud Always Free, `VM.Standard.A1.Flex`, **2 OCPU / 12 GB**,
aarch64, Python 3.12.

2 OCPU rather than the free 4 because the extra cores buy only 1.1–1.2×, and
smaller shape requests are far more likely to be granted — `Out of host
capacity` is common for the free ARM shape.

**2 OCPU is a floor, not a preference.** Dropping to 1 costs **2.30×** (measured;
see the memo's latency table), taking a 50-file batch from ~87 minutes to ~202
minutes. The shape form defaults to 1 OCPU / 6 GB — raise both sliders. 6 GB is
also below the 5.67 GiB measured peak once the OS and Caddy are accounted for.
If 2 OCPU is refused, change Availability Domain and retry rather than accepting
1.

**Why not a PaaS free tier.** The worker's measured peak is **5.67 GiB**.
Render's free tier is 512 MB — off by 11×. Every other free tier surveyed
(Hugging Face Spaces, Railway, Fly.io/Koyeb/Northflank, AWS/GCP/Azure micro
VMs) is 256 MB–1 GB; none come close. See the memo's hosting section for the
full table and reasons.

**TLS:** Caddy reverse-proxies `127.0.0.1:7860` to a free `duckdns.org`
hostname, obtaining a real Let's Encrypt certificate. DuckDNS specifically,
not `nip.io`/`sslip.io` — DuckDNS is on the Public Suffix List so it gets its
own Let's Encrypt rate-limit quota; the others are not on the list and share
one chronically-exhausted quota.

**Swap:** a 4 GiB swapfile insures the gap between the 3.91 GiB steady state
and the 5.67 GiB transient peak. Swapping is slow, but slow beats an OOM kill.

**Two things `setup.sh` cannot do, because they are Oracle console operations
rather than in-VM state:**

1. Open a VCN ingress rule for TCP 80 and 443. Port 80 must stay open
   *permanently* — Caddy's ACME renewals reuse it, not only the first
   certificate issuance.
2. Reserve the instance's public IP. Oracle's public IPs are ephemeral by
   default; an unreserved one can change on reboot and silently break both
   DNS and the certificate.

---

## Without an API key

The tone branch is two-tier and degrades on purpose:

1. `gpt-4o-mini` over an annotated transcript (primary), via `OPENAI_API_KEY`
2. `bart-large-mnli` zero-shot, entirely local (fallback)
3. Neutral prior only if both fail

A result produced without the primary classifier is **hard-capped at 0.55 confidence**, below the
review threshold, so it always reaches a human rather than shipping as if it were confident. The
path taken is recorded in `FileResult.reasoning` — a silent fallback is operationally
indistinguishable from success.

**The six signal fields never touch an API or a network** and are unaffected.

---

## Tests

```bash
python -m pytest -q                     # everything (~10 min on CPU)
python -m pytest -q -m "not slow"       # skip the end-to-end pipeline runs
python -m pytest tests/test_fuse.py -q  # a single fast module
```

Two tests are `xfail(strict=True)` and that is deliberate — they document known defects with the
measurements that diagnose them, and will flip to XPASS the moment either is genuinely fixed:

| test | defect |
|---|---|
| `test_code_switched_call_does_not_inflate_customer_speech` | speaker embeddings split the bot's English voice from its own Spanish voice; 12.36s attributed to the customer against a true ~0.9s |
| `test_static_is_typed_correctly_on_call_003` | no tested spectral feature discriminates line noise; all order the calls backwards |

---

## Layout

```
autoace/
  config.py         EVERY threshold, annotated MEASURED / DERIVED / UNFITTED
  schema.py         the 9-field output contract
  io_audio.py       decode; never normalizes loudness (absolute level is load-bearing)
  vad.py            Silero; segmentation shared by both branches
  diarize.py        ECAPA 2-cluster agent/customer assignment
  asr.py            faster-whisper; word timestamps + per-slice language detection
  acoustics.py      shared DSP primitives
  tagging.py        AudioSet typing (spectral artifact detector disabled - see docstring)
  quality.py        SQUIM + DSP detectors; peak-relative noise floor
  signal_branch.py  assembles the six deterministic fields
  prosody.py        eGeMAPS, speaker-relative activation, degenerate-baseline detection
  ser.py            dimensional arousal / dominance / valence
  tone_llm.py       prompt assembly + OpenAI structured output (only vendor-specific file)
  tone_nli.py       local zero-shot fallback and second approach
  fuse.py           intensity reconciliation, agreement-based confidence
  pipeline.py       per-file orchestration with fail isolation
  eval.py           manifest -> grouped metrics
  jobs.py           SQLite job store (WAL): queue, checkpoint log, results table
  worker.py         drain loop: claims one file, runs the pipeline, checkpoints
  app.py            Gradio dashboard
```

`config.py` is deliberately one large module. Keeping every threshold in one auditable place is a
rigor claim, not an oversight — several were found to be internally inconsistent with their own
anchors precisely because they sat side by side.
