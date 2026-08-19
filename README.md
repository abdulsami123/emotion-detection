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

```bash
AUTOACE_USER=autoace AUTOACE_PASSWORD=<password> python -m autoace.app
```

Serves on `0.0.0.0:7860` (override with `PORT`). It refuses to start without a password. Upload a
ZIP containing audio at the root plus one CSV manifest; the batch is validated before any inference
runs, progress is shown per file, one bad file cannot fail the batch, and results download as CSV
and JSON with original filenames preserved. If the manifest carries labels, the scoring view
renders too.

Rows flagged **REVIEW** — confidence below 0.60, or conflicting evidence — sort to the top.

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
  app.py            Gradio dashboard
```

`config.py` is deliberately one large module. Keeping every threshold in one auditable place is a
rigor claim, not an oversight — several were found to be internally inconsistent with their own
anchors precisely because they sat side by side.
