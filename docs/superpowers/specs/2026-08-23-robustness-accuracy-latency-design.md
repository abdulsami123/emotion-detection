# Robustness, accuracy and latency — design

**Date:** 2026-08-23
**Status:** proposed
**Supersedes:** nothing. Extends `2026-08-15-voice-tone-noise-design.md` (pipeline) and
`2026-08-19-hosted-dashboard-design.md` (hosting).

Every number in this document was measured on the deployment hardware (Oracle
`VM.Standard.A1.Flex`, 2 OCPU Ampere Neoverse-N1 aarch64, `OMP_NUM_THREADS=2`)
with the worker stopped so nothing contended for the two cores. Figures that are
projections are labelled PROJECTED. Figures fitted from two points are labelled
as such — a two-point fit cannot distinguish curvature from noise.

---

## 1. Why this work exists

The system is deployed and correct on the six deterministic signal fields, but
three separate audits converge on the same conclusion: **the expensive parts of
the pipeline are not the parts carrying the accuracy, and two of the fields that
look correct are unfalsifiable on the data we have.**

Specifically:

- 71% of runtime is two stages (ASR 48.5%, SQUIM 22.7%).
- SQUIM serves `audio_quality`, whose ground truth is `clear` on all three
  labelled calls, using thresholds `config.py` itself marks `UNFITTED`.
- The fusion layer demonstrably discards a correct answer (§3.2).
- Confidential audio persists in `/tmp` for days despite the README claiming
  it is deleted after analysis (§3.1).

None of this requires new labelled data to fix. The tone accuracy problem does,
and is therefore quarantined into a gated Phase 3.

---

## 2. Measured baseline

### 2.1 Stage attribution

Obtained by wrapping every stage at its call site (`pipeline.py` uses
`from x import y`, so the patch must target the *using* module, not the defining
one) and running the real `analyse_file`.

| stage | call_001 (30.9s) | call_003 (171.9s) | two-point fit | @80s | share |
|---|---|---|---|---|---|
| ASR (`large-v3-turbo`, int8, beam=5) | 52.5s | 152.7s | 30.5s + 0.711× | 87.4s | **48.5%** |
| SQUIM STOI | 32.5s | 56.9s | 27.2s + 0.173× | 41.0s | **22.7%** |
| SER dimensions | 13.4s | 13.0s | 13.5s + 0× | 13.3s | 7.4% |
| overlap detect (ECAPA) | 2.5s | 29.9s | 0.194× | 12.0s | 6.7% |
| AST noise type | 0.0s (skipped) | 29.6s | 0.210× | 10.3s | 5.7% |
| diarize (ECAPA) | 1.8s | 13.9s | 0.086× | 6.0s | 3.3% |
| tone LLM | 4.5s | 4.1s | 4.6s + 0× | 4.4s | 2.4% |
| VAD (**runs 3×**) | 1.5s | 7.5s | 0.043× | 3.6s | 2.0% |
| decode | 0.9s | 2.1s | 0.009× | 1.3s | 0.7% |
| eGeMAPS | 0.2s | 1.8s | 0.011× | 0.8s | 0.4% |
| **total** | **111.6s** | **313.9s** | **77s + 1.29×** | **180s** | |

Cross-check: an independent fit over five warm files in the systemd journal gives
`84s fixed + 1.31× audio`, within 5% of the stage model. The models corroborate.

**77s of every file is fixed cost**, independent of audio length. That is why
short calls have the worst realtime factors (call_001 at 3.31×, call_003 at 1.79×).

### 2.2 Memory

Happy path (LLM reachable), `/usr/bin/time -v` max RSS: **4.43 GB** on call_001,
**5.82 GB** on call_003. This closes an open question from `MEMO.md` — the
previously recorded **7.26 GiB** was the heavier NLI-fallback path, which remains
the sizing constraint against 10.9 GB usable.

### 2.3 Reliability

Zero errors, zero warnings, zero OOM events across every run in the journal.
Interruption recovery has been exercised unintentionally and worked
(`startup reconcile: 1 requeued`). **The durability design is not what needs
work.**

### 2.4 Determinism

Same audio, same process, twice per stage:

| stage | result |
|---|---|
| VAD | identical (31 segments, hash `00ddd2026802de84`) |
| diarization | identical (73.80s customer, 24 segments) |
| ASR | identical (376 words, `avg_logprob` −0.11379 both runs) |
| SER | identical to full float precision |
| tone LLM, same payload, N=5 within session | identical 5/5 |
| tone LLM, across days | **`frustrated/0.9` → `distressed/0.8`** |

The local pipeline is bit-reproducible. The only nondeterminism is the remote
model, and `LLM_MODEL = "gpt-4o-mini"` is a **floating alias** with no `seed`
passed. That is the whole explanation, and it makes the evaluation
irreproducible across days.

---

## 3. Findings this design responds to

### 3.1 Confidential audio persists in `/tmp` (highest priority)

```
2026-08-21 03:32  2.8 MB  /tmp/gradio/7418d8a2.../call_003.ogg
2026-08-22 16:34  503 KB  /tmp/gradio/6388bb02.../call_001.ogg
2026-08-21 03:00  2.8 MB  /tmp/autoace_batch_wjcudt5s/call_003.ogg   (failed job)
```

Observed 2026-08-23, so up to two days old. `README.md` states "Audio is deleted
as soon as each file is analysed" — true only of our own workdir copy.
Two independent leaks:

1. Gradio's upload cache (`/tmp/gradio/<hash>/`) retains the original upload.
   `jobs.py` never knew about it; it only ever removed `audio_path` and the
   workdir.
2. A job that fails validation sets `status='failed'` with `total_files=0`, so
   it has no `files` rows, so nothing ever triggers workdir removal. Confirmed
   against job `eb6ce32c` whose workdir still exists.

Brief §5 requires production audio be treated as confidential. This is the one
finding that is a compliance issue rather than a quality issue.

### 3.2 The Tier C intensity override discards a correct answer

Replaying the cached payload for call_001 against the live API:

```
call_001  truth = upset/high
  LLM RAW      : upset/high      <- correct on BOTH fields
  FUSED OUTPUT : upset/medium    <- reconcile_intensity(..., Tier.C) hard-returns MEDIUM
```

`reconcile_intensity` returns `(MEDIUM, False)` unconditionally for Tier C.
Two of three labelled calls are Tier C (7.1s and 12.4s of customer speech), so
**intensity is structurally incapable of emitting anything but `medium` on short
calls.** call_001's truth is `high` — unreachable by construction, and the
correct answer was available and thrown away.

The Tier C rationale was sound in origin: a 25s baseline window over <25s of
customer speech yields a mean z-score of exactly zero by construction, which
manufactured a fake `low`. But the fix suppressed *both* voters rather than only
the degenerate one. The prosody voter is unmeasurable on short calls; the LLM
voter is not.

### 3.3 Tone: prompt engineering is saturated, and the signal is not in the payload

Tested rather than assumed. `SYSTEM_PROMPT` already encodes the call_003 rule
verbatim ("a caller who stays warm and thanks the agent is SATISFIED even if the
agent failed") and still fails. Adding an explicit debiasing block — telling the
model the delivery vocabulary is arousal-only with no positive pole — did not
help at any tier and actively hurt the weakest model:

| model | tone (baseline) | intensity (baseline) | tone (debiased) | intensity (debiased) |
|---|---|---|---|---|
| `gpt-4o-mini` (current) | 1/3 | 2/3 | **0/3** | 1/3 |
| `gpt-4.1-mini` | 1/3 | **3/3** | 1/3 | **3/3** |
| `gpt-4o` | **2/3** | 1/3 | **2/3** | 2/3 |

Supporting evidence that this is a representation problem, not a prompting or
model-capability problem:

- **No model at any tier gets call_003 right.** Three-for-three agreement on the
  same error is meaningful in a way that the one-call differences between tiers
  (which at n=3 are noise) are not.
- **`satisfied` was never emitted once** across ~48 predictions (3 models × 2
  prompt arms × repeats). One of five labels is effectively unreachable.
- **SER valence is inverted on the label ordering.** Measured: call_001 `0.53`
  (upset), call_003 `0.55` (satisfied), call_002 `0.64` (neutral). Truth ordering
  must be 001 < 002 < 003; the two non-negative calls are swapped. The primary
  valence evidence points the wrong way.
- Every error is toward the negative pole: truth is upset/neutral/satisfied,
  prediction is upset/frustrated/distressed.

**Conclusion:** an audio-native tone path is now the only remaining lever, not a
nice-to-have. It is also the one item that cannot be validated at n=3, hence
Phase 3 being gated.

### 3.4 Confidence is not tracking correctness

| call | tone correct? | confidence | flagged? |
|---|---|---|---|
| call_001 | **yes** | 0.45 | yes |
| call_002 | no | 0.45 | yes |
| call_003 | no, by two labels | 0.80–0.90 | **no** |

The one call we get right is flagged; the badly-wrong one ships unflagged. n=3
so this is not significant, but confidence is currently constructed from voter
agreement rather than calibrated against outcomes, and there is no evidence it
carries information.

### 3.5 Two "working" fields have constant ground truth

`audio_quality` is `clear` on all three labels and `long_silence_present` is
`false` on all three. **A hardcoded literal scores 100% on both.** Neither
detector has ever been shown a positive example, and `SQUIM_STOI_SLIGHT` /
`SQUIM_STOI_SEVERE` are marked `UNFITTED` in `config.py` for exactly that reason.

This is the justification for gating SQUIM: it is the second-most-expensive
component in the system and it serves the field whose label never varies.

### 3.6 Dead code and dropped signals

- `asr.detect_language()` is **never called.** It carries a docstring describing
  it as the per-slice code-switch mechanism, and the `xfail` in
  `tests/test_diarize.py` documents the code-switch defect it was built to fix.
- `ToneRequest.interruptions` is **always 0**, while `signal_branch` separately
  and correctly reports `speaker_overlap_present: true` on the same call. The
  tone classifier is being told there were no interruptions on calls where we
  detected overlap.
- Clip-level `language` is reported as `en` for call_002, which is a Spanish
  request. Whisper detects language from the first 30s, which is mostly bot
  English.

### 3.7 ASR is fabricating evidence

call_002's tone payload contains, presented to the model as acoustic-and-lexical
evidence, a fluent ~12-word Spanish sentence about vehicle measurements that has
no relationship to the call. The call's entire caller content is a request to be
served in Spanish. Whisper hallucinating on short, noisy, code-switched audio is
a documented failure mode, and this is an instance of it. Truth for call_002 is
`neutral`; the system says `frustrated`. Fabricated text of that register is a
plausible cause.

(The verbatim span is deliberately not reproduced here. Transcripts derived from
the provided calls are confidential — `tests/fixtures/asr_baseline.json` is
gitignored for exactly this reason — and this repository is public.)

Separately, `_text_in_segment` assigns a word to *every* VAD segment it overlaps
rather than to one, so the last word of each segment reappears as the first word
of the next. On call_003, 5 of the first 9 annotated utterances begin with a
duplicate of the preceding utterance's final word. Every transcript therefore
reads as stammering, biasing the tone classifier toward disfluency and distress
on every call.

---

## 4. ASR replacement: Parakeet

### 4.1 Measured

`nemo-parakeet-tdt-0.6b-v2` via `onnx-asr`, int8, 2 threads:

| call | audio | Whisper | Parakeet | speedup |
|---|---|---|---|---|
| call_001 | 30.9s | 52.5s | **3.5–3.6s** | **~14.8×** |
| call_002 | 35.0s | — | 4.2s | — |
| call_003 | 171.9s | 152.7s | **33.5s** | **4.6×** |

Two-point fit: **`0.212× audio`, no fixed cost** (intercept −3.0s, i.e. zero),
against Whisper's `30.5s + 0.711×`. Model load 2.8s warm / 10.3s cold.

The short-call gain is disproportionate because Whisper's 30.5s fixed cost — the
~15s language-detect encoder pass plus warm-up — is 17% of every file and
Parakeet has no equivalent.

### 4.2 Use `onnx-asr`, not NeMo

Resolved on the deployment box (`--dry-run`, aarch64, Python 3.12):

| route | new packages | notes |
|---|---|---|
| `nemo_toolkit[asr]` | **~70** | pulls `wandb`, `sentry-sdk`, `tensorboard`, `pytorch-lightning`, `datasets`, `lhotse`, `cuda-bindings`, `nv-one-logger-*`; changes `setuptools`/`packaging` |
| `onnx-asr[cpu]` | **1** | `onnxruntime 1.29.0`, `protobuf`, `flatbuffers`, `numpy` already installed |

NeMo is a training framework. Two objections beyond weight: it would move
`setuptools`/`packaging` underneath the gradio/transformers/pydantic pins that
`README.md` explicitly warns must not be disturbed, and it installs two
telemetry SDKs (`wandb`, `sentry-sdk`) onto a host processing confidential
customer audio. `onnx-asr` is a pure-Python wrapper over an ONNX runtime already
present.

### 4.3 Interface compatibility

Both hard requirements are satisfiable:

- **Word timestamps.** `with_timestamps()` returns a flat per-token float list
  alongside a sub-word `tokens` list (`' C'`, `'ome'`, `' on'`, `'.'`). Words are
  recovered by grouping on leading-space boundaries. Resolution is 0.08s, finer
  than the prosody grid requires. `asr.Word(start, end, text)` is reconstructible
  losslessly at word granularity.
- **`avg_logprob`.** A `logprobs` array is returned, feeding
  `ConfidenceInputs.asr_avg_logprob` directly.
- `with_vad()` also exists, which is the same speech-region gating proposed
  independently for latency.

### 4.4 Language routing without a language-ID model

Parakeet v2 is English-only, so routing is required. Its own mean logprob
separates the languages cleanly:

| call | language | mean logprob |
|---|---|---|
| call_001 | en | **−0.0481** |
| call_003 | en | **−0.0411** |
| call_002 | es / code-switch | **−0.3035** |

A 6–7× gap. **Design: run Parakeet unconditionally; re-run on Whisper only when
mean logprob falls below a threshold.** This needs no LID model, no extra
download, and lets Whisper lazy-load so it never enters memory on English-only
batches — which *reduces* peak RSS in the common case rather than adding to it.

It also finally gives `detect_language()` a purpose, or retires it: per-slice
language detection becomes reachable once a cheap router exists.

**Caveat, stated plainly:** the threshold would be fitted on exactly one
non-English example. This is the same circularity that affects every other
threshold in the project. It must be recorded as `UNFITTED` in `config.py`, not
dressed up as measured.

### 4.5 The risk, and why the regression gate comes first

On call_001, Whisper renders the caller's four escalating one-word repeats as
four separate utterances. **Parakeet v2 collapses them into a single two-word
utterance.** That repetition is the evidence the prompt uses to reach `upset` —
the only tone answer the system currently gets right.

The damage is probably limited: the delivery tags (`louder`, `rising pitch`) are
computed by `prosody.extract_features` over VAD segments, not from ASR text, so
four segments with rising loudness survive whatever the text says. But "probably"
is not a basis for shipping, and this is why the golden-output regression gate is
Phase 1 work that must land before Phase 2 touches ASR.

`parakeet-tdt-0.6b-v3` (multilingual) is **not** a substitute — it truncated
call_002 from 94 tokens to 19, dropping the Spanish entirely, and its logprob
separation is only 2.3×, so it routes worse. It does preserve call_001's repeats as
distinct utterances, so it is retained as a fallback candidate if v2's collapse
proves to matter.

---

## 5. Cost constraint on the model swap

`gpt-4.1-mini` is the documented upgrade candidate and measures 3/3 on intensity
(§3.3). It also **breaches the $0.003/audio-minute ceiling on short calls in
three of four configurations.** Short calls are the worst case because 2500 of
2712 input tokens are the fixed system + few-shot prefix, amortised over less
audio.

| config | LLM | compute | total | |
|---|---|---|---|---|
| 4.1-mini, no caching, spot @62s | $0.00283 | $0.00086 | $0.00369 | breach −23% |
| 4.1-mini, cached prefix, on-demand @62s | $0.00183 | $0.00172 | $0.00355 | breach −18% |
| 4.1-mini, cached prefix, spot @62s | $0.00183 | $0.00086 | $0.00269 | ok +10% |
| 4.1-mini, cached prefix, spot @30s (post-Phase 2) | $0.00183 | $0.00042 | $0.00224 | ok +25% |

Rates from `MEMO.md`, which flags them for re-verification. Two consequences:

1. **The model swap is gated on Phase 2's compute reduction**, or on confirming
   prompt caching. It cannot ship standalone in Phase 1 as originally ranked.
2. **Prompt caching has never been verified.** The code never reads
   `usage.prompt_tokens_details.cached_tokens`, so "the prefix qualifies" is an
   assumption in the memo, not a measurement. Recording it is a few lines and
   decides the phase ordering.

The Batch API is a further 50% discount and fits this architecture — the system
is already asynchronous with a job queue and users are already told to expect
hours. It is recorded as an option, not adopted, because it changes the latency
contract from minutes to up-to-24h.

---

## 6. Design

### Phase 1 — correctness and the regression gate

No new data required. Nothing here changes a model or a threshold.

**1.1 Close both audio leaks.**
`jobs.py` gains knowledge of the upload's *source* path, not only the workdir
copy. Concretely: `enqueue` records the Gradio cache directory alongside
`workdir`; `_remove_workdir` is joined by `_remove_upload_cache`; and `reconcile`
sweeps workdirs belonging to jobs in a terminal state with zero outstanding files
— which is the existing orphan-job sweep, extended to `failed` jobs that never
had `files` rows. Filesystem work stays *after* the SQL commit, per the existing
atomicity pattern (`rmtree` cannot be rolled back).

`expire` additionally removes any `/tmp/gradio` and `autoace_batch_*` directory
older than `JOB_TTL_SECONDS`, so pre-existing residue drains without manual
intervention.

**1.2 Repair the Tier C intensity path.**
Suppress only the voter that is actually unmeasurable. `reconcile_intensity` on
Tier C returns the LLM's intensity when the prosody baseline is degenerate,
rather than discarding both and emitting the prior. Confidence remains capped —
a single-voter result is still a weaker result — but a correct answer stops being
thrown away. Expected effect: call_001 intensity `medium` → `high`.

**1.3 Make the remote model reproducible and observable.**
Pin a dated snapshot instead of the floating alias, pass `seed`, and persist
`system_fingerprint` and `cached_tokens` on every `FileResult`. This converts two
current assumptions — that temperature 0 is reproducible, and that the prefix is
being cached — into recorded facts. `cached_tokens` is what unblocks §5.

**1.4 Wire the dropped signals.**
`ToneRequest.interruptions` is populated from the overlap detector so the tone
classifier stops receiving `0` on calls where overlap was detected. This is a
one-line plumbing fix for an already-computed value.

**1.5 The golden-output regression gate.**
Freeze all 9 fields for all 3 calls as a committed fixture. Any diff fails the
suite and must be explicitly re-blessed with a recorded reason. With two of six
"working" fields having constant ground truth (§3.5), diff-detection is the only
mechanism that can catch a silent regression from Phases 2–3 — accuracy metrics
cannot, because at n=3 they have no resolution.

The fixture stores the 9 fields and `review_flagged`, not `confidence`, so
Phase 1.2's expected confidence change does not produce spurious failures.

### Phase 2 — latency

Gated on 1.5 existing. Target: **180s → ~62s per file, 150 → ~51 min per 50-file
batch** (2.92×, PROJECTED from the measured stage model plus the measured
Parakeet timings).

**2.1 Parakeet for English, Whisper for everything else.**
New `autoace/asr_parakeet.py` behind the existing `asr.transcribe` interface, so
`pipeline.py` is unchanged. Returns `Transcript` with `Word(start, end, text)`
reconstructed from token timestamps on leading-space boundaries, and
`avg_logprob` from `logprobs`. `asr.transcribe` becomes a router: run Parakeet,
and re-run on Whisper when mean logprob is below `PARAKEET_MIN_LOGPROB`
(`UNFITTED` — fitted on one example). Whisper is lazy-loaded so English-only
batches never page it in.

**2.2 Fix boundary word duplication.**
`_text_in_segment` assigns each word to exactly one segment — the one containing
its midpoint — rather than every segment it overlaps. Removes the stammering
artefact from every transcript.

**2.3 Gate SQUIM behind the DSP detectors.**
`assess_quality` already reaches a verdict from five deterministic detectors;
SQUIM only adds the ability to call a call impaired when no detector fires. Run
it only when the cheap evidence is ambiguous. Saves ~35s of 41s at an 80s call
on clean audio, with no measured accuracy cost — noting that "no measured cost"
here means the field's label never varies, which is a weaker statement than "no
cost".

**2.4 Dedupe VAD.**
Silero currently runs three times per file on identical audio:
`signal_branch:214`, `signal_branch:215` (via `non_speech_segments`, which calls
`speech_segments` internally), and `pipeline:172`. `lru_cache` sits on the model
loader, not the result, and numpy arrays are unhashable. Compute segments once in
`pipeline.analyse_file` and pass them down.

**2.5 Batch the ECAPA overlap windows.**
`_speaker_overlap_present` embeds 0.8s windows at 0.4s hop one forward pass at a
time. Batch them.

**2.6 Re-measure and re-cost.**
Re-run the stage profile and the cost table. This is what unblocks the
`gpt-4.1-mini` decision in §5.

### Phase 3 — tone (gated on 30–50 labelled calls)

**Does not start until the data exists.** §3.3 establishes that no text-model
tier and no prompt recovers call_003, and that SER valence is inverted on the
label ordering — so this phase is a genuine spike with an uncertain outcome, and
running it against n=3 would produce a result indistinguishable from noise.

- Audio-native tone classification, replacing or augmenting the
  text-serialised-prosody representation.
- Self-consistency vote (N=3) replacing the hand-built confidence, using vote
  margin as the uncertainty signal. ~9s at 4% of post-Phase-2 runtime.
- Revive or retire `detect_language()` for the code-switch path, now that §4.4
  gives it a caller.
- Re-fit `PARAKEET_MIN_LOGPROB` on real non-English examples.
- Calibrate confidence against outcomes rather than constructing it.

---

## 7. Explicitly excluded

| excluded | reason |
|---|---|
| A smaller Whisper model | Parakeet delivers a larger speedup with no accuracy trade. The smaller-Whisper option existed only to buy latency and is now redundant. |
| `parakeet-tdt-0.6b-v3` as the primary | Truncated call_002 from 94 to 19 tokens; routes worse. Retained as fallback only (§4.5). |
| NeMo | §4.2. |
| Re-tuning any existing threshold | They are already fitted on these same three calls. Refitting deepens the circularity rather than reducing it. |
| Batch API adoption | Recorded as an option in §5; changes the latency contract. |
| Removing SQUIM entirely | Gating preserves the impaired path for when a genuinely impaired example arrives. Deleting it would make the field permanently unfalsifiable. |
| Parallelising across files | 2 cores. Measured 1-core penalty is 2.30×, so two 1-thread workers is net worse than one 2-thread worker. |

---

## 8. Testing

- **Golden-output fixture** (1.5) is the primary gate for Phases 2–3.
- **Determinism test**: assert VAD, diarization, ASR and SER are bit-identical
  across two calls in one process. This currently passes (§2.4) and would catch a
  regression introduced by the ASR swap or the VAD dedupe.
- **Word-reconstruction test**: Parakeet token timestamps → `Word` list must
  round-trip against a committed fixture, including the leading-space grouping
  and the final-word end boundary.
- **Router test**: assert the English calls route to Parakeet and call_002 routes
  to Whisper, with the logprob values pinned. Marked as fitted on n=1.
- **Leak test**: after `complete`, `fail` and a validation-failed `enqueue`,
  assert no audio remains under the workdir *or* the recorded upload-cache path.
  The validation-failure case is the one that currently leaks and has no test.
- **Cost-observability test**: assert `cached_tokens` and `system_fingerprint`
  are persisted, so §5's open question cannot silently reopen.
- CI-safety is preserved: `jobs.py` must stay import-clean of `pipeline`,
  `schema`, `torch` and `gradio`. `asr_parakeet.py` imports `onnx_asr` lazily.

---

## 9. Limitations this design does not remove

- **n=3.** Every accuracy figure remains a regression check, not a validation.
  Two of the six "working" fields have constant ground truth. Phase 1's gate
  makes changes *visible*; it does not make them *correct*.
- **Threshold circularity.** 23 MEASURED thresholds are fitted on the same three
  calls they are scored against. Phase 2 adds one more (`PARAKEET_MIN_LOGPROB`,
  n=1).
- **`satisfied` has never been emitted.** Until Phase 3, one of five tone labels
  remains effectively unreachable.
- **Cost compliance depends on spot pricing.** Phase 2 improves the margin but
  does not change that on-demand CPU remains marginal.
- **The diarization code-switch defect** (`xfail`, 12.36s attributed to the
  customer against a true ~0.9s) is not addressed here. Phase 2's router makes
  the fix reachable; Phase 3 would implement it.

---

## 10. Build order

1. **1.5** golden-output fixture — first, because everything after it needs the gate
2. **1.1** audio leaks — highest severity, independent of everything else
3. **1.2** Tier C intensity, **1.3** model pinning + observability, **1.4** interruptions
4. **2.4** VAD dedupe, **2.5** ECAPA batching — lowest-risk latency work, validates the gate
5. **2.3** SQUIM gating
6. **2.2** boundary duplication fix — before the ASR swap, so the two changes are separable
7. **2.1** Parakeet router — the highest-value and highest-risk change, last in Phase 2
8. **2.6** re-measure, re-cost, decide `gpt-4.1-mini`
9. **Phase 3** — only once 30–50 labelled calls exist
