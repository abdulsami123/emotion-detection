# Technical Memo — Voice Tone & Background Noise

**AutoAce AI technical trial.** Prepared 2026-08-17.

---

## 1. Headline result

Measured on the three provided labelled calls. This is a **regression check, not a validation
set** — with n=3 it establishes that the system reproduces known-good behaviour and nothing more.

| field group | metric | result |
|---|---|---|
| `background_noise_present` | accuracy | **1.00** (3/3) |
| `background_noise_severity` | macro-F1 | **1.00** |
| `audio_quality` | accuracy | **1.00** (3/3) |
| `speaker_overlap_present` | accuracy | **1.00** (3/3) |
| `long_silence_present` | accuracy | **1.00** (3/3) |
| `background_noise_type` | accuracy | 0.67 (2/3) |
| `emotional_intensity` | accuracy | 0.67 — **equal to** the constant-`medium` baseline |
| `emotional_tone` | accuracy | 0.33, macro-F1 0.25 |

**The six deterministic signal fields are effectively solved; tone is not.** That split is the
honest summary and it is not an accident — it is what the architecture was designed around after
early analysis showed the tone problem is far harder on this data than it first appears.

*Tone and intensity are from the **live** `gpt-4o-mini` path, measured 2026-08-17.*

**The tone classifier is OpenAI `gpt-4o-mini`.** An earlier Anthropic Haiku 4.5 implementation was
never verifiable — that key returned HTTP 401 — and the provider was switched at the trial owner's
request. Only `classify_tone` is vendor-specific: `build_prompt` and `parse_response` are
provider-agnostic, so the prompt design survived the swap untouched.

**Tone improved from 0/3 to 1/3 during that work, and the cause was not the model.** Agent and
customer roles were **inverted on call_001**, so the tone branch was analysing the bot's uniformly
flat TTS delivery as the customer's. See §6(e) — the single largest defect found in this build.

---

## 2. Approaches compared

The brief requires comparing at least two materially different approaches. Three were built and
measured; several more were built and **refuted**, which is reported in §6 because the negative
results shaped the final design more than the positive ones.

| approach | role | status |
|---|---|---|
| **Prosody + SER + LLM over an annotated transcript** | primary tone | built; **unverified live** (401) |
| **Zero-shot NLI (`bart-large-mnli`) over the transcript** | second approach, no-API fallback | built and measured: tone 0.33 |
| **Deterministic DSP + AudioSet tagging** | all six signal fields | built and measured: 5 of 6 at 1.00 |

### Why the tone design is prosody-primary

Text alone scores at or below chance on this data. These are **AI-voice-agent calls** — a TTS bot
("Erica", a dealership receptionist) talking to human callers:

| call | label | what the customer actually says | what a text classifier reads |
|---|---|---|---|
| call_001 | **upset / high** | "Are you a real person?" then "Hello?" ×5 — ~8 words | neutral or confused |
| call_002 | **neutral / medium** | "Spanish, please." — two words | no signal at all |
| call_003 | **satisfied / medium** | long; repeatedly told the dealership is closed | **frustrated** |

call_003 is decisive: semantically the caller is blocked over and over, and the label is
`satisfied` because they stay warm and close with "Thank you". The lexical content points the
wrong way. So the design makes prosody primary and lets the transcript supply *situation* rather
than sentiment — and the system prompt encodes both failure directions explicitly (politeness
masking, and escalating repetition with neutral words).

The NLI result confirms the analysis: it returns `distressed` for both call_001 and call_002,
i.e. it reads short, fragmentary caller speech as distress. Text-only is not viable here.

---

## 3. Final architecture

```
raw audio -> decode 16k mono (no loudness normalization, ever)
             |
             +-- Silero VAD ---------------- shared speech / non-speech segmentation
             +-- ECAPA 2-cluster ----------- agent vs customer
             +-- faster-whisper ------------ transcript, word timestamps, language
                        |
      +-----------------+------------------+
      |                                    |
  SIGNAL BRANCH (raw audio)          TONE BRANCH (customer segments)
  floor -> present, severity         eGeMAPS -> speaker-relative z -> tags
  AST   -> type                      audeering SER -> arousal/dominance/valence
  SQUIM + DSP -> audio_quality       gpt-4o-mini over annotated transcript
  ECAPA intra-segment -> overlap        \-> NLI fallback if unavailable
  dead air -> long_silence           activation x trajectory -> intensity
      |                                    |
      +-----------------+------------------+
                        |
              merge + confidence from voter agreement -> 9-field JSON
```

**Absolute level is never normalized.** The noise floor in dB is both the primary presence
detector and the primary severity signal; normalizing gain destroys it. The tone branch obtains
level-invariance internally by z-scoring against each speaker's own baseline.

**The signal branch never touches an API or a network.** That is why it still returned all six
fields correctly while the tone branch was degraded by an invalid API key — the worst available
failure mode would have been losing the reliable points because a remote service was down.

---

## 4. Cost analysis

**Ceiling: $0.003 per audio minute.**

**Token usage is measured against the live API, not estimated.** A 30.9-second call sends
**2712 input tokens** and returns 124 output tokens (`gpt-4o-mini`) or 205 (`gpt-4.1-mini`). The
earlier figure of ~830 input tokens was an estimate and was wrong by 3.3× — the third cost-model
error in this build, all three found by measuring rather than reasoning.

Roughly **2500 of those input tokens are the fixed system + few-shot prefix**, so LLM cost per
*audio minute* is dominated by fixed prompt overhead on short calls and amortises on long ones:

| call length | input tokens | `gpt-4o-mini` $/call | $/audio-min |
|---|---|---|---|
| 0.5 min (call_001) | 2712 | $0.00048 | **$0.00093** |
| 2.9 min (call_003) | ~3700 | $0.00071 | **$0.00025** |

**Prompt caching is reachable here and was not on the previous provider.** OpenAI caches prefixes
over 1024 tokens automatically at half price and the ~2500-token prefix qualifies, so the few-shots
sit in a stable `_system_content()` prefix with only the per-call payload in the user message.
Haiku's 4096-token cache minimum was never met by this prompt, so this is a genuine gain from the
switch rather than a wash.

Rates assumed: `gpt-4o-mini` $0.15 / $0.60 per MTok, `gpt-4.1-mini` $0.40 / $1.60. **These should be
re-verified against OpenAI's current pricing page before submission** — they are the one input here
not measured directly from the API.

`gpt-4.1-mini` is the documented upgrade candidate at roughly 3× the input cost. Both models
returned the correct `upset / high` on call_001 in an isolated smoke test, so the choice is not
decidable on three labelled calls and is recorded as an A/B for a larger set.

### Measured compute

| path | instance | $/hr | s/audio-min | compute | **total** | vs ceiling |
|---|---|---|---|---|---|---|
| CPU | 4 vCPU spot | $0.05 | 62 | $0.00086 | **$0.00219** | 27% headroom |
| CPU | 4 vCPU on-demand | $0.10 | 62 | $0.00172 | **$0.00305** | ❌ breaches by 2% |
| GPU | T4 spot | $0.20 | ~14 | $0.00078 | **$0.00211** | 30% headroom |
| GPU | T4 on-demand | $0.35 | ~14 | $0.00136 | **$0.00269** | 10% headroom |

**CPU break-even: $0.0968/hr.** GPU figures are projected from stage profiles, not measured — no
CUDA device was available. **Recommended deployment: GPU**, which holds across the realistic
price range where CPU-on-demand does not.

### Excluded, rather than silently omitted

One-time model download and warm-up; dashboard hosting (fixed, not per-minute); storage. Other
assumptions: batch utilisation ≥70%, model weights resident across calls, no retry overhead.

### The cost model was wrong twice, in opposite directions

Worth stating plainly, because it is the most transferable lesson here:

| stage | estimated | measured | error |
|---|---|---|---|
| ASR | 60 s/audio-min | **24.5** | 2.4× too pessimistic |
| SQUIM quality | 1.5 s/audio-min | **28.6** | 19× too optimistic |
| AST typing | 1.0 s/audio-min | **30.9** | 31× too optimistic |

The ASR figure was inherited from an `openai-whisper` run rather than measured on the
CTranslate2 build actually shipped. SQUIM and AST were guessed. Correcting them **inverted** the
original conclusion that "GPU is both faster and cheaper" — at spot pricing the two are within
noise, and GPU is now recommended on **latency**, not cost.

Both heavyweight stages also scaled with call *duration*. But `background_noise_type` and
`audio_quality` are call-**level** properties, so neither needs every window: sampling a fixed
number of evenly-spaced windows (AST 6, SQUIM 3) made both **O(1) per call**, taking the 172-second
call from 1.13× to 0.36× realtime with every assertion preserved.

---

## 5. Latency analysis

Measured end to end, CPU only (torch 2.13.0+cpu, no CUDA), models resident:

| call | audio | wall-clock | × realtime |
|---|---|---|---|
| call_001 | 30.9s | 57.3s | 1.85× |
| call_002 | 35.0s | 58.4s | 1.67× |
| call_003 | 171.9s | 129.2s | **0.75×** |
| **total** | 237.8s | 244.9s | **1.03×** |

Short calls are dominated by fixed per-call costs (model warm-up, the O(1) AST/SQUIM samples), so
the marginal rate on long calls is the meaningful one. A 50-call batch of 3-minute calls runs in
roughly 1.9 hours on one CPU worker; stages are per-file independent, so a pool of 4 gives close
to linear speedup.

**CPU-only meets a batch target but not an interactive one.** GPU is the recommendation for
throughput.

---

## 6. What measurement refuted

Four design decisions did not survive contact with the data. These are reported because a design
doc that only records what worked is not a rigorous artifact.

**(a) "Tag non-speech regions only" made typing worse.** The argument was that speech masks
background noise. Measured, it inverts: whole-clip AST maps call_002 to its ground-truth `TV`,
while non-speech-only returns `music`. In telephony the gaps are near-**silent** — which is
exactly *why* the noise floor works as a presence detector — while the television is audible
*underneath* the speech. Removing the speech removed the evidence.

**(b) The spectral transmission-noise classifier does not work.** AudioSet genuinely is blind to
line artifacts (`static` scores 0.0013–0.0059 on every call, including the one whose ground truth
*is* "sharp static"). But no proposed replacement discriminates either:

| call | truth | flatness | transients/s | crest | kurtosis |
|---|---|---|---|---|---|
| call_001 | *(none)* | 0.0531 | **0.47** | **36.5** | **542** |
| call_002 | TV | 0.1157 | 0.28 | 12.9 | 81 |
| call_003 | **sharp static** | **0.0426** | 0.29 | 21.0 | 109 |

Flatness is *lowest* on the static call where broadband hiss should be highest; transient rate,
crest and kurtosis are all *highest* on the clean call; a 6 dB mains-hum test fired on all three.
The detector is **disabled** rather than shipped as an arbitrary-firing label, and call_003's type
is pinned as `xfail(strict=True)` with the measurements in the reason. This is the one field where
we knowingly emit a wrong answer (`radio` instead of `sharp static`).

**(c) Two threshold sets were internally inconsistent with their own anchors.**
`NOISE_SEVERITY_BANDS` placed the low/medium boundary at −50.0 while annotating −52.1 as a
`medium` anchor — so that anchor fell into `low` by construction, and call_002's severity was
wrong. Separately, the coherence-check valence bounds (0.45 / 0.55) sat *inside* the measured SER
range (valence spans only 0.534–0.638 on this audio), so the check would have fired on roughly
half of all calls regardless of correctness, penalising correct answers through the confidence
formula. Both corrected, both now covered by tests that assert the label the schema actually
emits rather than merely that values are ordered.

**(d) The intensity self-baseline was degenerate on short calls.** `BASELINE_WINDOW_S` is 25s but
the two short calls have only 7.1s and 12.4s of customer speech, so the baseline was built from
**all** the segments it then scored. Z-scoring a set against its own mean gives exactly zero *by
construction*:

| call | customer speech | baseline coverage | mean activation z |
|---|---|---|---|
| call_001 | 7.1s | **100%** | **−0.000** |
| call_002 | 12.4s | **100%** | **+0.000** |
| call_003 | 73.8s | 37% | +0.240 |

Zero activation trips the `low` threshold, so both short calls reported `low` from no evidence at
all — dragging intensity to 0.33, **below** the 0.67 majority-class baseline. Degenerate baselines
are now detected and routed to the Tier C path (emit the prior, cap confidence). Intensity rose to
0.67 and confidence on those calls dropped to 0.25, so the *absence* of a measurement is visible
instead of hidden behind a fabricated label.

---

**(e) Agent and customer roles were inverted on one of three calls — the largest defect found.**
Diarization clusters voices correctly, then has to decide which cluster is the bot, and its anchor
was *"the agent greets first"*. On call_001 the caller opens with an impatient `"Come on."` at 1.2s
and the bot greets at 3.7s, so that anchor picked the wrong cluster and **inverted every role in
the call**:

| segment | text | assigned (before) |
|---|---|---|
| 1.2–2.0 | "Come on." | agent ✗ |
| 3.7–6.9 | "Hi, I'm Erica from Toyota of Braintree. How can I help?" | customer ✗ |
| 8.7–11.3 | "Yes, hi. Are you a real person?" | agent ✗ |

The tone branch therefore spent the whole run analysing the bot's uniformly flat TTS delivery as
if it were the customer's, which is why tone scored **0/3 regardless of provider** — the same
model, given the caller's actual utterances by hand, returned the correct `upset / high`.

Fixed by `pipeline.verify_roles`, which re-anchors on **transcript content** (phrases a
receptionist structurally says — "how can I help", "transferring you", plus Spanish equivalents)
rather than on turn order, and only swaps on a strict majority of evidence. Phrases are
deliberately *not* name-specific, since the hidden set may use a different bot or dealership.
Measured: call_001 swaps and its tone becomes correct; calls 002 and 003 were already right and
are untouched.

Two lessons worth carrying: an assumption that holds on two of three examples is not a rule, and
the test asserting it (`test_agent_speaks_first`) claimed it was *"true on all three provided
calls"* while only ever checking the one where it held. Both are now corrected, with the
counter-example pinned.

---

## 7. Calibration position

**Thresholds are derived from definitions and physics, then validated against three points — not
fitted.** Every threshold lives in `autoace/config.py` and is annotated `MEASURED`, `DERIVED` or
`UNFITTED`. Class coverage in the labelled set is severely incomplete:

- `emotional_tone`: 3 of 5 classes (`frustrated`, `distressed` never appear as truth)
- `emotional_intensity`: 2 of 3 (`low` never appears)
- `background_noise_severity`: 2 of 4 (`low`, `high` never appear)
- `audio_quality`: 1 of 3 (only `clear`)
- `long_silence_present`: 1 of 2 (only `false`)

Two boolean fields are fully covered. **Five of the eight classified fields lack full coverage**,
so the `low`/`medium` intensity boundary and the `low`/`high` severity boundaries have no
supporting example at all.

**Leakage prevention.** No model is trained; the LLM is prompted and the thresholds are reasoned,
so leakage risk is confined to prompt engineering against three examples. Mitigations: the three
calls are a regression test rather than a tuning set; few-shot examples are **synthetic** rather
than drawn from the labelled calls; and the ECAPA agent-reference mechanism uses speaker identity
only and never reads `labels.csv`. No production module under `autoace/` reads the labels file.

**The three-call figures are therefore not an unbiased accuracy estimate**, and the confusion
matrices are a formality at this n.

---

## 8. Failure modes and limitations

1. **Tone is unverified on its primary path.** The API key returns 401; the measured 0.33 is the
   NLI fallback. Highest-priority item.
2. **Code-switching breaks speaker assignment.** On call_002, ECAPA separates the bot's *English*
   voice from its own *Spanish* voice rather than bot from human: **12.36s attributed to the
   customer against a true ~0.9s**. The `degraded` flag does not fire because the clusters are
   well separated — along the wrong axis. Proven unfixable at cluster level (the customer's single
   0.9s segment never forms its own cluster, so the floor is 5.2s). Pinned `xfail(strict=True)`.
   **The fix is now proven viable:** `asr.detect_language()` returns `en` on call_002's opening 8s
   and `es` on its final 12s, so per-segment language ID can merge the two clusters.
3. **Line-artifact typing is wrong by design** (§6b). `background_noise_type` will name the
   nearest environmental source for a call whose real noise is a codec or line artifact.
4. **Speaker assignment can be decided by noise.** On call_001 a reference similarity of 0.5529
   sits 0.003 above the 0.55 threshold.
5. **SER axes are compressed** on telephony audio (valence 0.534–0.638, arousal 0.594–0.646), far
   from the [0,1] the model nominally emits, so absolute-threshold logic against them is fragile.
6. **Averaging dilutes short emotional peaks.** SER and AST both summarise at call level from a
   fixed window sample; a brief escalation outside those windows is invisible.
7. **Multilingual prosody.** eGeMAPS speech-rate features are language-sensitive, and both the SER
   model and the LLM's prosody interpretation are English-centric. call_002 is Spanish, so this is
   live on the provided set, not hypothetical.
8. **No cross-validation is possible.** Leave-one-call-out on n=3 is not meaningful.

---

## 9. Hosted dashboard and deployment (brief §7)

Brief §7 asks for a hosted dashboard: authenticated login, batch upload,
validation, visible progress, per-file failure isolation, a review queue, and
exports. §13 of the design doc already specified all of that and `app.py`
already implemented it — **synchronously**. That does not survive contact
with the measured numbers below, so this section covers what changed to make
it asynchronous and durable, and where it now runs.
Full design: `docs/superpowers/specs/2026-08-19-hosted-dashboard-design.md`.

### 9.1 Why synchronous does not work

A 50-file batch runs at **~87 minutes** (§9.2). No HTTP request survives
that, and no evaluator should have to keep a browser tab open for it. The
fix is a job queue: `autoace/jobs.py` (new, SQLite/WAL — schema, enqueue,
exclusive claim, complete, fail, reconcile, expire) and `autoace/worker.py`
(new, the drain loop) run as two systemd units against one on-disk store.
`app.py`'s `run_batch` splits into `enqueue_batch` (validate, insert rows,
return a job ID immediately) and `poll_job` (read-only projection of the
store into the existing table/export shape via a `gr.Timer`). `jobs.py`
imports nothing from `pipeline`, torch, or `gradio`, which is what keeps its
tests fast and CI-safe (§9.8).

A thread was considered and rejected: openSMILE and the DSP paths do not
reliably release the GIL, so a worker thread would stall the UI exactly as
badly as inline processing does.

### 9.2 Measured constraints

Every figure below is measured on the three provided calls, not estimated —
worth stating because every cost figure in this project that came from
arithmetic instead of measurement has been wrong (§4, §6): ASR 2.4× too
pessimistic, SQUIM 19× and AST 31× too optimistic, LLM input tokens 3.3×
under.

| Quantity | Value |
|---|---|
| Peak worker memory | **5.67 GiB** (Windows peak working set, all three calls in one process, NLI-fallback path) |
| Steady-state memory | **3.91 GiB** |
| Per-file latency, 16 threads | 58.2 / 58.7 / 129.8 s (calls 001/002/003) |
| Per-file latency, 2 threads | 64.6 s (1.11×) / 160.8 s (1.24×) |
| Per-file latency, 1 thread | 148.6 s (2.30×) / 371.1 s (2.31×) — *against the 2-thread figures* |
| 50-file batch, 2 threads | **~87 min** (~105 s/file × 50) |
| 50-file batch, 1 thread | **~202 min** (~242 s/file × 50) |
| Slow worker integration test | 170 s end-to-end on one real call |
| Python dependency footprint | ~1.5 GiB (torch 527M, gradio 193M, llvmlite 117M, scipy 115M, transformers 97M, ctranslate2 60M) |
| Model weight cache | ~5 GiB |

Two consequences drive the deployment shape:

**Cores above two buy almost nothing, but the second core is mandatory.**
Going from 16 threads to 2 costs only 1.11–1.24×. Going from 2 to 1 costs
**2.30×** (148.6 s and 371.1 s against 64.6 s and 160.8 s) — a cliff, not a
taper, and consistent across both calls. A 50-file batch is ~87 min on two
cores and ~202 min on one. So `OMP_NUM_THREADS=2` in the worker unit is a
floor, not a tuning knob, and the instance must have two OCPUs.

Given two cores, **RAM is what sizes the box**, not CPU. `WhisperModel` is
constructed without `cpu_threads`, so CTranslate2 derives its intra-op count
from `OMP_NUM_THREADS` — the cap was genuinely applied in every run, so the
non-linearity is a property of the pipeline and not a measurement artefact.

**5.67 GiB is the *fallback* path, and we size for it anyway.**
`bart-large-mnli` (~1.6 GB) loads only inside the `except` handler in
`pipeline.py`; it is not a co-voter — the two voters are the LLM and the
dimensional SER. The deployed happy path should be materially lighter, but an
OpenAI outage must not OOM-kill the worker, so the box is sized against the
worse number.

**Caveat.** Windows peak working set is not Linux RSS. The magnitude should
hold; the exact number will differ on the VM and must be re-measured there.
Also unmeasured: the happy-path (LLM available) peak, which needs a valid
API key.

### 9.3 Free tiers refuted

Recorded so none of these get re-proposed:

| Option | Why it fails |
|---|---|
| Render free | 512 MB RAM, no persistent disk. Off by 11× against the 5.67 GiB peak. |
| Hugging Face Spaces free | Gradio/Docker Spaces now require a paid plan (PRO for personal accounts); only Static Spaces are free. |
| HF ZeroGPU (the free-account exception) | 5 minutes of GPU **per day**; `@spaces.GPU` is request-scoped and cannot host a long-running worker. |
| Railway | No free tier; $5 trial credit only. |
| Fly.io / Koyeb / Northflank | 256–512 MB free allowances. |
| AWS / GCP / Azure free VMs | 1 GB micro instances. |

**Selected: Oracle Cloud Always Free**, `VM.Standard.A1.Flex`, 2 OCPU / 12 GB,
Ubuntu 24.04 aarch64, Python 3.12. 2 OCPU rather than the free 4 because §9.2
shows the extra cores buy only 1.1–1.2×, while smaller shape requests are
markedly more likely to be granted — Oracle ARM returns `Out of host
capacity` frequently in popular regions. All ARM-risk dependencies publish
`manylinux_aarch64` cp312 wheels: `torch` 2.13.0, `ctranslate2` 4.8.1, `numba`
0.67.0, `opensmile` 2.6.0, `soundfile` (pure-python).

**TLS is Caddy plus a DuckDNS hostname, no domain purchased.** DuckDNS
specifically, not `nip.io`/`sslip.io`: Let's Encrypt scopes its 50-cert-per-
registered-domain rate limit using the Public Suffix List. `duckdns.org` is
on the PSL, so each `<name>.duckdns.org` gets its own quota; `nip.io` and
`sslip.io` are not on the PSL, so every certificate for either shares one
chronically-exhausted quota. Verified against the live list — a correctness
difference, not a preference. A 4 GiB swapfile insures the gap between the
3.91 GiB steady state and the 5.67 GiB peak; slow beats an OOM kill.

### 9.4 Privacy posture (brief §5)

- Audio is unlinked **per file**, as soon as its row is recorded — not at end
  of job. It sits on disk only for the ~90 s it is being processed.
- Results carry no audio. Retention is 7 days, then swept.
- **TLS terminates in Caddy on our own VM**, so no third party sees request
  plaintext. Every managed-platform option in §9.3 terminates TLS on someone
  else's edge. DuckDNS provides DNS resolution only and carries no traffic.
- The one genuine external dependency is OpenAI (§10), which receives the
  annotated **transcript**, not audio — already disclosed there and
  unchanged by this work.
- Gradio binds `127.0.0.1`, so Caddy is the only off-box listener and a wrong
  firewall rule cannot expose plaintext HTTP.

### 9.5 Durability and failure handling

The unit of work is one file (58–161 s), so any interruption costs at most
one file. `complete()`/`fail()` are atomic in SQL; filesystem work (deleting
the audio, removing the workdir) follows the commit, because an `rmtree`
cannot be rolled back if the commit then failed.

`reconcile()` runs on worker startup: it requeues orphaned `running` rows,
abandons rows past `MAX_ATTEMPTS = 2`, and finalises jobs orphaned with no
outstanding files.

**The retry cap is load-bearing, not a nicety.** Without it, a file that OOM-
kills the worker gets requeued by `reconcile()` and dies again on the same
input — forever. `attempts` increments at *claim* time rather than at
failure time, so a process killed before running any Python still counts
against the cap. This was found by review, not by inspection, and confirmed
empirically: a job stuck at `running` with 0 outstanding files, workdir and
audio still on disk.

One bad file marks its own row `failed` with the reason and appears in the
table **and** the exports — the batch continues rather than aborting.
Validation failures are reported before any row is inserted and before any
inference runs.

### 9.6 CI is structurally limited, and that is deliberate

`reference/` is gitignored because the audio is confidential (brief §5) and
the model weights are ~5 GiB, so GitHub Actions can only run tests needing
neither: `jobs.py`, `fuse.py`, schema, manifest, and the app's
validation/export tests. This is attributable to §5, not to thin coverage —
the pipeline and worker integration tests exist and pass locally against the
provided audio but cannot run in a public CI environment.

### 9.7 Two brief §7 gaps found in this build

Reported as found-and-fixed, not as new features:

1. **The results table displayed only 6 of the 9 schema fields.**
   `background_noise_present`, `speaker_overlap_present`, and
   `long_silence_present` were exported to CSV/JSON but never shown in the
   dashboard. Brief §7 requires the *displayed* prediction to use the
   required output schema, and §8 scores result review inside the 10%
   dashboard weighting, so exported-but-invisible did not count. Also fixed
   in the same pass: `results_to_table`'s sort key indexed columns by
   hard-coded position and would have silently sorted on the wrong column
   the next time a column was inserted.
2. **A manifest with a UTF-8 BOM would have failed an entire batch.**
   `load_manifest` opened with `encoding="utf-8"`, so an Excel-saved CSV —
   the likely form of an evaluator-supplied manifest — makes the first
   fieldname `﻿name`, and every `record["name"]` then raises `KeyError`
   before any inference runs. Confirmed as a real `KeyError`, not
   theoretical. Fixed to `utf-8-sig`, alongside handling for an omitted
   `result_json` column (which brief §7 explicitly permits for an unlabeled
   hidden test set), blank trailing rows, whitespace-padded names, and
   errors that name the offending row.

### 9.8 Limitations

1. **One worker: jobs are FIFO across users.** A second batch waits up to
   ~87 minutes behind the first. Surfaced as queue position and an ETA
   rather than a silent wait.
2. **Peak memory measured on Windows, not Linux ARM** — must be re-measured
   on the VM before the §9.2 figures are trusted there.
3. **The happy-path (LLM available) peak was never measured**; it needs a
   valid API key.
4. **Oracle ARM `Out of host capacity`** can block provisioning entirely;
   there is no in-repo fix, only the smaller-shape mitigation in §9.3.
5. **A file that dies twice is reported `failed` rather than analysed** —
   the alternative is a crash loop (§9.5).
6. `shutil.rmtree(ignore_errors=True)` silently no-ops on Windows when a
   handle is still open, so `_remove_workdir` reports whether the directory
   is actually gone rather than trusting the call. The exposure is lower on
   the Linux deployment target.
7. **Swap can mask memory pressure as latency.**
8. **No backups.** Results are reproducible and audio should not persist
   past processing, so none are taken.
9. **CI cannot cover the pipeline or worker end-to-end** (§9.6) — a
   consequence of brief §5, not of thin test-writing.

---

## 10. External API disclosure (brief §11)

| item | detail |
|---|---|
| Model | **`gpt-4o-mini` (OpenAI)** |
| Pricing assumed | $0.15 / MTok input, $0.60 / MTok output — **re-verify before submission** |
| Measured usage | 2712 input / 124 output tokens on a 30.9s call |
| Per audio minute | **$0.00093** on a 0.5-min call, **$0.00025** on a 2.9-min call |
| Upgrade candidate | `gpt-4.1-mini` at ~3× input cost; A/B pending a larger labelled set |
| Retention | zero-retention / no-training to be confirmed on the account before production use |
| **Does audio leave AutoAce infrastructure?** | **No.** Audio is never transmitted. |
| What does leave? | The customer-side transcript, discretised prosody tags, the SER triple, and an agent-behaviour summary |
| Fallback if egress is refused | `autoace/tone_nli.py` runs `bart-large-mnli` locally; no network at all |

The signal branch makes no external calls whatsoever. Confidential audio and transcripts derived
from it are excluded from version control (`reference/`, `tests/fixtures/asr_baseline.json`) because
the target repository is public.

---

## 11. Next steps, in priority order

1. **Fix code-switched speaker assignment** using per-segment `detect_language()` — viability
   already demonstrated (§8.2). It is now the **direct cause of one of the two remaining tone
   errors**: call_002 attributes 12.4s to a customer who speaks ~0.9s, so the classifier reads the
   bot's Spanish as the caller's.
2. **Attack politeness masking**, the other remaining tone error. call_003 is labelled `satisfied`
   while the caller is repeatedly blocked; the prompt carries an explicit rule for this and the
   model still reads the situation as frustration. Candidate signals: closing-turn sentiment
   weighted above mid-call content, and SER valence trend rather than call-level mean.
3. **Collect 100–300 labelled calls**, weighted toward the absent classes. Every remaining item is
   gated on this.
4. **Fit rather than reason** the activation weights and severity bands, and run a genuine
   stratified evaluation with per-class figures.
5. **A/B `gpt-4.1-mini` against `gpt-4o-mini`.** LLM cost is now a small fraction of the ceiling
   ($0.00025–$0.00093 per audio minute against $0.003), so the budget comfortably permits the
   stronger model — the constraint is evidence, not money. The OpenAI Batch API halves cost again
   if throughput allows.
6. **Find a real line-noise discriminator** for `background_noise_type`, which needs labelled
   examples of static, hum and crackle.
7. **Deploy to the VM and re-measure §9.2 on Linux ARM** — the peak-memory figure is currently
   Windows-only and is the last unverified number in the hosting section.
