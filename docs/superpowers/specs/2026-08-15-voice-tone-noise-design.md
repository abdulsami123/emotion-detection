# AutoAce Voice Tone & Background Noise — Design

**Date:** 2026-08-15
**Status:** Approved for implementation
**Deliverable:** AutoAce AI technical trial — hosted dashboard + pipeline + memo

---

## 1. Objective and constraints

Classify emotional tone and detect background noise in production call audio, emitting a
fixed 9-field JSON schema per clip. Scored on a hidden test set.

Binding constraints from the trial brief:

| Constraint | Value |
|---|---|
| Inference cost | ≤ **$0.003 per audio minute** (not per call) |
| Latency | Reasonable for production batch; must be measured and reported |
| Reproducibility | Runnable on new audio with documented setup |
| Generalization | No access to hidden test set; no tuning against it |
| Data handling | Production audio is confidential; no upload to unapproved services |
| Disclosure | Any paid API must be disclosed with model, pricing, retention, egress |
| Deadline | Friday 7:00 p.m. ET |

Scoring weights: hidden-set performance 45%, cost efficiency 15%, technical rigor 15%,
production practicality 10%, dashboard 10%, communication 5%.

### 1.1 Output schema

```
emotional_tone            enum  neutral | satisfied | frustrated | upset | distressed
emotional_intensity       enum  low | medium | high
background_noise_present  bool
background_noise_type     str   free text; "" when absent
background_noise_severity enum  none | low | medium | high
audio_quality             enum  clear | slightly_impaired | severely_impaired
speaker_overlap_present   bool
long_silence_present      bool
confidence                float 0.0–1.0
```

The brief states explicitly: **do not infer frustration or distress solely from loudness, and
do not infer background noise solely from poor audio quality.** Tone, noise, and audio quality
are scored separately and are computed by separate, independent code paths in this design.

---

## 2. What the data told us

### 2.1 These are AI-agent calls

The "agent" is a TTS voice assistant ("Erica") deployed at Toyota dealerships. The "customer"
is a human caller. This drives three consequences that shape the entire architecture:

1. **Customer utterances are very short.** call_002's customer says two words total.
2. **Tone lives in the voice, not the words.** See 2.3.
3. **The agent voice is consistent across calls**, which makes speaker assignment cheap (§4.3).

### 2.2 Ground truth (`labels.csv`)

| | tone | intensity | noise | type | severity | quality | overlap | silence |
|---|---|---|---|---|---|---|---|---|
| call_001 | upset | high | false | "" | none | clear | false | false |
| call_002 | neutral | medium | true | TV | medium | clear | true | false |
| call_003 | satisfied | medium | true | sharp static | medium | clear | true | false |

`confidence` is a constant 0.82 across all rows — it carries no gradient and is not
meaningfully scoreable. We satisfy it with a defensible ensemble-agreement number and do not
optimize it.

**Class coverage is severely incomplete.** Zero examples of: `low`/`high` noise severity,
`low` intensity, `slightly_impaired`/`severely_impaired` quality, `long_silence: true`, and
two of five tone classes (`frustrated` and `distressed` never appear).

Precisely: of the eight classified fields, **five have incomplete class coverage** —
`emotional_tone` (3 of 5), `emotional_intensity` (2 of 3), `background_noise_severity` (2 of 4),
`audio_quality` (1 of 3), `long_silence_present` (1 of 2). Two booleans are fully covered
(`background_noise_present`, `speaker_overlap_present`) and `background_noise_type` is open text.

**Consequence:** thresholds in this system are **derived from feature definitions and physics,
then validated against three points**. They are not fitted. All of them live in one config
file (§9) and the three calls act as a regression test. This is stated plainly in the memo.

### 2.3 Text alone would score at or below chance

| call | label | customer's actual words | what a text classifier reads |
|---|---|---|---|
| 001 | **upset / high** | "Yes, hi. Are you a real person?" then "Hello?" ×5 | neutral / confused |
| 002 | **neutral / medium** | "Spanish, please." | no signal at all |
| 003 | **satisfied / medium** | repeatedly told the dealership is closed, dates confused | **frustrated** |

Call 003 is decisive: semantically the customer is being blocked over and over, and the label
is `satisfied` — because they stay warm and polite and close with "Thank you". The lexical
content points the wrong way.

**Therefore: prosody is the primary tone signal; the transcript supplies *situation*, not
sentiment.**

### 2.4 Bot outcome predicts customer emotion

| call | what the bot did | tone |
|---|---|---|
| 001 | failed to respond; dead-air "Hello?" loop | upset |
| 002 | switched to Spanish on request | neutral |
| 003 | handled request, transferred to a human advisor | satisfied |

Customer emotion in this domain is largely a function of **whether the bot worked**. This is
extracted as an explicit feature (`agent_behavior`, §5.2). Holds on 3/3 — a hypothesis, not a
finding, and labelled as such in the memo.

### 2.5 Measured acoustics

| | SNR dB | noise floor dBFS | max non-speech gap | clip % | energy >3.4kHz | label |
|---|---|---|---|---|---|---|
| call_001 | 42.2 | **−56.3** | 2.87s | 0.00 | 4.4% | none |
| call_002 | 29.9 | **−52.1** | 3.29s | 0.00 | 4.3% | TV / medium |
| call_003 | 34.9 | **−47.0** | 7.35s | 0.00 | 6.0% | static / medium |

Four calibration facts, measured rather than guessed:

1. **Noise floor level tracks severity; SNR does not.** Floor is monotonic with the labels
   (−56 none, −52 medium, −47 medium); SNR separates 001 and 003 by only 7dB across a
   none/medium boundary. Floor is the primary severity feature, SNR secondary.
2. **`long_silence_present` is `false` at a 7.35-second gap.** The threshold must sit above
   that, and must require true dead air rather than merely non-speech.
3. **`audio_quality: clear` is the strong prior, and telephony band-limiting must not count
   against it.** All three carry only 4–6% energy above 3.4kHz — normal narrowband telephony.
   A muffling detector baselined on wideband speech flags all three and costs three points.
4. **Percentile-thresholded energy VAD is inadequate** — it ranks the "sharp static" call as
   having the *lowest* non-speech spectral flatness, which is backwards. Real VAD required.

### 2.6 The static/quality labelling conflict

The brief lists **static under `audio_quality`**. The labeller put "sharp static" in
`background_noise_type` and left `audio_quality: clear`.

**We follow the labels.** Operational rule: *`audio_quality` degrades only when speech
intelligibility is materially impaired; everything else audible goes to
`background_noise_type`.* Flagged in the memo as an observed ambiguity.

### 2.7 Enhancement is excluded from the production path

Evidence from an exploratory Resemble Enhance + Parakeet run (since deleted; findings recorded
here because they justify the exclusion):

- On call_001, the enhanced variant **deleted `"Come on."`** — the clearest exasperation marker.
- On call_002, it hallucinated fake Spanish (`"Spañol, mama vuevo. Cause voices me biqué..."`).
- On call_003, it never finished — timed out on CPU.
- It would erase the TV noise and static that `background_noise_present`, `_type`, `_severity`
  and `audio_quality` all depend on, and shift the noise floor that `long_silence_present` uses.

Resemble Enhance is a diffusion model and cannot fit $0.003/min regardless. `denoise()` is
retained only as an optional ASR fallback on badly degraded audio, gated behind a config flag,
and **never** feeds the signal branch or the prosody extractor.

### 2.8 Provenance of the evidence in this section

The workspace contains only `call_001–003.ogg`, `labels.csv`, the brief PDF, and this document.
Every exploratory artifact has been deleted. Claims above therefore fall into two classes, and
the distinction matters for reproducibility:

**Reproducible from surviving files** — re-verified 2026-08-15, all figures matched exactly:

| claim | source | status |
|---|---|---|
| §2.2 label table | `labels.csv` | ✅ verbatim match |
| §2.5 acoustics (SNR, floor, gaps, clip %, HF energy) | the three `.ogg` files | ✅ reproduced exactly |
| §4.1 duplicated-mono (`corr = 1.0000`) | the three `.ogg` files | ✅ reproduced exactly |
| §1 constraints, schema, scoring weights | the brief PDF | ✅ unchanged |

**Recorded-only — cannot currently be re-verified.** These come from exploratory runs whose
artifacts are gone. They are load-bearing for major design decisions, so **regenerating them is
the first task of implementation**, not an optional check:

| claim | what it justifies | regenerate via |
|---|---|---|
| §2.3 transcript excerpts | prosody-primary architecture (the whole tone design) | build step 2 (ASR) |
| §2.4 bot-outcome → emotion | the `agent_behavior` feature | build step 2 |
| §2.7 enhancement damage | excluding enhancement from the production path | one-off, optional |
| §7.1.1 AST vs PANNs | AST over PANNs; floor-primary presence; spectral line-artifact classifier | build step 3 |
| §11 ASR latency (33–46s / 31s file) | the GPU deployment recommendation | build step 2 |

Until each is regenerated as a checked-in test artifact, treat it as a documented prior rather
than a measurement. The §7.1.1 numbers in particular changed two design decisions and are worth
reproducing before any code depends on them.

---

## 3. Architecture

```
                            raw audio (.ogg/.wav/.mp3)
                                      │
                        ┌─────────────▼─────────────┐
                        │   SHARED FRONT END        │
                        │  decode → 16k mono        │
                        │  Silero VAD               │
                        │  ECAPA embed + 2-cluster  │
                        │  agent/customer assign    │
                        │  faster-whisper + words   │
                        └──────┬─────────────┬──────┘
                               │             │
        ┌──────────────────────▼───┐   ┌─────▼────────────────────────┐
        │      TONE BRANCH         │   │      SIGNAL BRANCH           │
        │  (customer segments)     │   │  (RAW full-mix audio)        │
        │                          │   │                              │
        │  eGeMAPS ─┐              │   │  AST/AudioSet → noise type   │
        │  audeering SER (A/V/D) ──┼──►│  floor + SNR  → severity     │
        │  agent_behavior (LLM) ───┤   │  SQUIM + DSP  → audio_quality│
        │           │              │   │  overlap segs → overlap      │
        │  ┌────────▼────────┐     │   │  dead air     → long_silence │
        │  │ Haiku 4.5       │     │   └──────────────────────────────┘
        │  │ structured out  │→tone│
        │  └─────────────────┘     │
        │  activation rule ──►intensity
        └──────────────────────────┘
                               │
                        ┌──────▼──────┐
                        │  FUSE       │ confidence from voter agreement
                        │  + validate │ schema enforcement
                        └─────────────┘
```

The two branches share only decode, VAD, diarization, and ASR. The signal branch needs no
transcript; the tone branch needs no audio tagger.

**Critical invariant: the signal branch always runs on raw, unmodified audio.** Any denoising
applied for ASR is applied to a *copy*.

---

## 4. Shared front end

### 4.1 Decode — `io_audio.py`

```python
def load(path: Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 @ 16 kHz, 16000). Raises UnsupportedAudio on failure."""
```

`librosa.load(path, sr=16000, mono=True)`. Accept `.ogg`, `.wav`, `.mp3`, `.m4a`, `.flac`.

**No loudness normalization anywhere before the signal branch.** Absolute level is a
load-bearing feature — the noise floor in dBFS is the primary severity signal (§7.3) and the
primary presence detector (§7.2). Normalizing gain destroys it. The tone branch needs
level-invariance and gets it by z-scoring against the speaker's own baseline *internally*
(§6.2), never by normalizing the shared audio.

Note the provided `.ogg` files are 48 kHz
2-channel with **byte-identical channels** (`corr = 1.0000`) — duplicated mono, not true
stereo. There is no free speaker separation. Handle genuinely-stereo input by checking
inter-channel correlation: if `< 0.98`, treat as dual-leg and skip diarization entirely.

### 4.2 VAD — `vad.py`

Silero VAD (`snakers4/silero-vad`, ~1 MB, CPU).

```python
def speech_segments(y: np.ndarray, sr: int = 16000) -> list[Segment]:
    """Segment = (start_s, end_s). Non-speech = complement."""
```

Config: `threshold=0.5`, `min_speech_duration_ms=250`, `min_silence_duration_ms=300`.

Outputs feed everything: speech regions for prosody, non-speech regions for noise estimation,
gaps for `long_silence_present`.

### 4.3 Speaker assignment — `diarize.py`

Full pyannote diarization is the expensive component on CPU (~0.5–1× realtime). We exploit a
domain fact instead: **there are exactly two speakers, and the agent is a known, consistent
TTS voice.**

```python
def assign_speakers(y, segments) -> dict[int, Literal["agent", "customer"]]:
```

1. ECAPA-TDNN embedding per VAD segment (`speechbrain/spkrec-ecapa-voxceleb`, ~20M params).
2. Agglomerative clustering into 2 clusters (cosine, average linkage).
3. Assign `agent` = cluster with highest mean cosine similarity to a stored reference bank of
   "Erica" embeddings, built from the three provided calls.
4. **Fallback** when max similarity < `AGENT_REF_MIN_SIM` (0.55): assign `agent` = cluster
   containing the first speech segment. The bot always greets first
   ("Hi, I'm Erica from…") — true in all three provided calls.
5. If clustering yields one effective cluster (silhouette < 0.1), mark
   `diarization_degraded=True`, treat all speech as customer, and cap final confidence.

This is cheaper and more robust than general diarization for the 2-speaker case, and degrades
predictably. Overlap detection still uses `pyannote/overlapped-speech-detection` (§6.5), which
is a much lighter model than the full diarization pipeline.

**Risk:** the hidden set may use a different TTS voice. The fallback in step 4 covers it, and
step 5 covers total failure.

### 4.4 ASR — `asr.py`

`faster-whisper` with `large-v3-turbo`, `compute_type="int8"` on CPU / `"float16"` on GPU.

```python
model.transcribe(
    path, word_timestamps=True, vad_filter=False,   # we already have Silero segments
    beam_size=5, language=None,                      # auto-detect; DO NOT hardcode "en"
)
```

Three requirements, each a gap observed in exploratory ASR runs (Whisper large-v3 /
large-v3-turbo / Parakeet TDT over the three calls; artifacts since deleted, findings recorded
here):

1. **Multilingual.** Parakeet TDT is English-only, which is why call_002's Spanish returned
   garbage. One of three provided calls is Spanish, so the hidden set will contain Spanish.
2. **`word_timestamps=True`.** The entire prosody-alignment design depends on them. Current
   output has none.
3. **Capture detected language.** Currently `null` in the output. It is a useful feature and
   needed for routing.

*Documented future optimization (not in v1):* cheap language ID → Parakeet TDT for English
(0.07× realtime measured) and Whisper only for non-English. Materially improves cost and
latency; deferred as complexity risk against the deadline.

---

## 5. Tone branch

### 5.1 Acoustic features — `prosody.py`, `ser.py`

**eGeMAPSv02** via the `opensmile` package, computed per customer utterance using Whisper word
timestamps as the alignment grid:

```python
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
```

Features used: loudness range (**not** mean — mean is recording gain), F0 mean and range,
speech rate, jitter, shimmer, HNR, spectral slope, pause structure.

**Dimensional SER**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, which outputs
continuous **arousal / dominance / valence** in [0,1]. This is a materially better choice than
categorical SER for this problem:

- Trained on **MSP-Podcast** — naturalistic speech, not acted studio recordings like RAVDESS.
- **Arousal** maps directly onto `emotional_intensity`.
- **Valence** gives tone polarity without forcing a 4-class label that doesn't match our schema.
- **Dominance** helps separate `upset` (high dominance) from `distressed` (low dominance) —
  the single hardest distinction in the schema.

Passed to the LLM as **the full distribution, never an argmax.** An out-of-domain model's
relative scores carry information; its hard label does not.

### 5.2 The LLM call — `tone_llm.py`

**Model:** `claude-haiku-4-5`, Anthropic API, structured outputs.

Chosen over Sonnet 5 for cost (§10) and for one rigor reason: Haiku 4.5 still accepts
`temperature=0`, which the 4.7+ generation removed. That buys near-deterministic
classification, which matters under "reproducibility".

Since the model never hears the audio, **the feature description *is* the tone signal** and
must be rich.

#### 5.2.1 System prompt

```
You are an expert call-audio analyst for AutoAce. You score calls between an AI
voice agent and a human caller. Classify the HUMAN CALLER's emotion only —
ignore the agent's tone entirely.

You do not receive audio. You receive acoustic measurements, a dimensional
speech-emotion model's output, a summary of what the agent did, and a transcript
annotated with per-utterance prosody. Treat the acoustic evidence as primary.

## Label definitions
<verbatim §2 definitions from the brief — do not paraphrase>

## Evidence weighting
- emotional_tone: SER valence and dominance, plus the prosodic annotation, are
  PRIMARY. Word choice is CORROBORATING. Never infer frustration, upset, or
  distress from loudness alone.
- emotional_intensity: prosodic activation and SER arousal are PRIMARY. It is
  independent of tone — a clearly positive caller can be high intensity, and a
  neutral caller can be medium.
- upset vs distressed: high dominance with high arousal indicates upset (angry,
  assertive). Low dominance with high arousal indicates distressed (overwhelmed,
  panicked).

## Domain rules
- Callers use politeness formulas ("yes, hi", "okay", "thank you") reflexively.
  These are weak evidence. Prosody outranks them in both directions.
- A caller who stays warm and thanks the agent is SATISFIED even if the agent
  failed to fulfil their request. Do not infer frustration from an unsuccessful
  call outcome alone.
- Conversely, escalating repetition — the same short utterance repeated with
  rising energy and shrinking gaps — indicates UPSET even when the words are
  neutral.
- Sarcasm and irony invert lexical polarity. If the words say the opposite of
  what the prosody indicates, follow the prosody.
- Under 3 seconds of caller speech is insufficient evidence. Report that through
  low self_confidence rather than guessing.

## Output
Return only the JSON object. When acoustic and lexical evidence conflict
strongly, set evidence_conflict true and lower self_confidence rather than
forcing a label.
```

#### 5.2.2 Few-shot examples

**Three synthetic examples**, covering the boundaries that matter: polite-but-blocked →
`satisfied`; escalating repetition with neutral words → `upset`; minimal speech → low
confidence.

**They are deliberately synthetic.** The three labelled calls are the entire validation set —
using them as few-shot examples would destroy the regression test, since the model would simply
reproduce cases it had been shown. The examples encode *rules* derived from inspecting the
labelled calls (which §12 already accounts for), not the answers to specific test items.

#### 5.2.3 Output contract

```json
{
  "emotional_tone": "<enum>",
  "emotional_intensity": "<enum>",
  "self_confidence": 0.0,
  "reasoning": "one or two sentences citing the specific evidence used",
  "lexical_intensity_markers": ["..."],
  "agent_failed": false,
  "evidence_conflict": false
}
```

`self_confidence` is deliberately **not** the schema's `confidence` field. It is one voter
input into §8, not the answer. Naming it separately prevents accidental passthrough.

#### 5.2.4 Coherence validation

Per the PADISO guidance on hallucinated confidence, the label is validated against the acoustic
evidence **deterministically** rather than by parsing the prose rationale:

```python
# tone polarity must not contradict SER valence
if tone in POSITIVE_TONES and ser_valence < VALENCE_POS_MIN:      flag_conflict()
if tone in NEGATIVE_TONES and ser_valence > VALENCE_NEG_MAX:      flag_conflict()
if intensity == "high" and ser_arousal < AROUSAL_HIGH_MIN:        flag_conflict()
```

A flag lowers confidence and marks the call for the dashboard review queue (§13). It never
silently overrides the label.

#### 5.2.5 Sampling

`temperature=0`. Haiku 4.5 still accepts sampling parameters — the 4.7+ generation removed
them — and determinism is a reproducibility requirement here: a consistent label is worth as
much as an accurate one when the grader re-runs the batch.

Two further API constraints on this model:

- **No `output_config.effort`** — the effort parameter errors on Haiku 4.5. Do not carry it
  over from newer-model examples.
- **No extended thinking.** Not needed for classification, and it would inflate the output
  token count the cost model depends on.

One call per audio file. No agent loop, no tools.

#### 5.2.6 Payload

```
USER:
Call duration: 30.9s | Customer speech: 6.2s (tier B) | Language: en
Customer baseline: established 0:04–0:09 (moderate volume, steady pitch, 3.1 syl/s)

SER (MSP-Podcast, dimensional): arousal 0.78 | dominance 0.71 | valence 0.24

Agent behavior: greeted; failed to respond to 5 consecutive customer utterances;
no task completed; no handoff

Annotated customer transcript:
[00:12–00:18] (baseline) "Yes, hi. Are you a real person?"
[00:21–00:22] (louder, rising pitch) "Hello?"
[00:24–00:25] (much louder, faster, shorter gap) "Hello?"
[00:26–00:27] (much louder, peak energy) "Hello?"

Trajectory: rising through 00:27; peak activation z=2.3
Interruptions by customer: 0 | Turn latency: decreasing
```

Output schema enforced via `output_config.format` (JSON schema, `additionalProperties: false`,
all fields `required`, enums as `enum` arrays). Returns `emotional_tone`,
`emotional_intensity`, `lexical_intensity_markers`, `agent_failed`, and a
`reasoning` string retained for the memo and for dashboard explainability.

**Privacy:** transcripts and derived features leave AutoAce infrastructure; **audio does not.**
Zero-retention is set on the account. Disclosed per §11 of the brief with model name, pricing,
retention policy, and egress boundary stated explicitly.

**Local fallback (required deliverable, §9 of the brief — "compare at least two materially
different approaches"):** a zero-shot NLI classifier (`facebook/bart-large-mnli`) scoring the
transcript against the five class definitions as entailment hypotheses. Runs on CPU, needs no
labels, costs nothing. Serves three purposes: satisfies the comparison requirement, acts as the
no-API fallback if AutoAce rejects transcript egress, and votes in the confidence ensemble.

---

## 6. Intensity — `fuse.py`

**Premise: intensity is activation, not tone-strength.** The labels prove this —
`neutral / medium` and `satisfied / medium` are incoherent under a tone-strength reading.
It is computed **independently of tone** and joined only at output.

**Intensity is never derived from classifier confidence.** That is a different quantity, and
conflating them is what produced constant `high` in earlier attempts.

### 6.1 Tiering by available customer speech

The customer barely speaks. Total customer speech duration selects the method:

| tier | customer speech | method | example |
|---|---|---|---|
| A | > 15s | self-baseline + trajectory | call_003 |
| B | 3–15s | activation vs corpus norms, no trajectory | call_001 (~6s) |
| C | < 3s | **insufficient evidence → `medium`, confidence ≤ 0.45** | call_002 (~1s) |

Tier C returns the correct label for call_002 — but by prior, not by measurement, and the
confidence must say so.

### 6.2 Activation score

Baseline = first ~20–25s of customer speech (tier A) or corpus norms (tier B). Every feature
is z-scored against it, because absolute level is recording gain, not emotion.

```
activation_z = w1·loudness_range_z + w2·f0_elevation_z + w3·f0_range_z
             + w4·rate_deviation_z + w5·jitter_shimmer_z
             + w6·interruption_rate_z + w7·turn_latency_z
             + w8·lexical_marker_count_z
             + w9·ser_arousal_z
```

Weights in `config.py`, initialised uniform within each channel (vocal / interactional /
lexical / SER) and documented as underived.

### 6.3 Trajectory

Fit activation across call thirds; record slope and peak-segment z. The brief's own definitions
require this — *"medium is clear and **sustained**"* vs *"high is strong, **escalated**"*.
Medium is elevated-but-stable; high requires escalation. Averaging destroys the distinction.

### 6.4 Mapping

```python
if activation_z > 1.0 and (slope_rising or peak_z > 2.0):   return "high"
if activation_z < 0.3 and not slope_rising:                 return "low"
return "medium"
```

### 6.5 Two-voter reconciliation

Deterministic rule (above) and the LLM's own intensity judgment from the annotated transcript.

- **Agree** → emit, full confidence contribution.
- **Disagree** → emit **`medium`**, drop confidence.

Not hedging: `medium` is the majority class and the 2/3 constant-guess baseline, so the
conservative fallback is also the statistically optimal one.

---

## 7. Signal branch

Runs on **raw full-mix audio**. Never denoised, never enhanced.

### 7.1 `background_noise_type` — `tagging.py`

**AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) over non-speech frames. Chosen over
YAMNet because it is PyTorch/transformers-native — the stack is already torch, and YAMNet
drags in TensorFlow — and scores higher on AudioSet (mAP 0.459 vs ~0.31).

AudioSet's 527 classes contain most of the brief's vocabulary: speech babble, music,
vehicle/road noise, television, typing, wind, mechanical hum.

Procedure: tag in 1s windows with 0.5s hop **over non-speech regions only** → suppress speech
classes → aggregate by mean logit → top class → map through `AUDIOSET_TO_LABEL` to short
informal strings. The labeller writes `"TV"`, not `"television"`; keep outputs short.

#### 7.1.1 Measured performance

From an AST-vs-PANNs bake-off over the three calls (artifact since deleted; method was 10s
whole-clip windows → mean sigmoid over 527 classes → speech classes suppressed → dominant
non-speech group). **Reproduce this as `tests/` output during implementation** — the numbers
below are load-bearing for two design decisions and currently exist only in this document.

| call | truth | AST dominant | AST p | PANNs dominant | PANNs p |
|---|---|---|---|---|---|
| 001 | *(none)* | music | 0.010 | music | 0.050 |
| 002 | **TV** | **TV** ✓ (top class `Television`) | 0.056 | radio ✗ | 0.144 |
| 003 | **sharp static** | keyboard typing ✗ | 0.028 | radio ✗ | 0.090 |

**AST is confirmed over PANNs** — it named `Television` exactly on the one call with an
environmental noise source, where PANNs said `Radio`. This validates §7.1's model choice
empirically rather than on mAP alone.

Two findings that change the design:

**(a) Absolute probabilities are near the floor — the §7.2 threshold was wrong by an order of
magnitude.** Every value above is between 0.001 and 0.144, against a specified gate of 0.35.
No call would have passed; every clip would have been classified as noise-free. Two causes:
these are telephony-band recordings and AudioSet models are trained on wideband general audio,
and the comparison run tagged **whole-clip 10s windows** rather than non-speech regions, so
speech masked the background throughout.

*Action:* rerun on Silero non-speech regions (the specified method) before fixing any number,
then gate on **relative dominance** — top group vs median group — rather than absolute
probability, since absolute values are uncalibrated in this domain.

**(b) Line artifacts are effectively invisible to AudioSet.** The `static` group scored
0.0013–0.0059 across all three calls — the *lowest* of any group, including on the call whose
ground truth is "sharp static". Rerunning on non-speech regions will not fix this. AudioSet is
trained on environmental audio from video; "sharp static" in telephony is a codec or line
artifact, a different physical phenomenon that simply is not well represented in the label
space.

> **REFUTED 2026-08-17 — both actions below were implemented and measured, and both failed.**
> See §7.1.2. The spectral classifier does not work, and restricting the tagger to non-speech
> regions made typing *worse*. The shipped module does whole-clip tagging with the spectral
> classifier disabled. The text below is retained because the reasoning is what the measurement
> tested.

*Action (attempted):* add a deterministic spectral classifier for transmission artifacts, and
prefer it when the top environmental group is weak:

| signature | measurement | label |
|---|---|---|
| static / hiss | high spectral flatness (>0.4) + flat band energy across non-speech | `"static"` |
| hum | narrowband peak at 50/60 Hz with harmonics | `"electrical hum"` |
| crackle / clicks | short high-energy broadband transients, <20ms, recurring | `"line crackle"` |
| codec artifact | spectral discontinuity at frame boundaries | `"digital distortion"` |

Selection: if `ast_relative_dominance < TYPE_AST_MIN_DOM` **and** a spectral signature fires,
emit the spectral label. Otherwise emit the AST group. This splits the field along the physical
boundary that matters — AudioSet owns *environmental* noise, DSP owns *transmission* noise —
rather than asking one model to cover both.

#### 7.1.2 What measurement actually established (2026-08-17)

Both §7.1.1 actions were built and tested. Both failed, and the module now reflects what holds.

**Refutation A — "tag non-speech regions only" is wrong for telephony.** The argument was that
speech masks background noise. Measured, it inverts:

| call | truth | whole-clip AST | non-speech-only AST |
|---|---|---|---|
| call_001 | *(none)* | music (reldom 3.7) | music |
| call_002 | **TV** | **TV** ✅ | music ✗ |
| call_003 | sharp static | radio (reldom 8.6) | keyboard typing |

In telephony the non-speech gaps are near-**silent** — which is exactly *why* the noise floor
works as a presence detector — while the television is audible *underneath* the speech.
Removing the speech removed the evidence. **Shipped: whole-clip tagging, speech classes
suppressed afterwards.**

**Refutation B — no spectral feature discriminates the transmission noise.** AudioSet genuinely
is blind to it (`static` group scores 0.0013–0.0059 on every call, including the static one).
But none of the proposed replacements works either, measured over the non-speech regions:

| call | truth | flatness | transients/s | crest | kurtosis |
|---|---|---|---|---|---|
| call_001 | *(none)* | 0.0531 | **0.47** | **36.5** | **542** |
| call_002 | TV | 0.1157 | 0.28 | 12.9 | 81 |
| call_003 | **sharp static** | **0.0426** | 0.29 | 21.0 | 109 |

Flatness is *lowest* on the static call, where broadband hiss should be highest. Transient rate,
crest factor and kurtosis are all *highest* on the clean call. A 6 dB mains-hum test fired on
all three. **Shipped: the detector is disabled**, kept as a documented seam rather than emitting
an arbitrary label. `background_noise_type` for a line-artifact call therefore gets the nearest
environmental label — a known, recorded inaccuracy, pinned as `xfail(strict=True)`.

**One useful signal did emerge.** Relative dominance (top group ÷ median group) separates clean
from noisy where absolute probability cannot: **3.7** on the clean call against **47.3** and
**8.6** on the two noisy ones, while all absolute probabilities sit between 0.006 and 0.023.
That makes it a usable secondary presence signal alongside the floor — note `TAG_MIN_DOM` is
currently 1.8, which all three clear, so it needs raising to ~5 to discriminate.

**Also corrected:** AST windowing was specified at 1.0s/0.5s. AST's native input is 10.24s, so
every short window was zero-padded to that length — ~97 padded forward passes per call instead
of ~10, on inputs outside the training distribution. Now 10.0s/5.0s.

### 7.2 `background_noise_present` — two-condition gate

```python
present = (noise_floor_dbfs > NOISE_FLOOR_PRESENT          # −55.0  — primary
           and (ast_relative_dominance > TAG_MIN_DOM       # top group vs median group
                or spectral_signature_fired))              # §7.1.1(b)
```

**The noise floor is the presence detector; the tagger only names the type.** §7.1.1 showed
absolute AudioSet probabilities are uncalibrated on telephony audio and do not separate the
noise-free call from the noisy ones (call_001 `music` 0.010 vs call_003 `keyboard` 0.028 —
no usable margin). The noise floor does separate them cleanly: −56.3 for the one negative,
−52.1 and −47.0 for the two positives.

Inverting the original design this way also satisfies the brief's warning that "barely
perceptible artifacts should not automatically count" — floor level is a direct measure of
perceptibility, whereas tagger confidence is a measure of a model's familiarity with the sound.

### 7.3 `background_noise_severity` — `quality.py`

Floor level primary, SNR secondary. Noise floor estimated by **minimum statistics / MCRA**
per-band over true (Silero) non-speech frames — not percentile-thresholded energy, which §2.5
showed is unreliable.

```python
NOISE_SEVERITY_BANDS = {   # noise_floor_dbfs upper bound → severity
    -55.0: "none",
    -50.0: "low",
    -44.0: "medium",
    999.0: "high",
}
```

Anchored at `none` (−56.3) and `medium` (−52.1, −47.0). **The low/medium and medium/high
boundaries are interpolated, not fitted** — there are no `low` or `high` examples. Stated in
the memo.

### 7.4 `audio_quality`

Default `clear`; requires positive evidence to move off it. All three provided calls are
`clear`, and per §2.6 static goes to noise, not quality.

Learned signal: **TorchAudio-SQUIM** (`torchaudio.pipelines.SQUIM_OBJECTIVE`), which estimates
STOI / PESQ / SI-SDR non-intrusively from a single waveform. Chosen over NISQA/DNSMOS because
it ships inside torchaudio — no vendored repo, no ONNX runtime, no extra dependency.

Deterministic detectors alongside it:

| defect | measurement |
|---|---|
| clipping | fraction of samples \|x\| > 0.98 (measured 0.00% on all three — will rarely fire) |
| dropouts | zero-runs > 30ms + spectral discontinuity |
| echo | autocorrelation peak at 20–200ms lag |
| muffling | spectral tilt **baselined against telephony**, not wideband (§2.5, finding 3) |
| low volume | LUFS below −35 |

Mapping: `clear` unless estimated STOI < 0.75 or any deterministic detector fires above its
threshold; `severely_impaired` when two or more fire or STOI < 0.55.

### 7.5 `speaker_overlap_present`

`pyannote/overlapped-speech-detection`, gated: overlap segments > 0.5s, total overlap > 1.0s.
Backchannel "mm-hm" must not fire it.

**Known trap:** call_002 has TV in the background *and* `overlap: true`. Television audio
contains speech, so the detector will fire on background TV regardless of whether the labeller
counted it. With three calls we cannot tell which. **Decision: do not suppress.** Let TV speech
count, matching the observed label. Recorded in the memo as an unresolved ambiguity and a
first-priority question for AutoAce.

### 7.6 `long_silence_present`

Requires energy **below the noise floor** (true dead air), not merely non-speech, for
`> LONG_SILENCE_SEC` (10.0s).

Measured bound: call_003 has a 7.35s non-speech gap and is labelled `false`. A naive
non-speech-gap detector at any threshold ≤ 7s produces a false positive on a provided call.

---

## 8. Confidence

Built from **voter agreement**, never from model self-report. The brief asks for calibration;
a model asked to rate its own certainty returns a number clustered at 0.8 that correlates with
nothing.

```python
confidence = clip(
    BASE                                        # 0.55
    + 0.15 * tone_voters_agree                  # Haiku vs BART-MNLI
    + 0.10 * intensity_voters_agree             # rule vs LLM
    + 0.10 * (ser_valence_consistent_with_tone)   # §5.2.4 coherence check
    + 0.05 * (llm_self_confidence > 0.8)          # demoted to one voter, not the answer
    - 0.15 * llm_evidence_conflict                # model flagged conflicting evidence
    - 0.05 * any_feature_near_threshold           # within 10% of any §9 threshold boundary
    - 0.20 * (tier == "C")
    - 0.15 * diarization_degraded
    - 0.10 * (asr_avg_logprob < ASR_MIN_LOGPROB),
    0.05, 0.98)

# Hard caps applied after the formula, since a tier-C result is a prior
# rather than a measurement no matter how well the other voters agree:
if tier == "C":            confidence = min(confidence, TIER_C_MAX_CONF)   # 0.45
if diarization_degraded:   confidence = min(confidence, DEGRADED_MAX_CONF) # 0.50
```

Every term is inspectable and appears in the dashboard's per-call detail view.

The `any_feature_near_threshold` term matters more here than it normally would. Because none of
the §9 thresholds are fitted (§2.2), a value sitting just either side of a boundary is a coin
flip we have no evidence to resolve. Surfacing that as reduced confidence — and into the
review queue (§13) — is the honest response, and it concentrates human attention exactly where
the calibration is weakest.

---

## 9. Configuration and repo layout

**Every threshold in this document lives in `config.py` and nowhere else.** This is the core
rigor claim: thresholds are derived, not fitted, and must be auditable in one place.

```
autoace/
  config.py        # ALL thresholds, weights, model IDs
  schema.py        # pydantic models; enum enforcement
  io_audio.py      # decode, resample, stereo detection
  vad.py           # Silero
  diarize.py       # ECAPA + clustering + agent reference bank
  asr.py           # faster-whisper, word timestamps, language ID
  prosody.py       # eGeMAPS, activation score, trajectory
  ser.py           # audeering dimensional A/V/D
  tagging.py       # AST AudioSet + label mapping
  quality.py       # SQUIM, noise floor/MCRA, DSP detectors
  tone_llm.py      # Haiku structured output
  tone_nli.py      # BART-MNLI local fallback / second approach
  fuse.py          # intensity rule, reconciliation, confidence, assembly
  pipeline.py      # orchestrator; per-file, fail-isolated
  eval.py          # harness against labels.csv
  app.py           # Gradio dashboard
tests/
  test_regression.py   # the 3 provided calls as a regression suite
docs/
  MEMO.md              # technical memo deliverable
```

---

## 10. Cost model

Per **audio minute**. Two paths reported; GPU is the production recommendation.

**LLM (both paths).** Per audio minute the model sees ≈ **830 input tokens** (system prompt
with label definitions and few-shots, annotated transcript, SER distribution, agent-behavior
summary) and emits ≈ **100 output tokens**.

| model | $/MTok in | $/MTok out | $/audio-min | verdict |
|---|---|---|---|---|
| **Haiku 4.5** | $1 | $5 | **$0.00133** | fits, with room for compute |
| Sonnet 5 | $3 | $15 | $0.00399 | **exceeds the entire ceiling on the LLM call alone** |
| Opus 5 | $5 | $25 | $0.00665 | 2.2× the ceiling |

**The model choice is forced by arithmetic, not preference.** Sonnet 5 and Opus 5 each blow the
$0.003 budget before a single second of ASR, diarization, or DSP is paid for. Sonnet 5 is
costed at its standard $3/$15 rather than the $2/$10 introductory rate, which expires
2026-08-31 — a production cost model should not depend on promotional pricing.

*The one upgrade lever the budget permits:* the Batch API halves LLM cost, which brings Sonnet 5
to ≈ $0.0020/min and Haiku to ≈ $0.00067/min. If batch latency is acceptable for the evaluation
workflow, a Sonnet-5-batched A/B against Haiku on the eval harness is the highest-value accuracy
experiment available. That is a decision to make against measured accuracy, not up front.

Prompt caching does **not** apply: Haiku 4.5's minimum cacheable prefix is 4096 tokens and our
system prompt is well under. Not counted in the model.

*Available lever, deliberately not taken in v1:* batching N calls per request amortizes the
~500-token system prompt across the batch, leaving only the ~330-token per-call payload. At
N=10 input drops 830 → 380 tokens/call (−54%), taking the LLM line from $0.00133 to ≈$0.00088;
at N≈20 the prefix also crosses the 4096-token caching minimum, compounding the saving.
**Rejected for v1** on two grounds: calls sharing one context can anchor on each other's
judgments, which is an accuracy risk on a graded evaluation, and it breaks the per-file
fail-isolation the dashboard requires (§13). Held in reserve if the ceiling tightens — see the
§10.1 sensitivity table for where that becomes necessary.

**Compute — CPU path** (4 vCPU @ ~$0.05/hr = $1.39e-5 per box-second):

| stage | s / audio-min |
|---|---|
| decode + VAD | 0.2 |
| ECAPA + clustering | 2 |
| faster-whisper large-v3-turbo int8 | 60 |
| eGeMAPS | 0.5 |
| audeering SER | 3 |
| AST tagging | 1 |
| SQUIM + DSP | 1.5 |
| overlap detection | 8 |
| **total** | **~76s** → **$0.00106** |

**Compute — GPU path** (T4 spot @ ~$0.20/hr = $5.6e-5 per box-second): ~11s → **$0.00062**

### 10.1 Sensitivity — and a counterintuitive result

Total = compute + $0.00133 LLM. Compute cost is instance-price-dependent, and **the CPU path's
viability flips inside the realistic price range**:

> **Revised 2026-08-16 against measurement.** The original model assumed faster-whisper ran at
> ~1.0× realtime on CPU, taken from an exploratory `openai-whisper` run. The CTranslate2 build
> at `int8` measures **0.41× realtime** over the three calls (22.3s / 18.9s / 56.1s of
> wall-clock for 30.9s / 35.0s / 171.9s of audio = 97.3s for 237.8s). That is a 2.4× error in
> the dominant CPU stage, and it changes the conclusion below.

Revised CPU stage total: **~40.7 s per audio-minute** (ASR 24.5 measured, down from 60
assumed; all other stages unchanged).

| path | instance | $/hr | s/audio-min | compute | **total** | vs ceiling |
|---|---|---|---|---|---|---|
| CPU | 4 vCPU spot | $0.05 | 40.7 | $0.00057 | **$0.00190** | 37% headroom |
| CPU | 4 vCPU on-demand | $0.10 | 40.7 | $0.00113 | **$0.00246** | 18% headroom |
| GPU | T4 spot | $0.20 | 11 | $0.00061 | **$0.00194** | 35% headroom |
| GPU | T4 on-demand | $0.35 | 11 | $0.00107 | **$0.00240** | 20% headroom |
| GPU | L4 | $0.50 | 11 | $0.00153 | **$0.00286** | 5% headroom |

**The earlier "GPU is both faster and cheaper" claim does not survive measurement.** At spot
pricing CPU ($0.00190) and GPU ($0.00194) are within noise of each other, and CPU at
*on-demand* pricing now fits comfortably where it previously breached by 15%. GPU remains
~4× faster in wall-clock, so it is still the right choice for batch throughput — but that is
now a **latency** argument, not a cost one.

**Revised CPU break-even: $0.148/hr** (was $0.079). Every realistic CPU instance price sits
below that, so the CPU path is no longer contingent on spot pricing.

**Recommended deployment: GPU for batch throughput, CPU entirely viable for cost.** The dev box
is CPU-only (torch 2.13.0+cpu, no CUDA) and the CPU path must stay runnable for reproducibility
regardless.

*Lesson for the memo:* the single largest term in the cost model was wrong by 2.4× because it
was inherited from a different ASR implementation rather than measured on the one actually
shipped. Every dominant term should be measured before the model is reported.

### 10.2 Stated exclusions

The brief asks for inference cost per audio minute. These are excluded rather than silently
omitted: one-time model download and warm-up, dashboard hosting (fixed, not per-minute), and
storage. Other assumptions: batch utilization ≥ 70%, model weights resident across calls, no
retry overhead.

---

## 11. Latency model

Measured on the dev box (CPU-only torch 2.13.0, no CUDA):

- Whisper large-v3 measured at **33–46s on a 31s file** ≈ 1–1.5× realtime — the dominant cost.
- Parakeet TDT measured at **2.2s on the same file** ≈ 0.07× realtime (English only).

| stage | CPU (× realtime) | GPU (× realtime) |
|---|---|---|
| decode + Silero VAD | 0.02 | 0.02 |
| ECAPA + clustering | 0.03 | 0.01 |
| **faster-whisper large-v3-turbo** | **0.41** *(measured 2026-08-16, int8)* | 0.05 |
| eGeMAPS | 0.01 | 0.01 |
| audeering SER | 0.05 | 0.01 |
| AST tagging | 0.02 | 0.01 |
| SQUIM + DSP detectors | 0.03 | 0.01 |
| overlap detection | 0.13 | 0.02 |
| Claude call | 1–3s fixed | 1–3s fixed |
| **end-to-end, 3-min call** | **~4 min** | **~35s** |

**Only the ASR row is measured.** Every other figure is an estimate and is labelled as such in
the memo. ASR still dominates the CPU path at roughly 60% of total wall-clock (down from ~80%
under the pre-measurement assumption), which is why GPU remains the throughput recommendation
even though §10.1 no longer supports it on cost.

Measured per call (CPU, `int8`, model resident): call_001 22.3s / 30.9s audio, call_002 18.9s /
35.0s, call_003 56.1s / 171.9s. Longer files are proportionally faster — the fixed model-load
and warm-up cost amortises, so 0.41× is a blended figure and the marginal rate on long calls is
closer to 0.33×.

CPU path: ~76s per audio minute. GPU path: ~11s per audio minute. Batch of 50 three-minute
calls: ~3.2 hours CPU / ~28 minutes GPU, with a worker pool of 4 giving near-linear speedup
since stages are per-file independent.

Reported as measured wall-clock in the memo, with the GPU figure marked as projected unless a
GPU box is provisioned before submission.

---

## 12. Evaluation harness — `eval.py`

Reads the brief's manifest format directly: `name,result_json` where `name` is the exact audio
filename and `result_json` is the expected JSON object.

Metrics are reported **by group**, matching how the brief says it scores:

| group | fields | metrics |
|---|---|---|
| Tone | `emotional_tone` | accuracy, macro-F1, 5×5 confusion matrix |
| Intensity | `emotional_intensity` | accuracy, macro-F1, 3×3 confusion, **plus the constant-`medium` baseline** |
| Noise | `_present`, `_type`, `_severity` | boolean accuracy; severity macro-F1; type by normalized string match with a synonym table |
| Technical | `audio_quality`, `speaker_overlap_present`, `long_silence_present` | per-field accuracy |

**Always report intensity against the constant-`medium` baseline**, which scores 2/3 on the
provided data. A metric that does not beat the majority-class baseline is not evidence of
anything, and reporting it unprompted is worth more under technical rigor than a flattering
accuracy number.

**Reported honestly.** With n=3 and five of eight fields having incomplete class coverage
(§2.2), a confusion matrix is a formality. The memo states: *these three calls are a regression test, not
a validation set; the reported figures establish that the system reproduces known-good
behaviour and nothing more.* Claiming "100% on 3/3" without that caveat is the single easiest
way to lose the 15% rigor allocation.

**Validation protocol:** leave-one-call-out, grouped by call so no segment from a call appears
on both sides. **No model is trained** — the LLM is prompted, the thresholds are derived from
definitions and physics — so leakage risk is confined to *prompt engineering against the three
examples*. That risk is real and is mitigated two ways: the three calls stay a regression test
rather than a tuning set, and the thresholds are written from the §2.5 measurements rather than
found by search.

The agent reference bank (§4.3) is the one component built *from* the provided calls. It is
speaker-identity only, carries no label information, and cannot leak tone or noise labels.

The few-shot examples (§5.2.2) are the one place where knowledge derived from inspecting the
labelled calls enters the prompt. They encode *rules*, not answers, and are synthetic — but the
three-call figures are therefore not an unbiased accuracy estimate, and the memo says so.

**Recommended addition if time permits:** self-label 20–30 clips from a public call-centre-like
corpus to give the low/high severity and `low` intensity classes at least one anchor each.

---

## 13. Dashboard — `app.py`

Gradio `Blocks` with built-in auth (`gr.Blocks(auth=...)`), satisfying the login requirement
with minimal code.

Required behaviours from §7 of the brief:

- **Upload** folder or ZIP: audio at root + one CSV manifest.
- **Validate**: report unmatched files (audio without a manifest row and vice versa) *before*
  processing; do not silently skip.
- **Process** with a visible progress bar (`gr.Progress()`), per-file status.
- **Fail-isolate**: a malformed file marks that row `error` with the reason and the batch
  continues. This is an explicit requirement and a common failure point.
- **Review**: per-file table of all 9 schema fields, plus a detail view exposing the confidence
  terms and the LLM `reasoning` string.
- **Review queue**: rows with `confidence < REVIEW_THRESHOLD` (0.60) or `evidence_conflict`
  set are visually flagged and sortable to the top. This makes the tier-C and
  conflicting-evidence cases (§6.1, §5.2.4) visible to a human rather than buried in a CSV,
  and demonstrates the production pattern of routing low-confidence predictions to review
  rather than trusting them silently.
- **Download**: CSV and JSON preserving original filenames.
- **Scoring view when labels are present**: if the uploaded manifest's `result_json` column is
  populated, additionally render the §12 metrics — per-group accuracy, macro-F1, confusion
  matrices, and the constant-`medium` intensity baseline. The evaluator can then score a
  labelled batch without touching the command line, which is the point of the deliverable.

**Deployment:** a VM AutoAce-approved or under our control (Fly.io / Render with private
storage). **Not** a public model-hosting Space — uploaded production audio is confidential per
§5 of the brief, and the deployment must stay available through the evaluation period.

---

## 14. Failure modes and limitations

Stated in the memo, not discovered by the grader:

1. **Threshold calibration is definitional, not empirical.** Five of eight fields lack full class
   coverage. Highest risk: the `low`/`medium` intensity boundary and the `low`/`high` noise
   severity boundaries, which have zero anchors.
2. **Tier C is a prior, not a measurement.** Calls where the customer says almost nothing get
   `medium` intensity by default. Correct on call_002; unfalsifiable on three calls.
3. **Agent reference bank may not transfer, and mis-assignment fails silently.** A different TTS
   voice on the hidden set falls back to first-speaker heuristics (§4.3 step 4), then to degraded
   mode (step 5). The failure is quiet and expensive: swapping the labels substitutes the bot's
   uniformly calm TTS delivery for the customer's voice, which would drag tone toward `neutral`
   and intensity toward `low` on every affected call. Cross-check the assignment against the
   LLM's own read of who is speaking, and route disagreement into `confidence`.
4. **TV-speech overlap ambiguity** (§7.5) is unresolved and could go either way on the hidden set.
4b. **Line-artifact typing is validated on one example.** §7.1.1(b) established that AudioSet
   cannot detect "sharp static" and added a deterministic spectral classifier for transmission
   noise. That classifier has exactly one ground-truth instance to check against, and its
   `hum` / `crackle` / `codec` branches have none. Environmental noise typing is on firmer
   ground (AST named `Television` correctly), but the transmission-noise half of the field is
   effectively unvalidated.
5. **ASR errors propagate**, and noisy calls are disproportionately the emotional ones. Mitigated
   by feeding `avg_logprob` into confidence, not by pretending it doesn't happen.
6. **The bot-outcome→emotion hypothesis** (§2.4) holds 3/3 and is otherwise unvalidated.
6b. **CONFIRMED DEFECT — code-switching breaks speaker assignment.** Not a risk; measured.
   On call_002 the bot greets in English, the customer says two words, and the bot then
   continues in Spanish. ECAPA embeddings are language-sensitive, so clustering separates the
   bot's **English voice from its own Spanish voice** rather than bot from human:
   **12.36s attributed to the customer against a true ~0.9s.** The `degraded` flag does not
   fire, because the two clusters *are* well separated — along the wrong axis.

   Consequence: call_002 should be **Tier C** (<3s → emit `medium`, cap confidence), which is
   how it reaches its correct `neutral / medium` label. Instead it takes Tier B and reasons
   over ~11s of bot audio it believes is the customer.

   Proven unfixable at the cluster level: the customer's single 0.9s segment never forms its
   own cluster at k ∈ (2,3), so the floor for `customer_speech_seconds` is 5.2s whatever
   reference vector is used. No cluster clears `AGENT_REF_MIN_SIM` (best 0.4427), and at k=3
   the mixed cluster outranks the pure-agent one — the similarity metric is not ordering
   correctly, not merely mis-thresholded.

   **Planned fix — now proven viable (2026-08-16).** Per-segment classification informed by
   ASR language ID, so segments in different languages can still map to one speaker. Two
   measurements establish the path:

   - `asr.detect_language()` over call_002's opening 8s returns `en`; over its final 12s
     returns `es`. Pinned by `test_detect_language_works_on_an_audio_slice`. The signal the
     fix needs demonstrably exists at slice granularity.
   - It must come from that function, **not** from `transcribe()`. faster-whisper's `Segment`
     dataclass carries no `language` field (verified against 1.2.1), and clip-level detection
     reports call_002 as **`en`** despite its Spanish second half, because detection samples
     only the opening window. `AsrSegment` therefore deliberately has no `language` field —
     an earlier draft populated one from the clip-level value, which would have looked
     per-segment while carrying no per-segment information and made the fix silently no-op.
   Tracked by `test_code_switched_call_does_not_inflate_customer_speech`, marked
   `xfail(strict=True)` so it flips to XPASS the moment it genuinely passes.

   **Also observed:** on call_001 a reference similarity of 0.5529 sits 0.003 above the 0.55
   threshold, so speaker assignment there is decided by noise. `AGENT_REF_MIN_SIM` needs
   re-deriving once a working method exists.

7. **Multilingual prosody.** eGeMAPS baselines and especially speech-rate features are
   language-sensitive. Z-scoring each caller against themselves absorbs much of this, but the
   SER model and the LLM's interpretation of prosody tags are both English-centric, and
   Whisper's transcription quality varies by language. call_002 is Spanish, so this is a live
   issue on the provided set, not a hypothetical.
8. **No audio reaches the LLM.** A deliberate privacy choice (§5.2) that costs some accuracy
   against a multimodal audio model. The size of that gap is unmeasured.
8. **No cross-validation is possible.** Leave-one-call-out on n=3 is not meaningful.

**Next steps, in priority order:**

1. **Collect 100–300 labelled calls**, weighted toward the absent classes — `low` intensity,
   both impaired quality levels, `low`/`high` noise severity, `long_silence: true`. Everything
   below is gated on this.
2. **Fit rather than reason** the §6.2 activation weights and the §7.3 severity bands, and run a
   genuine stratified evaluation with per-class figures.
3. **A/B Sonnet 5 via the Batch API** against Haiku on the harness (§10) — the one accuracy
   upgrade the cost ceiling permits.
4. **Evaluate an audio-LLM tiebreak** on the subset where the two intensity voters disagree
   (§6.5) — a bounded fraction of calls, so bounded cost and bounded audio egress. Requires
   AutoAce to approve audio leaving their infrastructure; the local-only design stands if they
   do not.

---

## 15. Build order

| # | Deliverable | Blocking? |
|---|---|---|
| 1 | `schema.py`, `config.py`, `eval.py`, regression test on 3 calls. **Pins §2.5 acoustics as an assertion** | yes — everything measures against this |
| 2 | Shared front end (decode, VAD, diarize, ASR + word timestamps). **Regenerates §2.3, §2.4, §11 evidence** | yes — both branches depend on it |
| 3 | Signal branch (six classified fields, all deterministic/local). **Regenerates §7.1.1 bake-off** | no — highest points per hour, independent |
| 4 | Tone branch: prosody + SER + Haiku | no |
| 5 | Intensity rule + reconciliation + confidence | depends on 4 |
| 6 | BART-MNLI second approach (comparison requirement) | no |
| 7 | Gradio dashboard + deployment | no — but 10% of score, do not leave to the last day |
| 8 | Memo, cost/latency analysis, failure modes | no |

Rationale for putting the signal branch before the tone branch: **six of the eight classified
fields**, fully deterministic, no API dependency, near-zero marginal cost. It is the
reliable-points block.
Tone is the hard, uncertain one — and it is bounded by three labels no matter how much time
goes into it.
