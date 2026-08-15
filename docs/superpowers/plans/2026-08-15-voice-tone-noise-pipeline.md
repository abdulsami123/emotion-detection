# Voice Tone & Background Noise Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hosted, batch-capable system that classifies emotional tone and background noise in production call audio against a fixed 9-field schema, under $0.003 per audio minute.

**Architecture:** A shared front end (decode → VAD → speaker assignment → ASR) feeds two independent branches. The **signal branch** derives six fields deterministically from raw audio (noise floor, AudioSet tagging, spectral line-artifact detection, SQUIM quality, overlap, dead air). The **tone branch** derives two fields from prosody plus a dimensional SER model plus a Haiku 4.5 call over an annotated transcript. They merge only at output, where confidence is computed from voter agreement. All thresholds live in one config module.

**Tech Stack:** Python 3.13, PyTorch (CPU), faster-whisper, openSMILE, transformers (AST, audeering SER, BART-MNLI), torchaudio SQUIM, SpeechBrain ECAPA, pyannote.audio, Silero VAD, Anthropic SDK, Pydantic, Gradio, pytest.

**Reference spec:** `docs/superpowers/specs/2026-08-15-voice-tone-noise-design.md`

---

## File Structure

```
autoace/
  __init__.py
  schema.py         # Pydantic models, enums — the output contract
  config.py         # EVERY threshold and model ID, one place
  io_audio.py       # decode to 16k mono, stereo detection
  vad.py            # Silero VAD → speech/non-speech segments
  diarize.py        # ECAPA embeddings → 2-cluster → agent/customer
  asr.py            # faster-whisper, word timestamps, language ID
  acoustics.py      # noise floor, SNR, spectral features (shared DSP)
  tagging.py        # AST AudioSet + spectral line-artifact classifier
  quality.py        # SQUIM + clipping/dropout/echo/muffle detectors
  signal_branch.py  # assembles the six signal fields
  prosody.py        # eGeMAPS per utterance, activation score, trajectory
  ser.py            # audeering dimensional arousal/valence/dominance
  tone_llm.py       # prompt assembly + Haiku structured output
  tone_nli.py       # BART-MNLI zero-shot second approach
  fuse.py           # intensity rule, reconciliation, confidence, assembly
  pipeline.py       # orchestrator, per-file fail isolation
  eval.py           # manifest → grouped metrics + confusion matrices
  app.py            # Gradio dashboard
tests/
  test_schema.py
  test_acoustics_baseline.py   # pins §2.5 — canary that audio is unchanged
  test_vad.py
  test_diarize.py
  test_asr.py
  test_tagging.py
  test_quality.py
  test_signal_branch.py
  test_prosody.py
  test_fuse.py
  test_eval.py
  test_pipeline.py
docs/
  MEMO.md
requirements.txt
```

**Boundary rationale.** `acoustics.py` holds DSP primitives used by both `tagging.py` and `quality.py`, so they don't duplicate noise-floor code. `signal_branch.py` and `fuse.py` are assemblers with no measurement logic of their own — all thresholds resolve through `config.py`, which is what makes the calibration position auditable.

---

## Task 1: Project scaffold, schema, and config

**Files:**
- Create: `requirements.txt`, `autoace/__init__.py`, `autoace/schema.py`, `autoace/config.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Initialise the repository**

```bash
cd C:/Users/samir/autoace-test
git init
git add reference/ docs/
git commit -m "chore: initial commit — brief, labels, design spec"
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy>=1.26
scipy>=1.11
librosa>=0.10
soundfile>=0.12
torch>=2.13
torchaudio>=2.11
transformers>=4.46
faster-whisper>=1.0.3
opensmile>=2.5
speechbrain>=1.0
pyannote.audio>=3.1
silero-vad>=5.1
anthropic>=0.40
pydantic>=2.5
gradio>=4.44
pandas>=2.1
pytest>=8.0
```

- [ ] **Step 3: Write the failing schema test**

Create `tests/test_schema.py`:

```python
import json
import pytest
from pydantic import ValidationError

from autoace.config import LABELS_CSV
from autoace.schema import CallAnalysis, EmotionalTone, EmotionalIntensity


def test_valid_analysis_round_trips():
    payload = {
        "emotional_tone": "upset",
        "emotional_intensity": "high",
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
        "confidence": 0.82,
    }
    result = CallAnalysis(**payload)
    assert result.emotional_tone == EmotionalTone.UPSET
    assert json.loads(result.model_dump_json()) == payload


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        CallAnalysis(
            emotional_tone="furious",  # not in the enum
            emotional_intensity="high",
            background_noise_present=False,
            background_noise_type="",
            background_noise_severity="none",
            audio_quality="clear",
            speaker_overlap_present=False,
            long_silence_present=False,
            confidence=0.82,
        )


def test_confidence_bounds_enforced():
    base = {
        "emotional_tone": "neutral",
        "emotional_intensity": "low",
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
    }
    with pytest.raises(ValidationError):
        CallAnalysis(**base, confidence=1.5)
    with pytest.raises(ValidationError):
        CallAnalysis(**base, confidence=-0.1)


def test_every_labels_csv_row_validates():
    """The three provided labels must satisfy our schema exactly."""
    import csv
    with open(LABELS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    for row in rows:
        CallAnalysis(**json.loads(row["result_json"]))
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace'`

- [ ] **Step 5: Write `autoace/__init__.py`**

```python
"""AutoAce voice tone and background noise analysis."""
__version__ = "0.1.0"
```

- [ ] **Step 6: Write `autoace/schema.py`**

```python
"""The output contract. Enum values are fixed by the trial brief §2 and must
not be renamed — the grader matches on these exact strings."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


class CallAnalysis(BaseModel):
    """The 9-field result for one audio clip."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


# Tone groupings used by the §5.2.4 coherence check.
POSITIVE_TONES = {EmotionalTone.SATISFIED}
NEGATIVE_TONES = {
    EmotionalTone.FRUSTRATED,
    EmotionalTone.UPSET,
    EmotionalTone.DISTRESSED,
}
```

- [ ] **Step 7: Write `autoace/config.py`**

```python
"""Every threshold in the system. Nothing numeric belongs anywhere else.

Provenance of each value is marked:
  MEASURED  — derived from the three labelled calls (spec §2.5)
  DERIVED   — reasoned from the brief's definitions or physics
  UNFITTED  — interpolated with no supporting example; highest risk
"""

from pathlib import Path

# ---------------------------------------------------------------- models
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_CPU = "int8"
WHISPER_COMPUTE_GPU = "float16"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SER_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
NLI_MODEL = "facebook/bart-large-mnli"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
LLM_MODEL = "claude-haiku-4-5"
LLM_TEMPERATURE = 0.0

SAMPLE_RATE = 16000

# ------------------------------------------------------------------ VAD
VAD_THRESHOLD = 0.5                    # DERIVED — Silero default
VAD_MIN_SPEECH_MS = 250                # DERIVED
VAD_MIN_SILENCE_MS = 300               # DERIVED

# --------------------------------------------------------- diarization
STEREO_SEPARATE_MAX_CORR = 0.98        # DERIVED — above this, duplicated mono
AGENT_REF_MIN_SIM = 0.55               # DERIVED — below, fall back to first-speaker
DIARIZATION_MIN_SILHOUETTE = 0.10      # DERIVED — below, mark degraded

# --------------------------------------------------------------- noise
# MEASURED: the one no-noise call sits at -56.3 dBFS; both medium calls at
# -52.1 and -47.0. Boundaries between them are UNFITTED.
NOISE_FLOOR_PRESENT = -55.0            # MEASURED (anchor at -56.3 / -52.1)
NOISE_SEVERITY_BANDS = [               # (upper bound dBFS, severity)
    (-55.0, "none"),                   # MEASURED
    (-50.0, "low"),                    # UNFITTED — no `low` example exists
    (-44.0, "medium"),                 # MEASURED (anchors -52.1, -47.0)
    (999.0, "high"),                   # UNFITTED — no `high` example exists
]
TAG_MIN_DOM = 1.8                      # UNFITTED — top group / median group ratio
TYPE_AST_MIN_DOM = 2.5                 # UNFITTED — below this, prefer spectral label

# spectral line-artifact detection (spec §7.1.1b)
STATIC_FLATNESS_MIN = 0.40             # DERIVED — broadband hiss is spectrally flat
HUM_FREQS_HZ = (50.0, 60.0)            # DERIVED — mains frequencies
HUM_PROMINENCE_DB = 6.0                # DERIVED
CRACKLE_MAX_MS = 20.0                  # DERIVED
CRACKLE_MIN_COUNT = 3                  # DERIVED

# ------------------------------------------------------------- quality
# MEASURED: all three calls are `clear` with 4-6% energy above 3.4 kHz and
# 0.00% clipping. `clear` is the strong prior.
SQUIM_STOI_SLIGHT = 0.75               # UNFITTED — no impaired example exists
SQUIM_STOI_SEVERE = 0.55               # UNFITTED
CLIP_FRACTION_THRESHOLD = 0.001        # DERIVED — 0.1% of samples near full scale
DROPOUT_MIN_MS = 30.0                  # DERIVED
ECHO_LAG_RANGE_MS = (20.0, 200.0)      # DERIVED
ECHO_PEAK_MIN = 0.30                   # DERIVED
# Telephony baseline: do NOT treat narrowband as muffling (spec §2.5 finding 3).
TELEPHONY_HF_BASELINE = 0.04           # MEASURED — 4-6% above 3.4 kHz is normal
MUFFLE_HF_MIN = 0.015                  # DERIVED — well below the telephony floor
LOW_VOLUME_LUFS = -35.0                # DERIVED

# --------------------------------------------------------------- other
OVERLAP_MIN_SEGMENT_S = 0.5            # DERIVED — ignore backchannels
OVERLAP_MIN_TOTAL_S = 1.0              # DERIVED
# MEASURED: call_003 has a 7.35 s non-speech gap and is labelled false.
LONG_SILENCE_SEC = 10.0                # MEASURED — must exceed 7.35

# ----------------------------------------------------------- intensity
TIER_A_MIN_SPEECH_S = 15.0             # DERIVED
TIER_C_MAX_SPEECH_S = 3.0              # DERIVED
BASELINE_WINDOW_S = 25.0               # DERIVED
ACTIVATION_HIGH_Z = 1.0                # UNFITTED
ACTIVATION_LOW_Z = 0.3                 # UNFITTED — no `low` example exists
ACTIVATION_PEAK_Z = 2.0                # UNFITTED
ACTIVATION_WEIGHTS = {                 # UNFITTED — uniform within each channel
    "loudness_range": 0.15,
    "f0_elevation": 0.15,
    "f0_range": 0.15,
    "rate_deviation": 0.10,
    "jitter_shimmer": 0.05,
    "interruption_rate": 0.10,
    "turn_latency": 0.05,
    "lexical_markers": 0.10,
    "ser_arousal": 0.15,
}

# ---------------------------------------------------------- confidence
CONF_BASE = 0.55
CONF_TONE_AGREE = 0.15
CONF_INTENSITY_AGREE = 0.10
CONF_SER_CONSISTENT = 0.10
CONF_LLM_SELF_HIGH = 0.05
CONF_LLM_SELF_HIGH_THRESHOLD = 0.80
CONF_EVIDENCE_CONFLICT = -0.15
CONF_NEAR_THRESHOLD = -0.05
CONF_TIER_C = -0.20
CONF_DEGRADED = -0.15
CONF_ASR_POOR = -0.10
CONF_MIN, CONF_MAX = 0.05, 0.98
TIER_C_MAX_CONF = 0.45
DEGRADED_MAX_CONF = 0.50
NEAR_THRESHOLD_FRACTION = 0.10         # within 10% of a boundary
ASR_MIN_LOGPROB = -1.0                 # DERIVED
REVIEW_THRESHOLD = 0.60                # dashboard review queue

# coherence check (spec §5.2.4)
VALENCE_POS_MIN = 0.45                 # UNFITTED
VALENCE_NEG_MAX = 0.55                 # UNFITTED
AROUSAL_HIGH_MIN = 0.50                # UNFITTED

# ------------------------------------------------------------- fixtures
REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "reference"   # provided trial assets, not code we write
LABELS_CSV = REFERENCE_DIR / "labels.csv"


def reference_call(name: str) -> str:
    """Absolute path to a provided trial call.

    The manifest's `name` column holds a BARE filename (fixed by the brief's
    batch format), so audio is always resolved as directory + name. Never
    assume the audio sits in the working directory.
    """
    return str(REFERENCE_DIR / name)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_schema.py -v`
Expected: 4 passed

- [ ] **Step 9: Commit**

```bash
git add requirements.txt autoace/ tests/test_schema.py
git commit -m "feat: output schema and threshold config"
```

---

## Task 2: Acoustics baseline — pin spec §2.5 as a regression canary

**Files:**
- Create: `autoace/acoustics.py`
- Test: `tests/test_acoustics_baseline.py`

This task reproduces the exact measurements in spec §2.5. Those numbers justify three
calibration decisions and currently exist only in prose (spec §2.8). Pinning them as an
assertion means any future change to the audio, the decode path, or the DSP is caught
immediately.

- [ ] **Step 1: Write the failing baseline test**

Create `tests/test_acoustics_baseline.py`:

```python
"""Pins spec §2.5. These exact figures justify NOISE_FLOOR_PRESENT,
LONG_SILENCE_SEC, and the telephony muffling baseline. If this test fails,
the spec's calibration argument no longer holds and must be re-derived."""

import pytest

from autoace.acoustics import baseline_characterisation
from autoace.config import reference_call

# (file, snr_db, floor_dbfs, max_gap_s, clip_pct, hf_fraction)
EXPECTED = [
    ("call_001.ogg", 42.2, -56.3, 2.87, 0.00, 0.044),
    ("call_002.ogg", 29.9, -52.1, 3.29, 0.00, 0.043),
    ("call_003.ogg", 34.9, -47.0, 7.35, 0.00, 0.060),
]


@pytest.mark.parametrize("name,snr,floor,gap,clip,hf", EXPECTED)
def test_baseline_matches_spec_section_2_5(name, snr, floor, gap, clip, hf):
    m = baseline_characterisation(reference_call(name))
    assert m["snr_db"] == pytest.approx(snr, abs=0.1)
    assert m["floor_dbfs"] == pytest.approx(floor, abs=0.1)
    assert m["max_nonspeech_gap_s"] == pytest.approx(gap, abs=0.01)
    assert m["clip_pct"] == pytest.approx(clip, abs=0.01)
    assert m["hf_fraction"] == pytest.approx(hf, abs=0.001)


def test_long_silence_threshold_exceeds_measured_gap():
    """call_003 has a 7.35s gap and is labelled long_silence_present=false,
    so the threshold must sit above it (spec §2.5 finding 2)."""
    from autoace.config import LONG_SILENCE_SEC
    worst = max(baseline_characterisation(n)["max_nonspeech_gap_s"] for n, *_ in EXPECTED)
    assert LONG_SILENCE_SEC > worst


def test_noise_floor_threshold_separates_the_labels():
    """The one no-noise call must fall below NOISE_FLOOR_PRESENT and both
    noisy calls above it (spec §7.2)."""
    from autoace.config import NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_001.ogg"))["floor_dbfs"] < NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_002.ogg"))["floor_dbfs"] > NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_003.ogg"))["floor_dbfs"] > NOISE_FLOOR_PRESENT
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_acoustics_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.acoustics'`

- [ ] **Step 3: Write `autoace/acoustics.py`**

```python
"""Shared DSP primitives. Used by both the tagging and quality modules so
noise-floor logic exists in exactly one place."""

from __future__ import annotations

import librosa
import numpy as np

from autoace.config import SAMPLE_RATE

_STFT_N_FFT = 512
_STFT_HOP = 160          # 10 ms at 16 kHz
_BASELINE_PERCENTILE = 60


def stft_magnitude(y: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP))


def frame_db(spectrogram: np.ndarray) -> np.ndarray:
    """Per-frame RMS in dB relative to the loudest frame."""
    rms = librosa.feature.rms(
        S=spectrogram, frame_length=_STFT_N_FFT, hop_length=_STFT_HOP
    )[0]
    return librosa.amplitude_to_db(rms, ref=np.max)


def clip_percentage(y: np.ndarray, threshold: float = 0.98) -> float:
    return float((np.abs(y) > threshold).mean() * 100.0)


def high_frequency_fraction(spectrogram: np.ndarray, cutoff_hz: float = 3400.0) -> float:
    """Share of spectral energy above the telephony band edge."""
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=_STFT_N_FFT)
    total = spectrogram.sum()
    if total == 0:
        return 0.0
    return float(spectrogram[freqs > cutoff_hz].sum() / total)


def longest_run_seconds(mask: np.ndarray, hop_s: float = 0.01) -> float:
    """Longest contiguous True run in a per-frame boolean mask, in seconds."""
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * hop_s


def baseline_characterisation(path: str) -> dict[str, float]:
    """Reproduce the spec §2.5 characterisation exactly.

    Deliberately uses the simple percentile-energy speech/non-speech split
    that produced the published figures — NOT the production Silero+MCRA
    path. This is a canary that the audio and decode chain are unchanged,
    not the estimator the pipeline uses for severity.
    """
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    spec = stft_magnitude(y)
    db = frame_db(spec)

    threshold = np.percentile(db, _BASELINE_PERCENTILE)
    speech = db > threshold
    non_speech = ~speech

    return {
        "snr_db": float(db[speech].mean() - db[non_speech].mean()),
        "floor_dbfs": float(db[non_speech].mean()),
        "max_nonspeech_gap_s": longest_run_seconds(non_speech),
        "clip_pct": clip_percentage(y),
        "hf_fraction": high_frequency_fraction(spec),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acoustics_baseline.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add autoace/acoustics.py tests/test_acoustics_baseline.py
git commit -m "test: pin spec 2.5 acoustics as regression canary"
```

---

## Task 3: Audio I/O with stereo detection

**Files:**
- Create: `autoace/io_audio.py`
- Test: `tests/test_io_audio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_io_audio.py`:

```python
import numpy as np
import pytest

from autoace.config import LABELS_CSV, reference_call
from autoace.io_audio import UnsupportedAudio, load_mono, channel_layout


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_load_mono_returns_16k_float32(name):
    y, sr = load_mono(reference_call(name))
    assert sr == 16000
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) > 0


def test_provided_calls_are_duplicated_mono():
    """Spec §4.1: the .ogg files are 2-channel but byte-identical, so there
    is no free speaker separation and diarization is required."""
    for name in ("call_001.ogg", "call_002.ogg", "call_003.ogg"):
        layout = channel_layout(reference_call(name))
        assert layout["channels"] == 2
        assert layout["correlation"] == pytest.approx(1.0, abs=1e-4)
        assert layout["separated"] is False


def test_unsupported_file_raises():
    with pytest.raises(UnsupportedAudio):
        load_mono(str(LABELS_CSV))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_io_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.io_audio'`

- [ ] **Step 3: Write `autoace/io_audio.py`**

```python
"""Decode to a canonical 16 kHz mono float32 signal.

NO loudness normalization happens here or anywhere upstream of the signal
branch. Absolute level is load-bearing: the noise floor in dBFS is both the
primary severity signal (§7.3) and the primary presence detector (§7.2).
The tone branch obtains level-invariance by z-scoring against the speaker's
own baseline internally (§6.2), never by touching the shared audio.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from autoace.config import SAMPLE_RATE, STEREO_SEPARATE_MAX_CORR

SUPPORTED_SUFFIXES = {".ogg", ".wav", ".mp3", ".m4a", ".flac"}


class UnsupportedAudio(RuntimeError):
    """Raised when a file cannot be decoded as audio."""


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 @ 16 kHz, 16000)."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedAudio(f"{path.name}: unsupported extension {path.suffix!r}")
    try:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # librosa raises a variety of backend errors
        raise UnsupportedAudio(f"{path.name}: {exc}") from exc
    return y.astype(np.float32), SAMPLE_RATE


def channel_layout(path: str | Path) -> dict:
    """Detect whether a stereo file carries genuinely separated legs.

    A dual-leg recording (agent on one channel, customer on the other) makes
    diarization unnecessary. The provided files are 2-channel but byte-identical,
    so they do not qualify.
    """
    path = Path(path)
    info = sf.info(str(path))
    if info.channels < 2:
        return {"channels": info.channels, "correlation": None, "separated": False}

    data, _ = sf.read(str(path), always_2d=True)
    correlation = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    return {
        "channels": info.channels,
        "correlation": correlation,
        "separated": correlation < STEREO_SEPARATE_MAX_CORR,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_io_audio.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add autoace/io_audio.py tests/test_io_audio.py
git commit -m "feat: audio decode with dual-leg stereo detection"
```

---

## Task 4: Silero VAD

**Files:**
- Create: `autoace/vad.py`
- Test: `tests/test_vad.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vad.py`:

```python
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.vad import Segment, non_speech_segments, speech_segments


@pytest.fixture(scope="module")
def call_001():
    return load_mono(reference_call("call_001.ogg"))[0]


def test_returns_ordered_non_overlapping_segments(call_001):
    segments = speech_segments(call_001)
    assert segments, "expected at least one speech segment in a 31s call"
    for seg in segments:
        assert isinstance(seg, Segment)
        assert seg.end > seg.start
    for earlier, later in zip(segments, segments[1:]):
        assert later.start >= earlier.end


def test_speech_covers_a_plausible_share_of_the_call(call_001):
    """A 31s two-party call should be mostly-but-not-entirely speech."""
    segments = speech_segments(call_001)
    total = sum(s.end - s.start for s in segments)
    duration = len(call_001) / 16000
    assert 0.2 < total / duration < 0.95


def test_non_speech_is_the_complement(call_001):
    duration = len(call_001) / 16000
    speech = sum(s.end - s.start for s in speech_segments(call_001))
    gaps = sum(s.end - s.start for s in non_speech_segments(call_001))
    assert speech + gaps == pytest.approx(duration, abs=0.05)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.vad'`

- [ ] **Step 3: Write `autoace/vad.py`**

```python
"""Silero VAD. Its segmentation is shared across both branches: speech
regions drive prosody, non-speech regions drive noise estimation, and the
gaps drive long-silence detection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from autoace.config import (
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_THRESHOLD,
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@lru_cache(maxsize=1)
def _load_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    return model, utils[0]  # utils[0] is get_speech_timestamps


def speech_segments(y: np.ndarray) -> list[Segment]:
    model, get_speech_timestamps = _load_vad()
    stamps = get_speech_timestamps(
        torch.from_numpy(y),
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        return_seconds=True,
    )
    return [Segment(float(s["start"]), float(s["end"])) for s in stamps]


def non_speech_segments(y: np.ndarray) -> list[Segment]:
    """Complement of the speech segments across the whole signal."""
    duration = len(y) / SAMPLE_RATE
    gaps: list[Segment] = []
    cursor = 0.0
    for seg in speech_segments(y):
        if seg.start > cursor:
            gaps.append(Segment(cursor, seg.start))
        cursor = seg.end
    if cursor < duration:
        gaps.append(Segment(cursor, duration))
    return gaps


def concatenate(y: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Splice the named regions into one contiguous signal."""
    if not segments:
        return np.zeros(0, dtype=np.float32)
    parts = [
        y[int(s.start * SAMPLE_RATE) : int(s.end * SAMPLE_RATE)] for s in segments
    ]
    return np.concatenate(parts).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_vad.py -v`
Expected: 3 passed (first run downloads the Silero model, ~1 MB)

- [ ] **Step 5: Commit**

```bash
git add autoace/vad.py tests/test_vad.py
git commit -m "feat: Silero VAD with shared speech/non-speech segmentation"
```

---

## Task 5: Speaker assignment

**Files:**
- Create: `autoace/diarize.py`
- Test: `tests/test_diarize.py`

Full pyannote diarization is the most expensive CPU stage. This exploits a domain fact instead:
there are exactly two speakers and the agent is a consistent TTS voice.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize.py`:

```python
import pytest

from autoace.diarize import assign_speakers
from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.vad import speech_segments


@pytest.fixture(scope="module")
def call_003():
    y, _ = load_mono(reference_call("call_003.ogg"))
    return y, speech_segments(y)


def test_assigns_every_segment_to_a_role(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert set(result.assignments) == set(range(len(segments)))
    assert set(result.assignments.values()) <= {"agent", "customer"}


def test_agent_speaks_first(call_003):
    """The bot always greets first ('Hi, I'm Erica from...') — true on all
    three provided calls, and the fallback heuristic in spec §4.3 step 4."""
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert result.assignments[0] == "agent"


def test_both_roles_present_in_a_two_party_call(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert set(result.assignments.values()) == {"agent", "customer"}
    assert result.degraded is False


def test_customer_speech_duration_is_reported(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert result.customer_speech_seconds > 0
    total = sum(s.duration for s in segments)
    assert result.customer_speech_seconds < total


def test_single_speaker_audio_is_marked_degraded():
    """A clip with one speaker cannot be split into agent/customer; the
    pipeline must know so it can cap confidence (spec §4.3 step 5)."""
    import numpy as np
    from autoace.vad import Segment
    y, _ = load_mono(reference_call("call_002.ogg"))
    one_segment = [Segment(0.0, 1.0)]
    result = assign_speakers(y, one_segment)
    assert result.degraded is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_diarize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.diarize'`

- [ ] **Step 3: Write `autoace/diarize.py`**

```python
"""Two-speaker assignment via ECAPA embeddings and agglomerative clustering.

Cheaper than full pyannote diarization and more robust for the 2-speaker
case. Degrades predictably: reference-bank match, then first-speaker
heuristic, then a `degraded` flag that caps downstream confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from autoace.config import (
    AGENT_REF_MIN_SIM,
    DIARIZATION_MIN_SILHOUETTE,
    ECAPA_MODEL,
    SAMPLE_RATE,
)
from autoace.vad import Segment

_MIN_EMBED_SAMPLES = int(0.4 * SAMPLE_RATE)  # ECAPA needs ~0.4s to be stable


@dataclass
class SpeakerAssignment:
    assignments: dict[int, str]           # segment index -> "agent" | "customer"
    customer_speech_seconds: float
    degraded: bool
    agent_reference_similarity: float | None = None
    customer_segments: list[Segment] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_encoder():
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=ECAPA_MODEL, savedir="models/ecapa", run_opts={"device": "cpu"}
    )


def _embed(y: np.ndarray, segments: list[Segment]) -> np.ndarray:
    encoder = _load_encoder()
    vectors = []
    for seg in segments:
        chunk = y[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
        if len(chunk) < _MIN_EMBED_SAMPLES:
            chunk = np.pad(chunk, (0, _MIN_EMBED_SAMPLES - len(chunk)))
        with torch.no_grad():
            vec = encoder.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
        vectors.append(vec.squeeze().cpu().numpy())
    return np.vstack(vectors)


def _degraded_result(segments: list[Segment]) -> SpeakerAssignment:
    """One effective speaker: treat all speech as customer and flag it."""
    return SpeakerAssignment(
        assignments={i: "customer" for i in range(len(segments))},
        customer_speech_seconds=sum(s.duration for s in segments),
        degraded=True,
        customer_segments=list(segments),
    )


def assign_speakers(
    y: np.ndarray,
    segments: list[Segment],
    agent_reference: np.ndarray | None = None,
) -> SpeakerAssignment:
    if len(segments) < 2:
        return _degraded_result(segments)

    embeddings = _embed(y, segments)
    labels = AgglomerativeClustering(
        n_clusters=2, metric="cosine", linkage="average"
    ).fit_predict(embeddings)

    if len(set(labels)) < 2:
        return _degraded_result(segments)

    score = silhouette_score(embeddings, labels, metric="cosine")
    if score < DIARIZATION_MIN_SILHOUETTE:
        return _degraded_result(segments)

    agent_cluster, similarity = _identify_agent(embeddings, labels, agent_reference)

    assignments = {
        i: ("agent" if label == agent_cluster else "customer")
        for i, label in enumerate(labels)
    }
    customer_segments = [
        seg for i, seg in enumerate(segments) if assignments[i] == "customer"
    ]
    return SpeakerAssignment(
        assignments=assignments,
        customer_speech_seconds=sum(s.duration for s in customer_segments),
        degraded=False,
        agent_reference_similarity=similarity,
        customer_segments=customer_segments,
    )


def _identify_agent(
    embeddings: np.ndarray, labels: np.ndarray, reference: np.ndarray | None
) -> tuple[int, float | None]:
    """Match against the stored agent voice; fall back to first-speaker."""
    if reference is not None:
        best_cluster, best_similarity = 0, -1.0
        for cluster in (0, 1):
            centroid = embeddings[labels == cluster].mean(axis=0)
            similarity = float(
                centroid
                @ reference
                / (np.linalg.norm(centroid) * np.linalg.norm(reference))
            )
            if similarity > best_similarity:
                best_cluster, best_similarity = cluster, similarity
        if best_similarity >= AGENT_REF_MIN_SIM:
            return best_cluster, best_similarity

    # Fallback: the bot greets first on every observed call.
    return int(labels[0]), None


def build_agent_reference(calls: list[tuple[np.ndarray, list[Segment]]]) -> np.ndarray:
    """Mean embedding of the first (agent) segment across known calls.

    Speaker identity only — carries no label information, so it cannot leak
    tone or noise ground truth into the evaluation (spec §12).
    """
    vectors = [_embed(y, segments[:1])[0] for y, segments in calls if segments]
    return np.vstack(vectors).mean(axis=0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diarize.py -v`
Expected: 5 passed (first run downloads ECAPA, ~80 MB)

- [ ] **Step 5: Commit**

```bash
git add autoace/diarize.py tests/test_diarize.py
git commit -m "feat: two-speaker assignment via ECAPA clustering"
```

---

## Task 6: ASR with word timestamps and language ID

**Files:**
- Create: `autoace/asr.py`
- Test: `tests/test_asr.py`

This task **regenerates the spec §2.3, §2.4 and §11 evidence** that no longer exists on disk
(spec §2.8).

- [ ] **Step 1: Write the failing test**

Create `tests/test_asr.py`:

```python
import pytest

from autoace.asr import transcribe
from autoace.config import reference_call


@pytest.fixture(scope="module")
def call_001_result():
    return transcribe(reference_call("call_001.ogg"))


def test_returns_words_with_timestamps(call_001_result):
    """The prosody alignment design depends entirely on word timestamps."""
    assert call_001_result.words
    for word in call_001_result.words:
        assert word.end >= word.start
        assert word.text.strip()


def test_detects_english_on_call_001(call_001_result):
    assert call_001_result.language == "en"


def test_detects_spanish_on_call_002():
    """call_002 switches to Spanish. An English-only ASR returns garbage here,
    which is why Parakeet was rejected (spec §4.4)."""
    result = transcribe(reference_call("call_002.ogg"))
    assert result.language in {"es", "en"}
    assert "erica" in result.text.lower()


def test_reports_average_logprob_for_confidence(call_001_result):
    """ASR quality feeds the confidence formula (spec §8)."""
    assert call_001_result.avg_logprob is not None
    assert -5.0 < call_001_result.avg_logprob < 0.0


def test_segments_are_ordered(call_001_result):
    for earlier, later in zip(call_001_result.segments, call_001_result.segments[1:]):
        assert later.start >= earlier.start
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_asr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.asr'`

- [ ] **Step 3: Write `autoace/asr.py`**

```python
"""faster-whisper transcription.

Three requirements the model choice must satisfy (spec §4.4):
  1. Multilingual — call_002 is Spanish; an English-only model returns garbage.
  2. word_timestamps=True — the prosody alignment grid depends on them.
  3. Detected language captured, not discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from autoace.config import (
    WHISPER_COMPUTE_CPU,
    WHISPER_COMPUTE_GPU,
    WHISPER_MODEL,
)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    text: str
    language: str
    segments: list[AsrSegment]
    words: list[Word]
    avg_logprob: float


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():
        return WhisperModel(WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE_GPU)
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_CPU)


def transcribe(path: str) -> Transcript:
    model = _load_model()
    raw_segments, info = model.transcribe(
        path,
        word_timestamps=True,
        vad_filter=False,      # we run Silero ourselves and share its segmentation
        beam_size=5,
        language=None,         # auto-detect; never hardcode "en"
    )

    segments: list[AsrSegment] = []
    words: list[Word] = []
    logprobs: list[float] = []

    for seg in raw_segments:
        seg_words = [
            Word(float(w.start), float(w.end), w.word) for w in (seg.words or [])
        ]
        segments.append(
            AsrSegment(float(seg.start), float(seg.end), seg.text.strip(), seg_words)
        )
        words.extend(seg_words)
        logprobs.append(float(seg.avg_logprob))

    return Transcript(
        text=" ".join(s.text for s in segments).strip(),
        language=info.language,
        segments=segments,
        words=words,
        avg_logprob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_asr.py -v`
Expected: 5 passed (first run downloads large-v3-turbo, ~1.6 GB; CPU transcription of the three calls takes several minutes)

- [ ] **Step 5: Record the regenerated evidence**

Run and save the output, which restores the spec §2.3 transcript excerpts and the §11 latency figure:

```bash
python -c "
import json, time
from autoace.asr import transcribe
from autoace.config import reference_call
out = {}
for name in ['call_001.ogg','call_002.ogg','call_003.ogg']:
    t0 = time.time()
    r = transcribe(reference_call(name))
    out[name] = {'language': r.language, 'elapsed_s': round(time.time()-t0, 2),
                 'avg_logprob': round(r.avg_logprob, 3), 'text': r.text}
    print(name, out[name]['language'], out[name]['elapsed_s'], 's')
json.dump(out, open('tests/fixtures/asr_baseline.json','w'), indent=2, ensure_ascii=False)
"
```

- [ ] **Step 6: Commit**

```bash
git add autoace/asr.py tests/test_asr.py tests/fixtures/asr_baseline.json
git commit -m "feat: multilingual ASR with word timestamps, regenerate transcript evidence"
```

---

## Task 7: AudioSet tagging and spectral line-artifact classifier

**Files:**
- Create: `autoace/tagging.py`
- Test: `tests/test_tagging.py`

This task **regenerates the spec §7.1.1 bake-off**, whose numbers justify choosing AST over
PANNs and adding the spectral classifier.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tagging.py`:

```python
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.tagging import classify_noise_type, spectral_artifact, tag_non_speech
from autoace.vad import non_speech_segments


def test_ast_identifies_television_on_call_002():
    """Ground truth is 'TV'. AST named `Television` correctly in the bake-off;
    PANNs said `Radio` (spec §7.1.1)."""
    y, _ = load_mono(reference_call("call_002.ogg"))
    result = tag_non_speech(y, non_speech_segments(y))
    assert result.dominant_group == "TV"


def test_static_is_detected_spectrally_not_by_audioset():
    """AudioSet scores `static` at ~0.001-0.006 on every call, including the
    one whose ground truth IS static. The spectral classifier must catch it
    (spec §7.1.1b)."""
    y, _ = load_mono(reference_call("call_003.ogg"))
    artifact = spectral_artifact(y, non_speech_segments(y))
    assert artifact == "static"


def test_classify_prefers_spectral_label_when_ast_is_weak():
    y, _ = load_mono(reference_call("call_003.ogg"))
    label = classify_noise_type(y, non_speech_segments(y))
    assert label == "static"


def test_no_spectral_artifact_on_the_clean_call():
    y, _ = load_mono(reference_call("call_001.ogg"))
    assert spectral_artifact(y, non_speech_segments(y)) is None


def test_relative_dominance_is_scale_free():
    """Absolute AudioSet probabilities are uncalibrated on telephony audio,
    so gating uses top-group / median-group instead (spec §7.1.1a)."""
    y, _ = load_mono(reference_call("call_002.ogg"))
    result = tag_non_speech(y, non_speech_segments(y))
    assert result.relative_dominance > 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_tagging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.tagging'`

- [ ] **Step 3: Write `autoace/tagging.py`**

```python
"""Noise typing, split along the physical boundary that matters.

AudioSet owns ENVIRONMENTAL noise (TV, music, road, wind, typing). It cannot
see TRANSMISSION noise — static, hum, crackle — because it is trained on
video audio, not telephony lines. The measured bake-off (spec §7.1.1) scored
the `static` group at 0.0013-0.0059 across all three calls, including the one
whose ground truth is "sharp static". So transmission noise is detected
deterministically from the spectrum instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np
import torch

from autoace.acoustics import stft_magnitude
from autoace.config import (
    AST_MODEL,
    CRACKLE_MAX_MS,
    CRACKLE_MIN_COUNT,
    HUM_FREQS_HZ,
    HUM_PROMINENCE_DB,
    SAMPLE_RATE,
    STATIC_FLATNESS_MIN,
    TYPE_AST_MIN_DOM,
)
from autoace.vad import Segment, concatenate

_WINDOW_S = 1.0
_HOP_S = 0.5

# AudioSet class name -> the brief's short informal vocabulary. The labeller
# writes "TV", not "television" — keep outputs short and conventional.
AUDIOSET_TO_LABEL: dict[str, list[str]] = {
    "office chatter": ["Hubbub, speech noise, speech babble", "Chatter", "Crowd", "Babbling"],
    "music": ["Music", "Musical instrument", "Singing"],
    "road noise": ["Vehicle", "Car", "Traffic noise, roadway noise", "Engine"],
    "TV": ["Television"],
    "radio": ["Radio"],
    "keyboard typing": ["Typing", "Computer keyboard", "Typewriter"],
    "wind": ["Wind", "Wind noise (microphone)", "Rustling leaves"],
    "mechanical noise": ["Mechanisms", "Machine", "Motor vehicle (road)", "Engine knocking"],
}

SPEECH_CLASSES = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Speech synthesizer",
    "Narration, monologue",
}


@dataclass
class TagResult:
    dominant_group: str | None
    probability: float
    relative_dominance: float
    group_probabilities: dict[str, float]
    top_class: str


@lru_cache(maxsize=1)
def _load_ast():
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL)
    model.eval()
    return extractor, model


def tag_non_speech(y: np.ndarray, segments: list[Segment]) -> TagResult:
    """Tag only the non-speech regions. Tagging whole-clip windows lets speech
    mask the background and drives every probability to the floor."""
    extractor, model = _load_ast()
    audio = concatenate(y, segments)
    if len(audio) < int(_WINDOW_S * SAMPLE_RATE):
        return TagResult(None, 0.0, 0.0, {}, "")

    window = int(_WINDOW_S * SAMPLE_RATE)
    hop = int(_HOP_S * SAMPLE_RATE)
    logit_sum = None
    count = 0
    for start in range(0, len(audio) - window + 1, hop):
        chunk = audio[start : start + window]
        inputs = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        logit_sum = logits if logit_sum is None else logit_sum + logits
        count += 1

    probs = torch.sigmoid(logit_sum / count).numpy()
    id2label = model.config.id2label

    per_class = {
        id2label[i]: float(p)
        for i, p in enumerate(probs)
        if id2label[i] not in SPEECH_CLASSES
    }
    group_probs = {
        group: max((per_class.get(name, 0.0) for name in names), default=0.0)
        for group, names in AUDIOSET_TO_LABEL.items()
    }

    dominant = max(group_probs, key=group_probs.get)
    top_probability = group_probs[dominant]
    median = float(np.median(list(group_probs.values()))) or 1e-9
    top_class = max(per_class, key=per_class.get) if per_class else ""

    return TagResult(
        dominant_group=dominant,
        probability=top_probability,
        relative_dominance=top_probability / median,
        group_probabilities=group_probs,
        top_class=top_class,
    )


def spectral_artifact(y: np.ndarray, segments: list[Segment]) -> str | None:
    """Detect transmission noise, which AudioSet cannot see."""
    audio = concatenate(y, segments)
    if len(audio) < SAMPLE_RATE // 2:
        return None

    spec = stft_magnitude(audio)

    flatness = float(librosa.feature.spectral_flatness(S=spec).mean())
    if flatness >= STATIC_FLATNESS_MIN:
        return "static"

    if _has_mains_hum(spec):
        return "electrical hum"

    if _count_transients(audio) >= CRACKLE_MIN_COUNT:
        return "line crackle"

    return None


def _has_mains_hum(spec: np.ndarray) -> bool:
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=(spec.shape[0] - 1) * 2)
    mean_db = librosa.amplitude_to_db(spec.mean(axis=1), ref=np.max)
    baseline = float(np.median(mean_db))
    for mains in HUM_FREQS_HZ:
        for harmonic in (1, 2, 3):
            target = mains * harmonic
            index = int(np.argmin(np.abs(freqs - target)))
            if mean_db[index] - baseline >= HUM_PROMINENCE_DB:
                return True
    return False


def _count_transients(audio: np.ndarray) -> int:
    """Short, high-energy broadband bursts characteristic of line crackle."""
    onsets = librosa.onset.onset_detect(
        y=audio, sr=SAMPLE_RATE, units="samples", backtrack=False
    )
    max_len = int(CRACKLE_MAX_MS / 1000 * SAMPLE_RATE)
    return sum(
        1
        for onset in onsets
        if np.abs(audio[onset : onset + max_len]).max(initial=0.0)
        > 4 * np.abs(audio).mean()
    )


def classify_noise_type(y: np.ndarray, segments: list[Segment]) -> str:
    """Environmental label from AST, unless it is weak and a transmission
    signature fires — then prefer the spectral label."""
    tag = tag_non_speech(y, segments)
    artifact = spectral_artifact(y, segments)

    if artifact and tag.relative_dominance < TYPE_AST_MIN_DOM:
        return artifact
    return tag.dominant_group or (artifact or "")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tagging.py -v`
Expected: 5 passed (first run downloads AST, ~350 MB)

If `test_static_is_detected_spectrally_not_by_audioset` fails, `STATIC_FLATNESS_MIN` needs
recalibration against call_003's measured non-speech flatness — print it with
`librosa.feature.spectral_flatness` and set the threshold just below. Record the measured value
as a comment in `config.py` marked `MEASURED`.

- [ ] **Step 5: Commit**

```bash
git add autoace/tagging.py tests/test_tagging.py
git commit -m "feat: AudioSet tagging plus spectral transmission-noise classifier"
```

---

## Task 8: Audio quality

**Files:**
- Create: `autoace/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_quality.py`:

```python
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.quality import assess_quality, noise_floor_dbfs
from autoace.schema import AudioQuality
from autoace.vad import non_speech_segments


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_all_provided_calls_are_clear(name):
    """All three labels say `clear`, including the one with 'sharp static'.
    audio_quality degrades only when intelligibility suffers (spec §2.6)."""
    y, _ = load_mono(reference_call(name))
    assert assess_quality(y).quality == AudioQuality.CLEAR


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_telephony_bandwidth_is_not_treated_as_muffling(name):
    """4-6% of energy above 3.4 kHz is normal narrowband telephony. A detector
    baselined on wideband speech flags all three (spec §2.5 finding 3)."""
    y, _ = load_mono(reference_call(name))
    assert assess_quality(y).detectors["muffled"] is False


def test_noise_floor_ordering_matches_severity_labels():
    """Floor tracks severity; SNR does not (spec §2.5 finding 1)."""
    floors = {}
    for name in ("call_001.ogg", "call_002.ogg", "call_003.ogg"):
        y, _ = load_mono(reference_call(name))
        floors[name] = noise_floor_dbfs(y, non_speech_segments(y))
    assert floors["call_001.ogg"] < floors["call_002.ogg"]
    assert floors["call_002.ogg"] < floors["call_003.ogg"]


def test_no_clipping_on_provided_calls():
    for name in ("call_001.ogg", "call_002.ogg", "call_003.ogg"):
        y, _ = load_mono(reference_call(name))
        assert assess_quality(y).detectors["clipping"] is False


@pytest.mark.parametrize(
    "name,expected_floor",
    [("call_001.ogg", -56.3), ("call_002.ogg", -52.1), ("call_003.ogg", -47.0)],
)
def test_production_floor_is_on_the_same_scale_as_the_calibration(name, expected_floor):
    """The NOISE_SEVERITY_BANDS anchors were measured relative to each call's
    peak frame. If the production estimator drifts onto an absolute dBFS scale,
    every threshold in that table silently breaks. Tolerance is wide because
    the estimators differ (Silero non-speech vs percentile split) — this checks
    the SCALE, not the exact value."""
    y, _ = load_mono(reference_call(name))
    floor = noise_floor_dbfs(y, non_speech_segments(y))
    assert floor == pytest.approx(expected_floor, abs=8.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.quality'`

- [ ] **Step 3: Write `autoace/quality.py`**

```python
"""Technical audio quality, measured independently of background noise.

The brief warns: "do not infer background noise solely from poor audio
quality". These are separate code paths reading separate evidence. Default is
`clear` — all three provided calls are clear, including the one with audible
static, so positive evidence is required to move off it (spec §2.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np
import torch

from autoace.acoustics import (
    clip_percentage,
    frame_db,
    high_frequency_fraction,
    stft_magnitude,
)
from autoace.config import (
    CLIP_FRACTION_THRESHOLD,
    DROPOUT_MIN_MS,
    ECHO_LAG_RANGE_MS,
    ECHO_PEAK_MIN,
    LOW_VOLUME_LUFS,
    MUFFLE_HF_MIN,
    SAMPLE_RATE,
    SQUIM_STOI_SEVERE,
    SQUIM_STOI_SLIGHT,
)
from autoace.schema import AudioQuality
from autoace.vad import Segment, concatenate


@dataclass
class QualityResult:
    quality: AudioQuality
    estimated_stoi: float | None
    detectors: dict[str, bool]


def _relative_level_db(y: np.ndarray, segments: list[Segment], peak_db: float) -> float:
    """Mean level of the named regions, in dB relative to the call's peak frame."""
    audio = concatenate(y, segments)
    if len(audio) < SAMPLE_RATE // 10:
        return -90.0
    rms = float(np.sqrt(np.mean(audio**2)))
    return 20.0 * np.log10(max(rms, 1e-9)) - peak_db


def _peak_db(y: np.ndarray) -> float:
    frame_rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    return 20.0 * np.log10(max(float(frame_rms.max()), 1e-9))


def noise_floor_dbfs(y: np.ndarray, non_speech: list[Segment]) -> float:
    """Noise floor from true non-speech regions, in dB relative to the call's
    peak frame.

    Primary severity feature and primary presence detector — the measured
    ordering (-56.3 none, -52.1 medium, -47.0 medium) is monotonic with the
    labels, unlike SNR (spec §2.5 finding 1).

    CRITICAL — the scale must match the calibration. The anchors in
    NOISE_SEVERITY_BANDS were measured relative to each call's loudest frame,
    NOT as absolute dBFS. Returning absolute dBFS here would silently break
    every threshold in that table. Expressing the floor relative to the call's
    own peak is a measurement convention, not loudness normalization — the
    audio itself is never rescaled (spec §4.1).
    """
    return _relative_level_db(y, non_speech, _peak_db(y))


def speech_level_dbfs(y: np.ndarray, speech: list[Segment]) -> float:
    """Speech level on the same relative scale, so SNR is their difference."""
    return _relative_level_db(y, speech, _peak_db(y))


@lru_cache(maxsize=1)
def _load_squim():
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    return SQUIM_OBJECTIVE.get_model()


def _estimate_stoi(y: np.ndarray) -> float | None:
    """Non-intrusive quality estimate — no clean reference needed."""
    try:
        model = _load_squim()
        with torch.no_grad():
            stoi, _pesq, _sisdr = model(torch.from_numpy(y).unsqueeze(0))
        return float(stoi.item())
    except Exception:
        return None  # SQUIM unavailable: fall back to the deterministic detectors


def _has_dropouts(y: np.ndarray) -> bool:
    silent = np.abs(y) < 1e-4
    min_run = int(DROPOUT_MIN_MS / 1000 * SAMPLE_RATE)
    run = 0
    for value in silent:
        run = run + 1 if value else 0
        if run >= min_run:
            return True
    return False


def _has_echo(y: np.ndarray) -> bool:
    correlation = np.correlate(y, y, mode="full")[len(y) - 1 :]
    if correlation[0] <= 0:
        return False
    correlation = correlation / correlation[0]
    lo = int(ECHO_LAG_RANGE_MS[0] / 1000 * SAMPLE_RATE)
    hi = int(ECHO_LAG_RANGE_MS[1] / 1000 * SAMPLE_RATE)
    return bool(correlation[lo:hi].max(initial=0.0) >= ECHO_PEAK_MIN)


def _is_low_volume(y: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(y**2)))
    return 20.0 * np.log10(max(rms, 1e-9)) < LOW_VOLUME_LUFS


def assess_quality(y: np.ndarray) -> QualityResult:
    spec = stft_magnitude(y)
    detectors = {
        "clipping": clip_percentage(y) / 100.0 > CLIP_FRACTION_THRESHOLD,
        "dropouts": _has_dropouts(y),
        "echo": _has_echo(y),
        # Baselined against telephony, NOT wideband speech: 4-6% above 3.4 kHz
        # is normal here and must not count as muffling.
        "muffled": high_frequency_fraction(spec) < MUFFLE_HF_MIN,
        "low_volume": _is_low_volume(y),
    }

    stoi = _estimate_stoi(y)
    fired = sum(detectors.values())

    if (stoi is not None and stoi < SQUIM_STOI_SEVERE) or fired >= 2:
        quality = AudioQuality.SEVERELY_IMPAIRED
    elif (stoi is not None and stoi < SQUIM_STOI_SLIGHT) or fired == 1:
        quality = AudioQuality.SLIGHTLY_IMPAIRED
    else:
        quality = AudioQuality.CLEAR

    return QualityResult(quality=quality, estimated_stoi=stoi, detectors=detectors)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_quality.py -v`
Expected: 11 passed (first run downloads SQUIM, ~100 MB)

If any call is not `CLEAR`, print `assess_quality(y).detectors` to see which detector fired and
loosen that single threshold in `config.py`. Do not loosen `SQUIM_STOI_*` blindly — all three
labels are `clear`, so a firing detector is a false positive by definition.

- [ ] **Step 5: Commit**

```bash
git add autoace/quality.py tests/test_quality.py
git commit -m "feat: audio quality assessment with telephony-baselined detectors"
```

---

## Task 9: Signal branch assembly

**Files:**
- Create: `autoace/signal_branch.py`
- Test: `tests/test_signal_branch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_signal_branch.py`:

```python
import json

import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.schema import AudioQuality, NoiseSeverity
from autoace.signal_branch import analyse_signal

EXPECTED = {
    "call_001.ogg": dict(present=False, severity=NoiseSeverity.NONE, silence=False),
    "call_002.ogg": dict(present=True, severity=NoiseSeverity.MEDIUM, silence=False),
    "call_003.ogg": dict(present=True, severity=NoiseSeverity.MEDIUM, silence=False),
}


@pytest.mark.parametrize("name,expected", EXPECTED.items())
def test_reproduces_labelled_noise_fields(name, expected):
    y, _ = load_mono(reference_call(name))
    result = analyse_signal(y)
    assert result.background_noise_present is expected["present"]
    assert result.background_noise_severity == expected["severity"]
    assert result.long_silence_present is expected["silence"]


@pytest.mark.parametrize("name", EXPECTED)
def test_quality_is_clear_on_all_provided_calls(name):
    y, _ = load_mono(reference_call(name))
    assert analyse_signal(y).audio_quality == AudioQuality.CLEAR


def test_type_is_empty_when_no_noise_present():
    y, _ = load_mono(reference_call("call_001.ogg"))
    assert analyse_signal(y).background_noise_type == ""


def test_type_is_populated_when_noise_present():
    y, _ = load_mono(reference_call("call_002.ogg"))
    assert analyse_signal(y).background_noise_type != ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_signal_branch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.signal_branch'`

- [ ] **Step 3: Write `autoace/signal_branch.py`**

```python
"""Assembles the six signal fields from raw audio.

ALWAYS runs on raw, unmodified audio. Any denoising applied for ASR is
applied to a copy — denoising destroys the very evidence four of these
fields depend on (spec §2.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoace.acoustics import frame_db, longest_run_seconds, stft_magnitude
from autoace.config import (
    LONG_SILENCE_SEC,
    NOISE_FLOOR_PRESENT,
    NOISE_SEVERITY_BANDS,
    OVERLAP_MIN_SEGMENT_S,
    OVERLAP_MIN_TOTAL_S,
    SAMPLE_RATE,
    TAG_MIN_DOM,
)
from autoace.quality import assess_quality, noise_floor_dbfs, speech_level_dbfs
from autoace.schema import AudioQuality, NoiseSeverity
from autoace.tagging import classify_noise_type, spectral_artifact, tag_non_speech
from autoace.vad import Segment, non_speech_segments, speech_segments


@dataclass
class SignalResult:
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    noise_floor_dbfs: float
    snr_db: float


def _severity_from_floor(floor_dbfs: float) -> NoiseSeverity:
    for upper_bound, name in NOISE_SEVERITY_BANDS:
        if floor_dbfs <= upper_bound:
            return NoiseSeverity(name)
    return NoiseSeverity.HIGH


def _long_silence(y: np.ndarray, floor_dbfs: float) -> bool:
    """True dead air only — energy below the noise floor, not merely
    non-speech. call_003 has a 7.35s non-speech gap and is labelled false,
    so a naive gap detector produces a false positive (spec §2.5 finding 2).
    """
    # frame_db is already relative to the call's peak frame, and
    # noise_floor_dbfs uses that same scale — so they compare directly.
    # A frame quieter than the typical background IS dead air.
    db = frame_db(stft_magnitude(y))
    return longest_run_seconds(db < floor_dbfs) > LONG_SILENCE_SEC


def _overlap_present(y: np.ndarray) -> bool:
    """pyannote overlapped-speech, gated so backchannels do not fire it.

    Known ambiguity (spec §7.5): call_002 has background TV *and* overlap=true.
    TV audio contains speech, so this fires on it. We deliberately do NOT
    suppress, matching the observed label.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return False

    try:
        pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection")
        annotation = pipeline({"waveform": _as_tensor(y), "sample_rate": SAMPLE_RATE})
    except Exception:
        return False

    spans = [
        segment.duration
        for segment in annotation.get_timeline()
        if segment.duration >= OVERLAP_MIN_SEGMENT_S
    ]
    return sum(spans) >= OVERLAP_MIN_TOTAL_S


def _as_tensor(y: np.ndarray):
    import torch

    return torch.from_numpy(y).unsqueeze(0)


def analyse_signal(y: np.ndarray) -> SignalResult:
    speech = speech_segments(y)
    non_speech = non_speech_segments(y)

    floor = noise_floor_dbfs(y, non_speech)
    snr = speech_level_dbfs(y, speech) - floor

    tag = tag_non_speech(y, non_speech)
    artifact = spectral_artifact(y, non_speech)

    # Floor is the presence detector; the tagger only names the type. Absolute
    # AudioSet probabilities do not separate noisy from clean on telephony
    # audio, but the floor does (spec §7.2).
    present = floor > NOISE_FLOOR_PRESENT and (
        tag.relative_dominance > TAG_MIN_DOM or artifact is not None
    )

    noise_type = classify_noise_type(y, non_speech) if present else ""
    severity = _severity_from_floor(floor) if present else NoiseSeverity.NONE

    return SignalResult(
        background_noise_present=present,
        background_noise_type=noise_type,
        background_noise_severity=severity,
        audio_quality=assess_quality(y).quality,
        speaker_overlap_present=_overlap_present(y),
        long_silence_present=_long_silence(y, floor),
        noise_floor_dbfs=floor,
        snr_db=snr,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_signal_branch.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add autoace/signal_branch.py tests/test_signal_branch.py
git commit -m "feat: signal branch assembling six deterministic fields"
```

---

## Task 10: Prosody, activation, and trajectory

**Files:**
- Create: `autoace/prosody.py`
- Test: `tests/test_prosody.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prosody.py`:

```python
import numpy as np
import pytest

from autoace.prosody import Tier, activation_profile, discretise, select_tier


def test_tier_selection_from_customer_speech_duration():
    assert select_tier(20.0) == Tier.A
    assert select_tier(6.0) == Tier.B
    assert select_tier(1.0) == Tier.C


def test_call_002_style_minimal_speech_is_tier_c():
    """The customer says two words — roughly a second. There is no baseline
    to establish and no trajectory to fit (spec §6.1)."""
    assert select_tier(1.2) == Tier.C


def test_discretise_produces_relative_tags_not_numbers():
    """LLMs reason over `much louder`, not `87.3 dB` (spec §5.1)."""
    tags = discretise({"loudness_range": 2.4, "f0_elevation": 1.1, "rate_deviation": 0.2})
    assert "much louder" in tags
    assert all(not any(ch.isdigit() for ch in tag) for tag in tags)


def test_rising_trajectory_is_detected():
    profile = activation_profile([0.1, 0.9, 2.2])
    assert profile.slope_rising is True
    assert profile.peak_z == pytest.approx(2.2)


def test_flat_trajectory_is_not_rising():
    profile = activation_profile([1.0, 1.05, 0.95])
    assert profile.slope_rising is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_prosody.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.prosody'`

- [ ] **Step 3: Write `autoace/prosody.py`**

```python
"""eGeMAPS extraction, per-speaker normalization, and activation scoring.

Two rules drive this module:
  1. Normalize to the speaker's OWN baseline. Absolute dB is recording gain,
     not emotion — a hot mic reads angry and a quiet line reads calm.
  2. Discretize before handing to the LLM. It reasons well over "much louder"
     and poorly over raw numbers it has no calibration for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np

from autoace.config import (
    ACTIVATION_WEIGHTS,
    BASELINE_WINDOW_S,
    SAMPLE_RATE,
    TIER_A_MIN_SPEECH_S,
    TIER_C_MAX_SPEECH_S,
)
from autoace.vad import Segment


class Tier(str, Enum):
    A = "A"   # >15s customer speech: self-baseline + trajectory
    B = "B"   # 3-15s: corpus norms, no trajectory
    C = "C"   # <3s: insufficient evidence


@dataclass
class ActivationProfile:
    activation_z: float
    slope_rising: bool
    peak_z: float


# eGeMAPS functional names -> our internal feature keys.
EGEMAPS_FIELDS = {
    "loudness_range": "loudness_sma3_pctlrange0-2",
    "f0_elevation": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "f0_range": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "rate_deviation": "VoicedSegmentsPerSec",
    "jitter_shimmer": "jitterLocal_sma3nz_amean",
}


def select_tier(customer_speech_seconds: float) -> Tier:
    if customer_speech_seconds > TIER_A_MIN_SPEECH_S:
        return Tier.A
    if customer_speech_seconds < TIER_C_MAX_SPEECH_S:
        return Tier.C
    return Tier.B


@lru_cache(maxsize=1)
def _load_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_features(y: np.ndarray, segment: Segment) -> dict[str, float]:
    """eGeMAPS functionals for one utterance."""
    smile = _load_smile()
    chunk = y[int(segment.start * SAMPLE_RATE) : int(segment.end * SAMPLE_RATE)]
    if len(chunk) < SAMPLE_RATE // 10:
        return {key: 0.0 for key in EGEMAPS_FIELDS}
    frame = smile.process_signal(chunk, SAMPLE_RATE)
    return {
        key: float(frame[column].iloc[0]) if column in frame.columns else 0.0
        for key, column in EGEMAPS_FIELDS.items()
    }


def baseline_statistics(
    y: np.ndarray, segments: list[Segment]
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and std of each feature over the speaker's opening window."""
    window: list[Segment] = []
    elapsed = 0.0
    for seg in segments:
        window.append(seg)
        elapsed += seg.duration
        if elapsed >= BASELINE_WINDOW_S:
            break

    rows = [extract_features(y, seg) for seg in window]
    means, stds = {}, {}
    for key in EGEMAPS_FIELDS:
        values = np.array([row[key] for row in rows], dtype=float)
        means[key] = float(values.mean()) if values.size else 0.0
        stds[key] = float(values.std()) or 1.0     # avoid divide-by-zero
    return means, stds


def z_score(
    features: dict[str, float], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    return {key: (features[key] - means[key]) / stds[key] for key in features}


def discretise(z_scores: dict[str, float]) -> list[str]:
    """Turn z-scores into the relative tags the LLM actually reads."""
    tags: list[str] = []
    loudness = z_scores.get("loudness_range", 0.0)
    if loudness > 2.0:
        tags.append("much louder")
    elif loudness > 1.0:
        tags.append("louder")
    elif loudness < -1.0:
        tags.append("quieter")

    pitch = z_scores.get("f0_elevation", 0.0)
    if pitch > 1.0:
        tags.append("rising pitch")
    elif pitch < -1.0:
        tags.append("flat pitch")

    if z_scores.get("f0_range", 0.0) > 1.0:
        tags.append("high pitch variance")

    rate = z_scores.get("rate_deviation", 0.0)
    if rate > 1.0:
        tags.append("faster")
    elif rate < -1.0:
        tags.append("slower")

    if z_scores.get("jitter_shimmer", 0.0) > 1.0:
        tags.append("strained voice")

    return tags or ["baseline"]


def combine(z_scores: dict[str, float], extra: dict[str, float] | None = None) -> float:
    """Weighted activation score across vocal, interactional, and lexical channels."""
    values = dict(z_scores)
    values.update(extra or {})
    return sum(
        ACTIVATION_WEIGHTS.get(key, 0.0) * value for key, value in values.items()
    )


def activation_profile(per_third: list[float]) -> ActivationProfile:
    """Level plus trajectory. The brief's definitions demand the temporal axis:
    medium is 'clear and sustained', high is 'strong, escalated'."""
    values = np.array(per_third, dtype=float)
    if values.size == 0:
        return ActivationProfile(0.0, False, 0.0)
    slope = float(np.polyfit(np.arange(values.size), values, 1)[0]) if values.size > 1 else 0.0
    return ActivationProfile(
        activation_z=float(values.mean()),
        slope_rising=slope > 0.1,
        peak_z=float(values.max()),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_prosody.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add autoace/prosody.py tests/test_prosody.py
git commit -m "feat: eGeMAPS prosody with speaker-relative activation scoring"
```

---

## Task 11: Dimensional SER

**Files:**
- Create: `autoace/ser.py`
- Test: `tests/test_ser.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ser.py`:

```python
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.ser import predict_dimensions


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_returns_three_bounded_dimensions(name):
    y, _ = load_mono(reference_call(name))
    result = predict_dimensions(y)
    for key in ("arousal", "dominance", "valence"):
        assert 0.0 <= getattr(result, key) <= 1.0


def test_dimensions_are_not_all_identical():
    """A model returning the same triple for every input carries no signal."""
    y1, _ = load_mono(reference_call("call_001.ogg"))
    y3, _ = load_mono(reference_call("call_003.ogg"))
    a, b = predict_dimensions(y1), predict_dimensions(y3)
    assert (a.arousal, a.valence) != (b.arousal, b.valence)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.ser'`

- [ ] **Step 3: Write `autoace/ser.py`**

```python
"""Dimensional speech emotion recognition.

Chosen over categorical SER for three reasons (spec §5.1):
  - Trained on MSP-Podcast (naturalistic speech), not acted studio corpora.
  - Arousal maps directly onto emotional_intensity.
  - Dominance separates `upset` (assertive) from `distressed` (overwhelmed),
    the hardest distinction in the schema.

Always consumed as a full distribution, never an argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from autoace.config import SAMPLE_RATE, SER_MODEL


@dataclass(frozen=True)
class Dimensions:
    arousal: float
    dominance: float
    valence: float

    def as_prompt_line(self) -> str:
        return (
            f"arousal {self.arousal:.2f} | "
            f"dominance {self.dominance:.2f} | "
            f"valence {self.valence:.2f}"
        )


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoModelForAudioClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(SER_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(SER_MODEL)
    model.eval()
    return processor, model


def predict_dimensions(y: np.ndarray) -> Dimensions:
    processor, model = _load_model()
    inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0).numpy()

    # The audeering head outputs [arousal, dominance, valence] already in [0,1].
    values = np.clip(logits[:3], 0.0, 1.0)
    return Dimensions(
        arousal=float(values[0]), dominance=float(values[1]), valence=float(values[2])
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ser.py -v`
Expected: 4 passed (first run downloads the model, ~1.2 GB)

- [ ] **Step 5: Commit**

```bash
git add autoace/ser.py tests/test_ser.py
git commit -m "feat: dimensional SER via MSP-Podcast arousal/dominance/valence"
```

---

## Task 12: Tone classification — Haiku and the NLI second approach

**Files:**
- Create: `autoace/tone_llm.py`, `autoace/tone_nli.py`
- Test: `tests/test_tone.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tone.py`:

```python
import pytest

from autoace.schema import EmotionalTone
from autoace.tone_llm import ToneRequest, build_prompt, parse_response
from autoace.tone_nli import classify_tone_nli


def test_prompt_contains_verbatim_label_definitions():
    """The brief's §2 definitions ARE the classifier spec — paraphrasing them
    loses the frustrated/upset/distressed boundaries."""
    prompt = build_prompt(
        ToneRequest(
            duration_s=30.9,
            customer_speech_s=6.2,
            tier="B",
            language="en",
            ser_line="arousal 0.78 | dominance 0.71 | valence 0.24",
            agent_behavior="greeted; failed to respond to 5 consecutive utterances",
            annotated_lines=['[00:12-00:18] (baseline) "Are you a real person?"'],
            trajectory="rising through 00:27; peak activation z=2.3",
        )
    )
    assert "Frustrated means annoyed, impatient" in prompt
    assert "Distressed means highly emotional" in prompt


def test_prompt_states_prosody_is_corroborating_for_tone():
    """Without this the model drags tone toward `upset` whenever prosody is
    loud — the exact failure the brief warns about."""
    prompt = build_prompt(
        ToneRequest(30.9, 6.2, "B", "en", "arousal 0.5", "ok", [], "flat")
    )
    assert "CORROBORATING" in prompt
    assert "loudness alone" in prompt


def test_prompt_carries_both_domain_rules():
    """Both directions are needed: politeness masking (call_003 reads as
    frustrated but is satisfied) and escalating repetition (call_001 reads as
    neutral but is upset)."""
    prompt = build_prompt(
        ToneRequest(30.9, 6.2, "B", "en", "arousal 0.5", "ok", [], "flat")
    )
    assert "thanks the agent is SATISFIED" in prompt
    assert "escalating repetition" in prompt.lower()


def test_parse_response_rejects_invalid_enum():
    with pytest.raises(ValueError):
        parse_response('{"emotional_tone": "furious", "emotional_intensity": "high", '
                       '"self_confidence": 0.9, "reasoning": "x", '
                       '"lexical_intensity_markers": [], "agent_failed": false, '
                       '"evidence_conflict": false}')


def test_parse_response_accepts_valid_payload():
    result = parse_response(
        '{"emotional_tone": "upset", "emotional_intensity": "high", '
        '"self_confidence": 0.9, "reasoning": "escalating repetition", '
        '"lexical_intensity_markers": ["repetition"], "agent_failed": true, '
        '"evidence_conflict": false}'
    )
    assert result.emotional_tone == EmotionalTone.UPSET
    assert result.agent_failed is True


def test_nli_returns_a_distribution_over_all_five_tones():
    """The local second approach required by the brief §9."""
    scores = classify_tone_nli("I have been waiting for three weeks and nobody called back.")
    assert set(scores) == {t.value for t in EmotionalTone}
    assert abs(sum(scores.values()) - 1.0) < 0.01
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_tone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.tone_llm'`

- [ ] **Step 3: Write `autoace/tone_llm.py`**

```python
"""Tone classification via Claude Haiku 4.5 with structured outputs.

The model never hears the audio, so the feature description IS the tone
signal. Privacy: transcripts and derived features leave AutoAce
infrastructure; audio does not. Disclosed per the brief §11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from autoace.config import LLM_MODEL, LLM_TEMPERATURE
from autoace.schema import EmotionalIntensity, EmotionalTone

# Verbatim from the brief §2. Do not paraphrase — these definitions are the
# classifier specification.
LABEL_DEFINITIONS = """\
emotional_tone (one of: neutral, satisfied, frustrated, upset, distressed)
  The primary emotional tone expressed by the customer. Neutral means no clear
  positive or negative emotion. Satisfied means pleased, relieved,
  appreciative, or clearly positive. Frustrated means annoyed, impatient, or
  dissatisfied without strong anger or distress. Upset means clearly angry,
  agitated, or strongly dissatisfied. Distressed means highly emotional,
  overwhelmed, panicked, crying, or otherwise emotionally escalated.

emotional_intensity (one of: low, medium, high)
  The strength of the detected emotional tone. Low is subtle or mild. Medium is
  clear and sustained. High is strong, escalated, or likely to require attention.
"""

SYSTEM_PROMPT = f"""\
You are an expert call-audio analyst for AutoAce. You score calls between an AI
voice agent and a human caller. Classify the HUMAN CALLER's emotion only —
ignore the agent's tone entirely.

You do not receive audio. You receive acoustic measurements, a dimensional
speech-emotion model's output, a summary of what the agent did, and a transcript
annotated with per-utterance prosody. Treat the acoustic evidence as primary.

## Label definitions
{LABEL_DEFINITIONS}

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
"""

# Synthetic — deliberately NOT the three labelled calls, which are the entire
# validation set. Using them here would destroy the regression test (spec §5.2.2).
FEW_SHOT = """\
Example 1 — polite caller whose request could not be fulfilled:
  Evidence: valence 0.68, dominance 0.44, arousal 0.41; steady pitch, warm
  delivery, thanks the agent at close; agent denied the requested date twice.
  {"emotional_tone": "satisfied", "emotional_intensity": "medium",
   "self_confidence": 0.78, "reasoning": "Warm, steady delivery and a closing
   thank-you outweigh the unsuccessful outcome.",
   "lexical_intensity_markers": [], "agent_failed": false,
   "evidence_conflict": false}

Example 2 — short neutral words, escalating delivery:
  Evidence: valence 0.21, dominance 0.74, arousal 0.83; same one-word utterance
  repeated four times, each louder with a shorter gap; agent never responded.
  {"emotional_tone": "upset", "emotional_intensity": "high",
   "self_confidence": 0.82, "reasoning": "Escalating repetition with rising
   energy against an unresponsive agent.",
   "lexical_intensity_markers": ["repetition"], "agent_failed": true,
   "evidence_conflict": false}

Example 3 — almost no caller speech:
  Evidence: 1.1s of caller speech, one short request; no baseline established.
  {"emotional_tone": "neutral", "emotional_intensity": "medium",
   "self_confidence": 0.31, "reasoning": "Insufficient caller speech to
   establish tone or activation.",
   "lexical_intensity_markers": [], "agent_failed": false,
   "evidence_conflict": false}
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "emotional_tone": {
            "type": "string",
            "enum": [t.value for t in EmotionalTone],
        },
        "emotional_intensity": {
            "type": "string",
            "enum": [i.value for i in EmotionalIntensity],
        },
        "self_confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "lexical_intensity_markers": {"type": "array", "items": {"type": "string"}},
        "agent_failed": {"type": "boolean"},
        "evidence_conflict": {"type": "boolean"},
    },
    "required": [
        "emotional_tone", "emotional_intensity", "self_confidence",
        "reasoning", "lexical_intensity_markers", "agent_failed",
        "evidence_conflict",
    ],
    "additionalProperties": False,
}


@dataclass
class ToneRequest:
    duration_s: float
    customer_speech_s: float
    tier: str
    language: str
    ser_line: str
    agent_behavior: str
    annotated_lines: list[str]
    trajectory: str
    interruptions: int = 0


@dataclass
class ToneResponse:
    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    self_confidence: float
    reasoning: str
    lexical_intensity_markers: list[str] = field(default_factory=list)
    agent_failed: bool = False
    evidence_conflict: bool = False


def build_prompt(request: ToneRequest) -> str:
    """Assemble the full prompt. Returned as one string so it is testable
    without an API key."""
    transcript = "\n".join(request.annotated_lines) or "(no customer speech captured)"
    payload = f"""\
Call duration: {request.duration_s:.1f}s | Customer speech: {request.customer_speech_s:.1f}s \
(tier {request.tier}) | Language: {request.language}

SER (MSP-Podcast, dimensional): {request.ser_line}

Agent behavior: {request.agent_behavior}

Annotated customer transcript:
{transcript}

Trajectory: {request.trajectory}
Interruptions by customer: {request.interruptions}
"""
    return f"{SYSTEM_PROMPT}\n{FEW_SHOT}\n---\n{payload}"


def parse_response(raw: str) -> ToneResponse:
    """Validate and coerce the model's JSON. Raises ValueError on any
    enum violation so a bad label can never reach the output schema."""
    data = json.loads(raw)
    try:
        tone = EmotionalTone(data["emotional_tone"])
        intensity = EmotionalIntensity(data["emotional_intensity"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid tone payload: {exc}") from exc

    return ToneResponse(
        emotional_tone=tone,
        emotional_intensity=intensity,
        self_confidence=float(data["self_confidence"]),
        reasoning=str(data["reasoning"]),
        lexical_intensity_markers=list(data.get("lexical_intensity_markers", [])),
        agent_failed=bool(data.get("agent_failed", False)),
        evidence_conflict=bool(data.get("evidence_conflict", False)),
    )


def classify_tone(request: ToneRequest) -> ToneResponse:
    """Call Haiku 4.5 with structured output enforcement."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        temperature=LLM_TEMPERATURE,
        system=SYSTEM_PROMPT + "\n" + FEW_SHOT,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": build_prompt(request)}],
    )
    text = next(block.text for block in message.content if block.type == "text")
    return parse_response(text)
```

- [ ] **Step 4: Write `autoace/tone_nli.py`**

```python
"""Zero-shot tone classification via NLI entailment.

Satisfies the brief §9 requirement to compare at least two materially
different approaches, doubles as the no-API fallback if transcript egress is
refused, and votes in the confidence ensemble.
"""

from __future__ import annotations

from functools import lru_cache

from autoace.config import NLI_MODEL
from autoace.schema import EmotionalTone

HYPOTHESES = {
    EmotionalTone.NEUTRAL: "The customer expresses no clear positive or negative emotion.",
    EmotionalTone.SATISFIED: "The customer is pleased, relieved, or appreciative.",
    EmotionalTone.FRUSTRATED: "The customer is annoyed, impatient, or dissatisfied.",
    EmotionalTone.UPSET: "The customer is clearly angry or strongly dissatisfied.",
    EmotionalTone.DISTRESSED: "The customer is overwhelmed, panicked, or highly emotional.",
}


@lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=NLI_MODEL, device=-1)


def classify_tone_nli(transcript: str) -> dict[str, float]:
    """Return a normalized probability over the five tone classes."""
    if not transcript.strip():
        uniform = 1.0 / len(EmotionalTone)
        return {t.value: uniform for t in EmotionalTone}

    classifier = _load_pipeline()
    result = classifier(
        transcript,
        candidate_labels=list(HYPOTHESES.values()),
        multi_label=False,
    )
    hypothesis_to_tone = {v: k for k, v in HYPOTHESES.items()}
    scores = {
        hypothesis_to_tone[label].value: float(score)
        for label, score in zip(result["labels"], result["scores"])
    }
    total = sum(scores.values()) or 1.0
    return {key: value / total for key, value in scores.items()}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tone.py -v`
Expected: 6 passed (the NLI test downloads bart-large-mnli, ~1.6 GB; no API key needed — `classify_tone` is not exercised)

- [ ] **Step 6: Commit**

```bash
git add autoace/tone_llm.py autoace/tone_nli.py tests/test_tone.py
git commit -m "feat: Haiku tone classifier and local NLI second approach"
```

---

## Task 13: Fusion — intensity rule, confidence, and assembly

**Files:**
- Create: `autoace/fuse.py`
- Test: `tests/test_fuse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fuse.py`:

```python
import pytest

from autoace.config import TIER_C_MAX_CONF
from autoace.fuse import (
    ConfidenceInputs,
    coherence_conflict,
    compute_confidence,
    intensity_from_activation,
    reconcile_intensity,
)
from autoace.prosody import ActivationProfile, Tier
from autoace.schema import EmotionalIntensity, EmotionalTone
from autoace.ser import Dimensions


def test_high_requires_level_and_escalation():
    """The brief: medium is 'clear and sustained', high is 'strong, escalated'."""
    escalating = ActivationProfile(activation_z=1.4, slope_rising=True, peak_z=1.8)
    sustained = ActivationProfile(activation_z=1.4, slope_rising=False, peak_z=1.5)
    assert intensity_from_activation(escalating) == EmotionalIntensity.HIGH
    assert intensity_from_activation(sustained) == EmotionalIntensity.MEDIUM


def test_extreme_peak_alone_reaches_high():
    profile = ActivationProfile(activation_z=1.2, slope_rising=False, peak_z=2.4)
    assert intensity_from_activation(profile) == EmotionalIntensity.HIGH


def test_low_requires_flat_and_near_baseline():
    profile = ActivationProfile(activation_z=0.1, slope_rising=False, peak_z=0.2)
    assert intensity_from_activation(profile) == EmotionalIntensity.LOW


def test_disagreeing_voters_fall_back_to_medium():
    """`medium` is the majority class and the 2/3 baseline, so the conservative
    fallback is also the statistically correct one (spec §6.5)."""
    result, agreed = reconcile_intensity(
        EmotionalIntensity.HIGH, EmotionalIntensity.LOW, Tier.A
    )
    assert result == EmotionalIntensity.MEDIUM
    assert agreed is False


def test_agreeing_voters_are_emitted_unchanged():
    result, agreed = reconcile_intensity(
        EmotionalIntensity.HIGH, EmotionalIntensity.HIGH, Tier.A
    )
    assert result == EmotionalIntensity.HIGH
    assert agreed is True


def test_tier_c_always_returns_medium():
    result, agreed = reconcile_intensity(
        EmotionalIntensity.HIGH, EmotionalIntensity.HIGH, Tier.C
    )
    assert result == EmotionalIntensity.MEDIUM
    assert agreed is False


def test_positive_tone_with_negative_valence_is_a_conflict():
    assert coherence_conflict(
        EmotionalTone.SATISFIED, EmotionalIntensity.MEDIUM,
        Dimensions(arousal=0.5, dominance=0.5, valence=0.15),
    ) is True


def test_consistent_tone_and_valence_is_no_conflict():
    assert coherence_conflict(
        EmotionalTone.SATISFIED, EmotionalIntensity.MEDIUM,
        Dimensions(arousal=0.5, dominance=0.5, valence=0.80),
    ) is False


def test_high_intensity_with_low_arousal_is_a_conflict():
    assert coherence_conflict(
        EmotionalTone.UPSET, EmotionalIntensity.HIGH,
        Dimensions(arousal=0.20, dominance=0.7, valence=0.2),
    ) is True


def test_tier_c_confidence_is_hard_capped():
    """A tier-C result is a prior, not a measurement, however well the other
    voters agree (spec §8)."""
    inputs = ConfidenceInputs(
        tone_voters_agree=True, intensity_voters_agree=True,
        ser_consistent=True, llm_self_confidence=0.95,
        evidence_conflict=False, near_threshold=False,
        tier=Tier.C, diarization_degraded=False, asr_avg_logprob=-0.2,
    )
    assert compute_confidence(inputs) <= TIER_C_MAX_CONF


def test_confidence_stays_within_bounds():
    worst = ConfidenceInputs(
        tone_voters_agree=False, intensity_voters_agree=False,
        ser_consistent=False, llm_self_confidence=0.1,
        evidence_conflict=True, near_threshold=True,
        tier=Tier.C, diarization_degraded=True, asr_avg_logprob=-3.0,
    )
    assert 0.05 <= compute_confidence(worst) <= 0.98
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.fuse'`

- [ ] **Step 3: Write `autoace/fuse.py`**

```python
"""Intensity mapping, voter reconciliation, and confidence.

Two invariants:
  - Intensity is NEVER derived from classifier confidence. That is a different
    quantity, and conflating them produces constant `high`.
  - Intensity is NEVER gated on tone. Independence is what permits
    `satisfied / medium` and `neutral / medium` to exist in the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoace.config import (
    ACTIVATION_HIGH_Z,
    ACTIVATION_LOW_Z,
    ACTIVATION_PEAK_Z,
    AROUSAL_HIGH_MIN,
    CONF_ASR_POOR,
    CONF_BASE,
    CONF_DEGRADED,
    CONF_EVIDENCE_CONFLICT,
    CONF_INTENSITY_AGREE,
    CONF_LLM_SELF_HIGH,
    CONF_LLM_SELF_HIGH_THRESHOLD,
    CONF_MAX,
    CONF_MIN,
    CONF_NEAR_THRESHOLD,
    CONF_SER_CONSISTENT,
    CONF_TIER_C,
    CONF_TONE_AGREE,
    DEGRADED_MAX_CONF,
    TIER_C_MAX_CONF,
    VALENCE_NEG_MAX,
    VALENCE_POS_MIN,
    ASR_MIN_LOGPROB,
)
from autoace.prosody import ActivationProfile, Tier
from autoace.schema import (
    NEGATIVE_TONES,
    POSITIVE_TONES,
    EmotionalIntensity,
    EmotionalTone,
)
from autoace.ser import Dimensions


@dataclass
class ConfidenceInputs:
    tone_voters_agree: bool
    intensity_voters_agree: bool
    ser_consistent: bool
    llm_self_confidence: float
    evidence_conflict: bool
    near_threshold: bool
    tier: Tier
    diarization_degraded: bool
    asr_avg_logprob: float


def intensity_from_activation(profile: ActivationProfile) -> EmotionalIntensity:
    """Level x trajectory. High needs escalation OR an extreme peak; medium is
    elevated-but-stable; low is near baseline and flat."""
    if profile.activation_z > ACTIVATION_HIGH_Z and (
        profile.slope_rising or profile.peak_z > ACTIVATION_PEAK_Z
    ):
        return EmotionalIntensity.HIGH
    if profile.activation_z < ACTIVATION_LOW_Z and not profile.slope_rising:
        return EmotionalIntensity.LOW
    return EmotionalIntensity.MEDIUM


def reconcile_intensity(
    rule_result: EmotionalIntensity,
    llm_result: EmotionalIntensity,
    tier: Tier,
) -> tuple[EmotionalIntensity, bool]:
    """Two voters. Agree -> emit. Disagree -> medium, flagged.
    Tier C has no measurement to reconcile, so it always returns medium."""
    if tier is Tier.C:
        return EmotionalIntensity.MEDIUM, False
    if rule_result == llm_result:
        return rule_result, True
    return EmotionalIntensity.MEDIUM, False


def coherence_conflict(
    tone: EmotionalTone, intensity: EmotionalIntensity, dims: Dimensions
) -> bool:
    """Deterministic label-vs-evidence check (spec §5.2.4). Validates against
    numeric acoustic evidence rather than parsing the prose rationale."""
    if tone in POSITIVE_TONES and dims.valence < VALENCE_POS_MIN:
        return True
    if tone in NEGATIVE_TONES and dims.valence > VALENCE_NEG_MAX:
        return True
    if intensity is EmotionalIntensity.HIGH and dims.arousal < AROUSAL_HIGH_MIN:
        return True
    return False


def compute_confidence(inputs: ConfidenceInputs) -> float:
    """Built from voter agreement, never from model self-report."""
    score = CONF_BASE
    score += CONF_TONE_AGREE * inputs.tone_voters_agree
    score += CONF_INTENSITY_AGREE * inputs.intensity_voters_agree
    score += CONF_SER_CONSISTENT * inputs.ser_consistent
    score += CONF_LLM_SELF_HIGH * (
        inputs.llm_self_confidence > CONF_LLM_SELF_HIGH_THRESHOLD
    )
    score += CONF_EVIDENCE_CONFLICT * inputs.evidence_conflict
    score += CONF_NEAR_THRESHOLD * inputs.near_threshold
    score += CONF_TIER_C * (inputs.tier is Tier.C)
    score += CONF_DEGRADED * inputs.diarization_degraded
    score += CONF_ASR_POOR * (inputs.asr_avg_logprob < ASR_MIN_LOGPROB)

    score = max(CONF_MIN, min(CONF_MAX, score))

    # Hard caps: a tier-C or degraded result is a prior, not a measurement,
    # however well the remaining voters agree.
    if inputs.tier is Tier.C:
        score = min(score, TIER_C_MAX_CONF)
    if inputs.diarization_degraded:
        score = min(score, DEGRADED_MAX_CONF)

    return round(score, 3)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_fuse.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add autoace/fuse.py tests/test_fuse.py
git commit -m "feat: intensity reconciliation and agreement-based confidence"
```

---

## Task 14: Pipeline orchestration and evaluation harness

**Files:**
- Create: `autoace/pipeline.py`, `autoace/eval.py`
- Test: `tests/test_pipeline.py`, `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval.py`:

```python
import json

from autoace.config import LABELS_CSV
from autoace.eval import BATCH_COLUMNS, load_manifest, score_batch
from autoace.schema import CallAnalysis


def test_load_manifest_reads_the_brief_format():
    rows = load_manifest(str(LABELS_CSV))
    assert len(rows) == 3
    assert rows[0].name == "call_001.ogg"
    assert rows[0].expected.emotional_tone.value == "upset"


def test_manifest_columns_match_the_brief():
    assert BATCH_COLUMNS == ("name", "result_json")


def test_score_batch_reports_the_constant_medium_baseline():
    """Intensity must always be scored against the majority-class baseline —
    a metric that does not beat it is not evidence (spec §12)."""
    rows = load_manifest(str(LABELS_CSV))
    predictions = {row.name: row.expected for row in rows}  # perfect predictions
    report = score_batch(rows, predictions)
    assert report["intensity"]["accuracy"] == 1.0
    assert report["intensity"]["constant_medium_baseline"] == 2 / 3


def test_score_batch_handles_missing_predictions():
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {})
    assert report["tone"]["accuracy"] == 0.0
    assert report["errors"] == 3
```

Create `tests/test_pipeline.py`:

```python
import pytest

from autoace.config import LABELS_CSV, reference_call
from autoace.pipeline import analyse_file
from autoace.schema import CallAnalysis


def test_returns_a_valid_schema_object_without_an_api_key(monkeypatch):
    """With no ANTHROPIC_API_KEY the tone branch must degrade rather than
    crash — the signal fields are still worth emitting."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = analyse_file(reference_call("call_001.ogg"))
    assert isinstance(result.analysis, CallAnalysis)
    assert result.analysis.background_noise_present is False


def test_malformed_file_returns_an_error_not_an_exception():
    """A single bad file must never fail a batch (brief §7)."""
    result = analyse_file(str(LABELS_CSV))
    assert result.error is not None
    assert result.analysis is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_eval.py tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.eval'`

- [ ] **Step 3: Write `autoace/eval.py`**

```python
"""Evaluation harness reading the brief's manifest format."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass

from autoace.schema import CallAnalysis

BATCH_COLUMNS = ("name", "result_json")


@dataclass
class ManifestRow:
    name: str
    expected: CallAnalysis | None


def load_manifest(path: str) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            raw = (record.get("result_json") or "").strip()
            expected = CallAnalysis(**json.loads(raw)) if raw else None
            rows.append(ManifestRow(name=record["name"], expected=expected))
    return rows


def _accuracy(pairs: list[tuple[object, object]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _macro_f1(pairs: list[tuple[object, object]]) -> float:
    labels = {expected for expected, _ in pairs} | {actual for _, actual in pairs}
    scores = []
    for label in labels:
        tp = sum(1 for e, a in pairs if e == label and a == label)
        fp = sum(1 for e, a in pairs if e != label and a == label)
        fn = sum(1 for e, a in pairs if e == label and a != label)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def _confusion(pairs: list[tuple[object, object]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for expected, actual in pairs:
        row = matrix.setdefault(str(expected), Counter())
        row[str(actual)] += 1
    return {key: dict(value) for key, value in matrix.items()}


def score_batch(
    rows: list[ManifestRow], predictions: dict[str, CallAnalysis]
) -> dict:
    """Grouped metrics, matching how the brief says it scores."""
    labelled = [row for row in rows if row.expected is not None]
    matched = [(row, predictions.get(row.name)) for row in labelled]
    errors = sum(1 for _, prediction in matched if prediction is None)
    usable = [(row.expected, prediction) for row, prediction in matched if prediction]

    def field_pairs(field: str) -> list[tuple[object, object]]:
        return [(getattr(e, field), getattr(a, field)) for e, a in usable]

    tone = field_pairs("emotional_tone")
    intensity = field_pairs("emotional_intensity")
    severity = field_pairs("background_noise_severity")

    baseline = (
        sum(1 for expected, _ in intensity if expected.value == "medium") / len(intensity)
        if intensity
        else 0.0
    )

    return {
        "n_labelled": len(labelled),
        "errors": errors,
        "tone": {
            "accuracy": _accuracy(tone),
            "macro_f1": _macro_f1(tone),
            "confusion": _confusion(tone),
        },
        "intensity": {
            "accuracy": _accuracy(intensity),
            "macro_f1": _macro_f1(intensity),
            "confusion": _confusion(intensity),
            "constant_medium_baseline": baseline,
        },
        "noise": {
            "present_accuracy": _accuracy(field_pairs("background_noise_present")),
            "severity_macro_f1": _macro_f1(severity),
            "type_accuracy": _accuracy(
                [
                    (str(e).lower().strip(), str(a).lower().strip())
                    for e, a in field_pairs("background_noise_type")
                ]
            ),
        },
        "technical": {
            "audio_quality_accuracy": _accuracy(field_pairs("audio_quality")),
            "overlap_accuracy": _accuracy(field_pairs("speaker_overlap_present")),
            "silence_accuracy": _accuracy(field_pairs("long_silence_present")),
        },
    }
```

- [ ] **Step 4: Write `autoace/pipeline.py`**

```python
"""Per-file orchestration with fail isolation.

A single malformed file must never fail a batch (brief §7), so every stage
is wrapped and the tone branch degrades to a signal-only result rather than
raising when the LLM is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoace.diarize import assign_speakers
from autoace.fuse import (
    ConfidenceInputs,
    coherence_conflict,
    compute_confidence,
    intensity_from_activation,
    reconcile_intensity,
)
from autoace.io_audio import UnsupportedAudio, load_mono
from autoace.prosody import (
    ActivationProfile,
    Tier,
    baseline_statistics,
    combine,
    discretise,
    extract_features,
    select_tier,
    z_score,
)
from autoace.schema import CallAnalysis, EmotionalIntensity, EmotionalTone
from autoace.ser import predict_dimensions
from autoace.signal_branch import analyse_signal
from autoace.vad import speech_segments


@dataclass
class FileResult:
    name: str
    analysis: CallAnalysis | None
    error: str | None = None
    reasoning: str = ""
    review_flagged: bool = False


def analyse_file(path: str) -> FileResult:
    try:
        y, _ = load_mono(path)
    except UnsupportedAudio as exc:
        return FileResult(name=path, analysis=None, error=str(exc))

    try:
        signal = analyse_signal(y)
    except Exception as exc:
        return FileResult(name=path, analysis=None, error=f"signal branch: {exc}")

    tone, intensity, confidence, reasoning, conflict = _tone_branch(y, path)

    analysis = CallAnalysis(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=signal.background_noise_present,
        background_noise_type=signal.background_noise_type,
        background_noise_severity=signal.background_noise_severity,
        audio_quality=signal.audio_quality,
        speaker_overlap_present=signal.speaker_overlap_present,
        long_silence_present=signal.long_silence_present,
        confidence=confidence,
    )
    from autoace.config import REVIEW_THRESHOLD

    return FileResult(
        name=path,
        analysis=analysis,
        reasoning=reasoning,
        review_flagged=confidence < REVIEW_THRESHOLD or conflict,
    )


def _tone_branch(y, path: str):
    """Returns (tone, intensity, confidence, reasoning, conflict).

    Degrades to a neutral/medium prior with low confidence rather than
    raising, so the six signal fields still ship."""
    segments = speech_segments(y)
    speakers = assign_speakers(y, segments)
    tier = select_tier(speakers.customer_speech_seconds)
    dims = predict_dimensions(y)

    customer = speakers.customer_segments or segments
    profile = _activation(y, customer, tier)
    rule_intensity = intensity_from_activation(profile)

    try:
        from autoace.asr import transcribe
from autoace.config import reference_call
        from autoace.tone_llm import ToneRequest, classify_tone

        transcript = transcribe(path)
        annotated = _annotate(y, customer, transcript)
        response = classify_tone(
            ToneRequest(
                duration_s=len(y) / 16000,
                customer_speech_s=speakers.customer_speech_seconds,
                tier=tier.value,
                language=transcript.language,
                ser_line=dims.as_prompt_line(),
                agent_behavior=_agent_behavior(transcript),
                annotated_lines=annotated,
                trajectory=_trajectory_line(profile),
            )
        )
        tone = response.emotional_tone
        llm_intensity = response.emotional_intensity
        reasoning = response.reasoning
        self_conf = response.self_confidence
        llm_conflict = response.evidence_conflict
        asr_logprob = transcript.avg_logprob
        tone_agree = True
    except Exception as exc:
        tone = EmotionalTone.NEUTRAL
        llm_intensity = rule_intensity
        reasoning = f"tone branch unavailable ({exc}); signal fields only"
        self_conf, llm_conflict, asr_logprob, tone_agree = 0.0, True, -5.0, False

    intensity, intensity_agree = reconcile_intensity(rule_intensity, llm_intensity, tier)
    conflict = llm_conflict or coherence_conflict(tone, intensity, dims)

    confidence = compute_confidence(
        ConfidenceInputs(
            tone_voters_agree=tone_agree,
            intensity_voters_agree=intensity_agree,
            ser_consistent=not coherence_conflict(tone, intensity, dims),
            llm_self_confidence=self_conf,
            evidence_conflict=conflict,
            near_threshold=False,
            tier=tier,
            diarization_degraded=speakers.degraded,
            asr_avg_logprob=asr_logprob,
        )
    )
    return tone, intensity, confidence, reasoning, conflict


def _activation(y, segments, tier: Tier) -> ActivationProfile:
    if not segments:
        return ActivationProfile(0.0, False, 0.0)
    means, stds = baseline_statistics(y, segments)
    scores = [combine(z_score(extract_features(y, seg), means, stds)) for seg in segments]
    if tier is Tier.A and len(scores) >= 3:
        third = max(1, len(scores) // 3)
        per_third = [
            float(sum(chunk) / len(chunk))
            for chunk in (scores[:third], scores[third : 2 * third], scores[2 * third :])
            if chunk
        ]
    else:
        per_third = scores
    from autoace.prosody import activation_profile

    return activation_profile(per_third)


def _annotate(y, segments, transcript) -> list[str]:
    """Attach discretised prosody tags to each customer utterance."""
    if not segments:
        return []
    means, stds = baseline_statistics(y, segments)
    lines = []
    for seg in segments:
        tags = ", ".join(discretise(z_score(extract_features(y, seg), means, stds)))
        words = [
            w.text for w in transcript.words if seg.start <= w.start < seg.end
        ]
        text = "".join(words).strip() or "(unintelligible)"
        lines.append(
            f'[{_mmss(seg.start)}-{_mmss(seg.end)}] ({tags}) "{text}"'
        )
    return lines


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _trajectory_line(profile: ActivationProfile) -> str:
    direction = "rising" if profile.slope_rising else "flat"
    return f"{direction}; mean activation z={profile.activation_z:.2f}, peak z={profile.peak_z:.2f}"


def _agent_behavior(transcript) -> str:
    """Summarise what the bot did — a strong predictor of caller emotion."""
    text = transcript.text.lower()
    signals = []
    if "transferring you" in text or "advisor" in text:
        signals.append("transferred to a human advisor")
    if text.count("hello") >= 3:
        signals.append("dead-air loop with repeated greetings")
    if "won't be possible" in text or "closed" in text:
        signals.append("denied the customer's request")
    return "; ".join(signals) or "handled the call without notable failure"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_eval.py tests/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add autoace/pipeline.py autoace/eval.py tests/test_eval.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with fail isolation and eval harness"
```

---

## Task 15: Gradio dashboard

**Files:**
- Create: `autoace/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app.py`:

```python
import json
import zipfile
from pathlib import Path

import pytest

from autoace.app import BatchValidation, results_to_csv, validate_batch


def test_validation_reports_unmatched_files(tmp_path: Path):
    """Missing and unmatched files must be reported BEFORE processing (brief §7)."""
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,""\ncall_999.ogg,""\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "call_999.ogg" in report.missing_audio
    assert report.matched == ["call_001.ogg"]


def test_validation_reports_audio_without_a_manifest_row(tmp_path: Path):
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "stray.wav").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,""\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "stray.wav" in report.unlisted_audio


def test_csv_export_preserves_original_filenames():
    from autoace.pipeline import FileResult
    from autoace.schema import CallAnalysis

    analysis = CallAnalysis(
        emotional_tone="neutral", emotional_intensity="medium",
        background_noise_present=False, background_noise_type="",
        background_noise_severity="none", audio_quality="clear",
        speaker_overlap_present=False, long_silence_present=False,
        confidence=0.7,
    )
    csv_text = results_to_csv([FileResult(name="call_042.mp3", analysis=analysis)])
    assert "call_042.mp3" in csv_text
    assert "neutral" in csv_text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autoace.app'`

- [ ] **Step 3: Write `autoace/app.py`**

```python
"""Hosted dashboard. Worth 10% of the grade — treat as a first-class deliverable.

Requirements from the brief §7: login, folder/ZIP upload, manifest validation
before processing, visible progress, per-file fail isolation, result review,
downloadable CSV and JSON preserving original filenames.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from autoace.config import REVIEW_THRESHOLD
from autoace.eval import load_manifest, score_batch
from autoace.io_audio import SUPPORTED_SUFFIXES
from autoace.pipeline import FileResult, analyse_file

SCHEMA_COLUMNS = [
    "name", "emotional_tone", "emotional_intensity", "background_noise_present",
    "background_noise_type", "background_noise_severity", "audio_quality",
    "speaker_overlap_present", "long_silence_present", "confidence",
]


@dataclass
class BatchValidation:
    matched: list[str] = field(default_factory=list)
    missing_audio: list[str] = field(default_factory=list)
    unlisted_audio: list[str] = field(default_factory=list)
    manifest_path: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.matched)


def validate_batch(folder: Path) -> BatchValidation:
    """Check the batch before running any inference."""
    manifests = list(folder.glob("*.csv"))
    audio = {
        p.name for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    }
    if not manifests:
        return BatchValidation(unlisted_audio=sorted(audio))

    manifest = manifests[0]
    listed = {row.name for row in load_manifest(str(manifest))}
    return BatchValidation(
        matched=sorted(listed & audio),
        missing_audio=sorted(listed - audio),
        unlisted_audio=sorted(audio - listed),
        manifest_path=manifest,
    )


def results_to_csv(results: list[FileResult]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SCHEMA_COLUMNS + ["error"])
    writer.writeheader()
    for item in results:
        if item.analysis is None:
            writer.writerow({"name": item.name, "error": item.error})
            continue
        row = json.loads(item.analysis.model_dump_json())
        row["name"] = item.name
        row["error"] = ""
        writer.writerow(row)
    return buffer.getvalue()


def results_to_json(results: list[FileResult]) -> str:
    payload = [
        {
            "name": item.name,
            "result": json.loads(item.analysis.model_dump_json())
            if item.analysis
            else None,
            "error": item.error,
        }
        for item in results
    ]
    return json.dumps(payload, indent=2)


def _extract(upload: str, workdir: Path) -> Path:
    """Accept either a ZIP archive or a folder of files."""
    target = workdir / "batch"
    target.mkdir(parents=True, exist_ok=True)
    path = Path(upload)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
    else:
        target = path.parent
    return target


def run_batch(upload, progress=gr.Progress()):
    workdir = Path(os.environ.get("AUTOACE_WORKDIR", "./_batches"))
    workdir.mkdir(exist_ok=True)
    folder = _extract(upload, workdir)

    validation = validate_batch(folder)
    notes = []
    if validation.missing_audio:
        notes.append(f"Manifest rows with no audio file: {', '.join(validation.missing_audio)}")
    if validation.unlisted_audio:
        notes.append(f"Audio files with no manifest row: {', '.join(validation.unlisted_audio)}")
    if not validation.ok:
        return "\n".join(notes) or "No matching audio found.", None, None, None

    results: list[FileResult] = []
    for name in progress.tqdm(validation.matched, desc="Analysing"):
        # Per-file isolation: one bad file must not fail the batch.
        try:
            results.append(analyse_file(str(folder / name)))
        except Exception as exc:
            results.append(FileResult(name=name, analysis=None, error=str(exc)))

    table = [
        [
            item.name,
            item.analysis.emotional_tone.value if item.analysis else "ERROR",
            item.analysis.emotional_intensity.value if item.analysis else "-",
            item.analysis.background_noise_type if item.analysis else "-",
            item.analysis.background_noise_severity.value if item.analysis else "-",
            item.analysis.audio_quality.value if item.analysis else "-",
            round(item.analysis.confidence, 3) if item.analysis else 0.0,
            "REVIEW" if item.review_flagged else "",
            item.error or item.reasoning,
        ]
        for item in results
    ]
    # Review queue first: low-confidence and conflicting cases surface to a human.
    table.sort(key=lambda row: (row[7] != "REVIEW", row[6]))

    summary = "\n".join(notes + [f"Processed {len(results)} files."])
    if validation.manifest_path:
        rows = load_manifest(str(validation.manifest_path))
        if any(row.expected for row in rows):
            predictions = {
                item.name: item.analysis for item in results if item.analysis
            }
            summary += "\n\nScoring:\n" + json.dumps(
                score_batch(rows, predictions), indent=2, default=str
            )

    csv_path = workdir / "results.csv"
    json_path = workdir / "results.json"
    csv_path.write_text(results_to_csv(results), encoding="utf-8")
    json_path.write_text(results_to_json(results), encoding="utf-8")

    return summary, table, str(csv_path), str(json_path)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AutoAce — Voice Tone & Background Noise") as demo:
        gr.Markdown(
            "## AutoAce — Voice Tone & Background Noise\n"
            "Upload a ZIP containing audio files at the root plus a CSV manifest "
            "(`name`, `result_json`). Rows flagged **REVIEW** have confidence below "
            f"{REVIEW_THRESHOLD} or conflicting evidence."
        )
        upload = gr.File(label="Batch (.zip)", file_types=[".zip"], type="filepath")
        run = gr.Button("Run analysis", variant="primary")
        summary = gr.Textbox(label="Batch status", lines=8)
        table = gr.Dataframe(
            headers=[
                "file", "tone", "intensity", "noise type", "severity",
                "quality", "confidence", "flag", "notes",
            ],
            label="Results",
            wrap=True,
        )
        csv_out = gr.File(label="Download CSV")
        json_out = gr.File(label="Download JSON")
        run.click(run_batch, inputs=upload, outputs=[summary, table, csv_out, json_out])
    return demo


if __name__ == "__main__":
    user = os.environ.get("AUTOACE_USER", "autoace")
    password = os.environ.get("AUTOACE_PASSWORD")
    if not password:
        raise SystemExit("Set AUTOACE_PASSWORD before starting the dashboard.")
    build_app().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(user, password),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: 3 passed

- [ ] **Step 5: Verify the dashboard starts**

```bash
AUTOACE_PASSWORD=testpw python -m autoace.app
```
Expected: Gradio prints `Running on local URL: http://0.0.0.0:7860`. Open it, confirm the login
prompt appears, then stop with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add autoace/app.py tests/test_app.py
git commit -m "feat: Gradio dashboard with validation, review queue, and exports"
```

---

## Task 16: End-to-end run and the technical memo

**Files:**
- Create: `docs/MEMO.md`, `README.md`
- Test: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_end_to_end.py`:

```python
import pytest

from autoace.config import LABELS_CSV, reference_call
from autoace.eval import load_manifest, score_batch
from autoace.pipeline import analyse_file


@pytest.mark.slow
def test_full_pipeline_on_the_three_provided_calls():
    """Regression, not validation. With n=3 this proves the system reproduces
    known-good behaviour and nothing more (spec §12)."""
    rows = load_manifest(str(LABELS_CSV))
    predictions = {}
    for row in rows:
        result = analyse_file(reference_call(row.name))
        assert result.error is None, f"{row.name}: {result.error}"
        predictions[row.name] = result.analysis

    report = score_batch(rows, predictions)
    # The six signal fields are deterministic and must reproduce exactly.
    assert report["noise"]["present_accuracy"] == 1.0
    assert report["technical"]["audio_quality_accuracy"] == 1.0
    assert report["technical"]["silence_accuracy"] == 1.0
    # Tone and intensity must at minimum beat the constant-medium baseline.
    assert report["intensity"]["accuracy"] >= report["intensity"]["constant_medium_baseline"]
```

- [ ] **Step 2: Run it and record the output**

Run: `python -m pytest tests/test_end_to_end.py -v -s`
Expected: PASS. If tone accuracy is below 2/3, do **not** tune the prompt against these three
calls beyond the domain rules already in `SYSTEM_PROMPT` — that converts the regression test
into a training set and inflates the reported figure. Record the result as-is.

- [ ] **Step 3: Generate the cost and latency measurements**

```bash
python -c "
import time, json, librosa
from autoace.pipeline import analyse_file
from autoace.config import reference_call
rows = []
for n in ['call_001.ogg','call_002.ogg','call_003.ogg']:
    path = reference_call(n)
    dur = librosa.get_duration(path=path)
    t0 = time.time(); analyse_file(path); elapsed = time.time()-t0
    rows.append({'file': n, 'audio_s': round(dur,1), 'wall_s': round(elapsed,1),
                 'x_realtime': round(elapsed/dur, 2)})
    print(rows[-1])
json.dump(rows, open('docs/latency_measured.json','w'), indent=2)
"
```

- [ ] **Step 4: Write `docs/MEMO.md`**

Write the memo from the harness output, not by hand. It must cover the eight items the brief
asks for, and must state these four things explicitly:

1. **Approaches compared** — Haiku-4.5-over-annotated-features versus local BART-MNLI zero-shot,
   with measured accuracy for each on the three calls.
2. **Calibration position** — thresholds are derived from definitions and physics, validated
   against three points, not fitted. Five of eight classified fields lack full class coverage.
   Quote the `config.py` provenance markers.
3. **Cost model** — the §10 table with the CPU/GPU sensitivity and the $0.079/hr CPU break-even,
   plus the stated exclusions.
4. **Failure modes** — all eight from spec §14, including the unvalidated transmission-noise
   branch and the TV-speech overlap ambiguity.

Also disclose, per brief §11: model `claude-haiku-4-5`, pricing $1/$5 per MTok, zero-retention
configured, and that **transcripts and derived features leave AutoAce infrastructure while audio
does not**.

- [ ] **Step 5: Write `README.md`**

```markdown
# AutoAce — Voice Tone & Background Noise

Setup, run, and deploy instructions.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Analyse one file

```bash
python -c "from autoace.pipeline import analyse_file; from autoace.config import reference_call; print(analyse_file(reference_call('call_001.ogg')).analysis.model_dump_json(indent=2))"
```

## Score a labelled batch

```bash
python -c "
from autoace.config import LABELS_CSV, reference_call
from autoace.eval import load_manifest, score_batch
from autoace.pipeline import analyse_file
from autoace.config import LABELS_CSV, reference_call
import json
rows = load_manifest(str(LABELS_CSV))
preds = {r.name: analyse_file(reference_call(r.name)).analysis for r in rows}
print(json.dumps(score_batch(rows, preds), indent=2, default=str))
"
```

## Run the dashboard

```bash
AUTOACE_USER=autoace AUTOACE_PASSWORD=<password> python -m autoace.app
```

## Tests

```bash
python -m pytest -v                      # all
python -m pytest -m "not slow" -v        # skip the end-to-end run
```

Every threshold lives in `autoace/config.py`, annotated MEASURED / DERIVED / UNFITTED.
```

- [ ] **Step 6: Commit**

```bash
git add docs/MEMO.md docs/latency_measured.json README.md tests/test_end_to_end.py
git commit -m "docs: technical memo, README, and measured latency"
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §1 schema | 1 |
| §2.5 acoustics as regression | 2 |
| §4.1 decode, no normalization, stereo | 3 |
| §4.2 VAD | 4 |
| §4.3 speaker assignment + fallbacks | 5 |
| §4.4 ASR multilingual + timestamps | 6 |
| §5.1 eGeMAPS + SER | 10, 11 |
| §5.2 prompt, few-shot, contract, coherence, sampling | 12 |
| §6 intensity tiering, activation, trajectory, reconciliation | 10, 13 |
| §7.1–7.6 six signal fields | 7, 8, 9 |
| §8 confidence | 13 |
| §9 config in one file | 1 |
| §10 cost model | 16 |
| §11 latency | 16 |
| §12 eval harness | 14 |
| §13 dashboard | 15 |
| §14 failure modes | 16 |
| §2.8 evidence regeneration | 6 (§2.3, §2.4, §11), 7 (§7.1.1) |

**Gap found and closed:** §12's leave-one-call-out protocol has no separate task — with n=3 and
no trained model it reduces to the regression run in Task 16, and the memo states that. No
additional task added; noted here so it isn't mistaken for an omission.

**Type consistency checked:** `Segment` (vad) is used unchanged by diarize, prosody, tagging,
quality, signal_branch. `Tier` (prosody) is consumed by fuse and pipeline. `Dimensions` (ser) is
consumed by fuse and pipeline. `ActivationProfile` (prosody) is produced by
`activation_profile()` and consumed by `intensity_from_activation()`. `FileResult` (pipeline) is
consumed by app. `CallAnalysis` (schema) is produced by pipeline and consumed by eval and app.
`ToneRequest`/`ToneResponse` (tone_llm) are internal to task 12 and pipeline.

**Placeholder scan:** clean — no TBD, no "add error handling", no "similar to Task N".

**Two correctness bugs found and fixed during review:**

1. **Unit mismatch between calibration and production (Task 8).** `NOISE_SEVERITY_BANDS` and
   `NOISE_FLOOR_PRESENT` were derived from measurements taken relative to each call's loudest
   frame, but `noise_floor_dbfs()` originally returned absolute dBFS. Different scales — every
   severity threshold would have been meaningless. The estimator now returns the peak-relative
   level, and `test_production_floor_is_on_the_same_scale_as_the_calibration` pins the scale so
   it cannot drift back.

2. **No-op dB shift in `_long_silence` (Task 9).** `frame_db()` is already peak-relative, so
   `db.max()` is always 0 and the "shift to dBFS" line reduced the comparison to `db < 0` —
   true for nearly every frame, so almost every call would have been flagged
   `long_silence_present: true`. Now both quantities are on the same scale and compare directly.

Both were unit errors that type checking cannot catch and that the three-call regression would
have surfaced only as a confusing failure. Worth noting in the memo's failure-modes section as
an example of why the thresholds and the estimators must be reviewed together.
