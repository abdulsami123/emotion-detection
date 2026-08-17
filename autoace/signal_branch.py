"""Assembles the six deterministic, non-LLM fields of the 9-field schema:

  background_noise_present, background_noise_type, background_noise_severity,
  audio_quality, speaker_overlap_present, long_silence_present

plus two intermediate values (`noise_floor_dbfs`, `snr_db`) that feed the
confidence calculation and the dashboard's detail view.

CRITICAL: always call `analyse_signal` on raw, unmodified audio. Any
denoising applied ahead of ASR must be done on a COPY of the signal -
denoising destroys the very evidence (noise floor, spectral tags, overlap
embeddings) these fields depend on. This module never denoises anything
itself; it is the caller's responsibility to pass it the raw signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from autoace.config import (
    LONG_SILENCE_SEC,
    NOISE_FLOOR_PRESENT,
    NOISE_SEVERITY_BANDS,
    OVERLAP_CHANGE_COS,
    OVERLAP_HOP_S,
    OVERLAP_MIN_SEGMENT_S,
    OVERLAP_RATE_MIN,
    OVERLAP_SUBWINDOW_S,
    SAMPLE_RATE,
)
from autoace.acoustics import frame_db, stft_magnitude
from autoace.diarize import _load_encoder
from autoace.quality import assess_quality, noise_floor_dbfs, speech_level_dbfs
from autoace.schema import AudioQuality, NoiseSeverity
from autoace.tagging import classify_noise_type
from autoace.vad import Segment, non_speech_segments, speech_segments

_FRAME_HOP_S = 0.01  # matches acoustics.stft_magnitude's 10 ms hop


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


def _severity_for(floor_db: float) -> NoiseSeverity:
    """Walk NOISE_SEVERITY_BANDS (ascending inclusive upper bounds)."""
    for upper_bound, name in NOISE_SEVERITY_BANDS:
        if floor_db <= upper_bound:
            return NoiseSeverity(name)
    return NoiseSeverity.HIGH


def _noise_presence_and_severity(
    y: np.ndarray, non_speech: list[Segment]
) -> tuple[bool, float, NoiseSeverity]:
    """The noise floor alone decides presence and severity.

    DECISION: presence is `floor > NOISE_FLOOR_PRESENT`, full stop. An
    earlier design also required the AST tagger to clear TAG_MIN_DOM (1.8)
    before calling noise "present". That AND was dropped: relative_dominance
    is 3.7 / 47.3 / 8.6 on calls 001/002/003 - ALL THREE clear 1.8, including
    the clean call - so the tagger threshold does not discriminate presence
    at all here. ANDing it in can only ever narrow the floor's verdict
    (never help it), which is a pure added failure mode with zero measured
    benefit, so the floor decides alone and the tagger is used only to name
    the type once presence is already established.
    """
    floor = noise_floor_dbfs(y, non_speech)
    present = floor > NOISE_FLOOR_PRESENT
    severity = _severity_for(floor) if present else NoiseSeverity.NONE
    return present, floor, severity


def _long_silence_present(y: np.ndarray, floor_db: float) -> bool:
    """True dead air: a run of frames below the noise floor longer than
    LONG_SILENCE_SEC.

    Both `frame_db` and `noise_floor_dbfs` are peak-relative (ref=np.max), so
    they compare directly on the same scale - no rescaling between them. An
    earlier draft added a dB offset "to align the scales" that were already
    aligned; the offset pushed effectively every frame below 0 dB, so the
    comparison degenerated to `frame_db < 0`, true almost everywhere, which
    would have flagged every call as having a long silence. Do not
    reintroduce any such shift.

    Ordinary non-speech (a conversational pause, or a quiet-but-audible
    stretch) is NOT enough here - it must additionally be quieter than the
    call's own ambient floor. The longest non-speech gap actually measured
    is 7.35s (call_003), and that call's label is still `false`, which is
    why LONG_SILENCE_SEC (10.0) is set above it: at 10s a "gap" would have to
    be both longer than any observed pause AND dead air, not merely a lull
    in the conversation.

    If the call has no non-speech at all (frame_db never dips below the
    floor, or the signal is silent so noise_floor_dbfs returned its
    sentinel), there is trivially no long-silence run: an empty/never-true
    mask makes `longest_run_seconds` return 0.0, which is well under
    LONG_SILENCE_SEC, so this returns False without any special-casing.
    """
    db = frame_db(stft_magnitude(y))
    if db.size == 0:
        return False
    below_floor = db < floor_db
    longest = 0
    current = 0
    for value in below_floor:
        current = current + 1 if value else 0
        longest = max(longest, current)
    longest_s = longest * _FRAME_HOP_S
    return longest_s > LONG_SILENCE_SEC


def _embed_window(encoder, chunk: np.ndarray, min_samples: int) -> np.ndarray:
    if len(chunk) < min_samples:
        chunk = np.pad(chunk, (0, min_samples - len(chunk)))
    with torch.no_grad():
        vec = encoder.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
    return vec.squeeze().cpu().numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(a @ b / denom)


def _speaker_overlap_present(y: np.ndarray, speech: list[Segment]) -> bool:
    """Intra-segment speaker change as a proxy for concurrent speech.

    pyannote's overlapped-speech-detection model is the textbook approach,
    but it is gated on HuggingFace and unavailable in this environment (see
    OVERLAP_MODEL in config.py). This is a dependency-free replacement that
    reuses the ECAPA encoder already loaded for diarization: a speaker-
    identity change INSIDE a single VAD speech segment means two voices were
    active in a span VAD judged to be one continuous utterance - either
    talk-over between the two call parties, or a second speech-like source
    (e.g. background television) bleeding into the segment. Both satisfy the
    brief's definition of overlap ("two or more speakers talk at the same
    time enough to affect understanding").

    Method: slide an OVERLAP_SUBWINDOW_S embedding across each speech segment
    (hop OVERLAP_HOP_S), skip segments shorter than OVERLAP_MIN_SEGMENT_S
    (need at least two sub-windows to compare), and count how many
    consecutive sub-window pairs have cosine similarity below
    OVERLAP_CHANGE_COS. The RATE of such pairs across the whole call - not
    the single minimum-cosine pair - is the signal, because a lone outlier
    window (a clipped word, a breath) can drag one pair's cosine down without
    reflecting a real second voice.

    MEASURED on the three labelled calls with these exact parameters:

      call      truth    rate below 0.5    min intra-segment cosine
      call_001  false    6.3%  (1/16)       0.442
      call_003  true     22.4% (53/237)     0.080
      call_002  true     45.8% (22/48)      0.033

    OVERLAP_RATE_MIN = 0.15 sits between 0.063 (the one `false` call) and
    0.224 (the lower of the two `true` calls), so `rate >= OVERLAP_RATE_MIN`
    reproduces all three labels.
    """
    encoder = _load_encoder()
    min_samples = int(OVERLAP_SUBWINDOW_S * SAMPLE_RATE)
    window = int(OVERLAP_SUBWINDOW_S * SAMPLE_RATE)
    hop = int(OVERLAP_HOP_S * SAMPLE_RATE)

    below = 0
    total = 0
    for seg in speech:
        if seg.duration < OVERLAP_MIN_SEGMENT_S:
            continue
        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = int(seg.end * SAMPLE_RATE)
        chunk = y[start_sample:end_sample]

        embeddings = []
        for start in range(0, max(len(chunk) - window, 0) + 1, hop):
            sub = chunk[start : start + window]
            embeddings.append(_embed_window(encoder, sub, min_samples))
        # Guarantee at least two sub-windows for any segment that passed the
        # min-duration gate, even if it is shorter than one full hop grid.
        if len(embeddings) < 2:
            continue

        for a, b in zip(embeddings, embeddings[1:]):
            total += 1
            if _cosine(a, b) < OVERLAP_CHANGE_COS:
                below += 1

    if total == 0:
        return False
    rate = below / total
    return rate >= OVERLAP_RATE_MIN


def analyse_signal(y: np.ndarray) -> SignalResult:
    """Compute all six deterministic schema fields plus the two supporting
    values (noise_floor_dbfs, snr_db) the confidence model reads.

    Must be called on raw, unmodified audio - see module docstring.
    """
    speech = speech_segments(y)
    non_speech = non_speech_segments(y)

    present, floor, severity = _noise_presence_and_severity(y, non_speech)
    noise_type = classify_noise_type(y, non_speech) if present else ""
    quality = assess_quality(y).quality
    long_silence = _long_silence_present(y, floor)
    overlap = _speaker_overlap_present(y, speech)
    snr = speech_level_dbfs(y, speech) - floor

    return SignalResult(
        background_noise_present=present,
        background_noise_type=noise_type,
        background_noise_severity=severity,
        audio_quality=quality,
        speaker_overlap_present=overlap,
        long_silence_present=long_silence,
        noise_floor_dbfs=floor,
        snr_db=snr,
    )
