"""faster-whisper transcription.

Three requirements the model choice must satisfy:
  1. Multilingual — call_002 is Spanish; an English-only model returns garbage.
  2. word_timestamps=True — the prosody alignment grid depends on them.
  3. Detected language captured, not discarded. Also the planned input to
     fixing the code-switch defect in speaker assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from emotion_detection.config import (
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
    words: list[Word] = field(default_factory=list)
    # NOTE: deliberately no `language` field. faster-whisper's Segment carries
    # no per-segment language (verified against 1.2.1), so populating one from
    # the clip-level `info.language` would look per-segment while carrying no
    # per-segment information. Use detect_language() below when you need it.


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
        return WhisperModel(
            WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE_GPU
        )
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE_CPU)


def transcribe(path: str) -> Transcript:
    model = _load_model()
    raw_segments, info = model.transcribe(
        path,
        word_timestamps=True,
        vad_filter=False,      # Silero VAD runs separately and its segmentation is shared
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


def detect_language(audio: np.ndarray) -> tuple[str, float]:
    """Detect the language of an arbitrary 16 kHz mono audio array.

    Separate from `transcribe` because faster-whisper reports language only
    per clip, never per segment. This operates on any slice of audio, which
    is what makes it usable on individual VAD segments.

    This is the intended input to fixing the code-switch defect in speaker
    assignment: on a call where the agent greets in English and then continues
    in Spanish, speaker embeddings split one speaker into two clusters. Knowing
    each segment's language lets those clusters be recognised as one speaker.

    Returns (language_code, probability).
    """
    language, probability, _all_probs = _load_model().detect_language(
        audio=audio.astype(np.float32)
    )
    return language, float(probability)
