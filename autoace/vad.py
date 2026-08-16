"""Silero VAD. Its segmentation is shared across both branches: speech
regions drive prosody and speaker assignment, non-speech regions drive noise
estimation, and the gaps drive long-silence detection."""

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
    n = len(y)
    parts = [
        y[int(s.start * SAMPLE_RATE) : min(int(s.end * SAMPLE_RATE), n)]
        for s in segments
    ]
    return np.concatenate(parts).astype(np.float32)
