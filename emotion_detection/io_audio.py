"""Decode to a canonical 16 kHz mono float32 signal.

NO loudness normalization happens here or anywhere upstream of the signal
branch. Absolute level is load-bearing: the noise floor is both the primary
severity signal and the primary presence detector for background noise.
The tone branch obtains level-invariance by z-scoring against the speaker's
own baseline internally, never by touching the shared audio.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from emotion_detection.config import SAMPLE_RATE, STEREO_SEPARATE_MAX_CORR

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
    except Exception as exc:  # librosa surfaces a variety of backend errors
        raise UnsupportedAudio(f"{path.name}: {exc}") from exc
    return y.astype(np.float32), SAMPLE_RATE


def channel_layout(path: str | Path) -> dict:
    """Detect whether a stereo file carries genuinely separated legs.

    A dual-leg recording (agent on one channel, customer on the other) makes
    diarization unnecessary. The provided files are 2-channel but
    byte-identical, so they do not qualify.
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
