"""Shared DSP primitives. Used by both the tagging and quality modules so
noise-floor logic exists in exactly one place."""

from __future__ import annotations

import librosa
import numpy as np

from emotion_detection.config import SAMPLE_RATE

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
    """Reproduce the published acoustic characterisation exactly.

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
