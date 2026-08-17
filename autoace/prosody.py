"""Prosody feature extraction and activation scoring.

Intensity tracks activation (how animated the caller is), not tone or
classifier confidence - see the module-level design notes in the task spec.
Every acoustic feature is z-scored against the speaker's own opening window
before it reaches the LLM, and only ever surfaces as a categorical tag
("much louder", "rising pitch"), never as a raw number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np

from autoace.config import ACTIVATION_WEIGHTS, BASELINE_WINDOW_S, SAMPLE_RATE
from autoace.vad import Segment


class Tier(str, Enum):
    """How much customer speech is available, which selects the method."""

    A = "A"   # >15s: self-baseline + trajectory
    B = "B"   # 3-15s: corpus norms, no trajectory
    C = "C"   # <3s: insufficient evidence


@dataclass
class ActivationProfile:
    activation_z: float
    slope_rising: bool
    peak_z: float


# eGeMAPSv02 functional names, verified present against Smile.feature_names.
EGEMAPS_FIELDS = {
    "loudness_range": "loudness_sma3_pctlrange0-2",
    "f0_elevation": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "f0_range": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "rate_deviation": "VoicedSegmentsPerSec",
    "jitter_shimmer": "jitterLocal_sma3nz_amean",
}

_MIN_SLICE_S = 0.1


def select_tier(customer_speech_seconds: float) -> Tier:
    """Total customer speech selects the method - see the tiering table."""
    from autoace.config import TIER_A_MIN_SPEECH_S, TIER_C_MAX_SPEECH_S

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
    """eGeMAPS functionals for one utterance, keyed by our internal names.

    Slices too short for openSMILE to produce a stable estimate return zeros
    rather than invoking it. All five eGeMAPS columns are verified present on
    this feature set/level, so a KeyError pulling one out is a real failure
    (e.g. a version mismatch) and must surface, not be papered over with a
    fallback 0.0.
    """
    if segment.duration < _MIN_SLICE_S:
        return {name: 0.0 for name in EGEMAPS_FIELDS}

    start = int(segment.start * SAMPLE_RATE)
    end = int(segment.end * SAMPLE_RATE)
    chunk = y[start:end]
    if len(chunk) < int(_MIN_SLICE_S * SAMPLE_RATE):
        return {name: 0.0 for name in EGEMAPS_FIELDS}

    smile = _load_smile()
    result = smile.process_signal(chunk, SAMPLE_RATE)
    row = result.iloc[0]
    return {name: float(row[column]) for name, column in EGEMAPS_FIELDS.items()}


def baseline_statistics(
    y: np.ndarray, segments: list[Segment]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-field mean/std over the speaker's opening window.

    Accumulates segments (in order) until BASELINE_WINDOW_S of speech is
    covered, extracting features for each. Std is guarded against zero (a
    single baseline segment, or several identical ones, would otherwise
    blow up or silently clamp every z-score downstream) by falling back
    to 1.0.
    """
    covered = 0.0
    per_field: dict[str, list[float]] = {name: [] for name in EGEMAPS_FIELDS}
    for seg in segments:
        if covered >= BASELINE_WINDOW_S:
            break
        features = extract_features(y, seg)
        for name, value in features.items():
            per_field[name].append(value)
        covered += seg.duration

    means = {name: float(np.mean(values)) if values else 0.0 for name, values in per_field.items()}
    stds = {
        name: (float(np.std(values)) if values and np.std(values) > 0.0 else 1.0)
        for name, values in per_field.items()
    }
    return means, stds


def z_score(
    features: dict[str, float], means: dict[str, float], stds: dict[str, float]
) -> dict[str, float]:
    return {
        name: (value - means[name]) / stds[name]
        for name, value in features.items()
        if name in means
    }


def discretise(z_scores: dict[str, float]) -> list[str]:
    """Categorical tags for the tone LLM - never raw numbers.

    Returns ["baseline"] when nothing deviates, so the tone model always
    gets at least one tag.
    """
    tags: list[str] = []

    loudness = z_scores.get("loudness_range")
    if loudness is not None:
        if loudness > 2.0:
            tags.append("much louder")
        elif loudness > 1.0:
            tags.append("louder")
        elif loudness < -1.0:
            tags.append("quieter")

    f0_elevation = z_scores.get("f0_elevation")
    if f0_elevation is not None:
        if f0_elevation > 1.0:
            tags.append("rising pitch")
        elif f0_elevation < -1.0:
            tags.append("flat pitch")

    f0_range = z_scores.get("f0_range")
    if f0_range is not None and f0_range > 1.0:
        tags.append("high pitch variance")

    rate = z_scores.get("rate_deviation")
    if rate is not None:
        if rate > 1.0:
            tags.append("faster")
        elif rate < -1.0:
            tags.append("slower")

    jitter = z_scores.get("jitter_shimmer")
    if jitter is not None and jitter > 1.0:
        tags.append("strained voice")

    return tags or ["baseline"]


def combine(
    z_scores: dict[str, float], extra: dict[str, float] | None = None
) -> float:
    """Weighted sum of activation channels using ACTIVATION_WEIGHTS.

    `extra` carries channels that arrive from outside eGeMAPS entirely
    (interactional and lexical signals, dimensional SER arousal) so they can
    be merged into the same weighted score.
    """
    combined = dict(z_scores)
    if extra:
        combined.update(extra)
    return sum(
        value * ACTIVATION_WEIGHTS[name]
        for name, value in combined.items()
        if name in ACTIVATION_WEIGHTS
    )


def activation_profile(per_third: list[float]) -> ActivationProfile:
    """Summarise a trajectory of per-window activation scores.

    `per_third` is the combined activation score for each of (typically)
    three successive windows across the customer's speech. Handles 0 or 1
    points, where a slope cannot be fit.
    """
    if not per_third:
        return ActivationProfile(activation_z=0.0, slope_rising=False, peak_z=0.0)

    values = np.asarray(per_third, dtype=float)
    activation_z = float(values.mean())
    peak_z = float(values.max())

    if len(values) < 2:
        slope_rising = False
    else:
        xs = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(xs, values, 1)[0])
        slope_rising = slope > 0.1

    return ActivationProfile(
        activation_z=activation_z, slope_rising=slope_rising, peak_z=peak_z
    )
