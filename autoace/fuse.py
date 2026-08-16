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
    ASR_MIN_LOGPROB,
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
    """Deterministic label-vs-evidence check. Validates the emitted label
    against numeric acoustic evidence rather than parsing a prose rationale,
    which is unreliable to machine-check."""
    if tone in POSITIVE_TONES and dims.valence < VALENCE_POS_MIN:
        return True
    if tone in NEGATIVE_TONES and dims.valence > VALENCE_NEG_MAX:
        return True
    if intensity is EmotionalIntensity.HIGH and dims.arousal < AROUSAL_HIGH_MIN:
        return True
    return False


def compute_confidence(inputs: ConfidenceInputs) -> float:
    """Built from voter agreement, never from model self-report.

    A model asked to rate its own certainty returns a number that clusters
    around 0.8 and correlates with nothing, so its self-rating is demoted to
    one weak term among several.
    """
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
