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
    fallback is also the statistically correct one."""
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
    voters agree."""
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


# ---------------------------------------------------------------------------
# A MISSING tone voter is not the same as two voters DISAGREEING, and the
# original formula could not tell them apart: tone_voters_agree=False merely
# withheld a bonus. Measured on call_001 with the Haiku key invalid, the
# NLI-only path produced `distressed/low` (truth: `upset/high`) at confidence
# 0.75 - above REVIEW_THRESHOLD 0.60 - so a wrong answer from a degraded path
# would never have reached a human.
# ---------------------------------------------------------------------------


def test_missing_llm_voter_is_penalised_beyond_mere_disagreement():
    """Same inputs, differing only in whether the LLM ran at all."""
    common = dict(
        tone_voters_agree=False, intensity_voters_agree=True,
        ser_consistent=True, llm_self_confidence=0.0,
        evidence_conflict=False, near_threshold=False,
        tier=Tier.B, diarization_degraded=False, asr_avg_logprob=-0.2,
    )
    disagreed = compute_confidence(ConfidenceInputs(**common, llm_unavailable=False))
    absent = compute_confidence(ConfidenceInputs(**common, llm_unavailable=True))
    assert absent < disagreed


def test_missing_llm_voter_always_lands_in_the_review_queue():
    """However well everything else agrees, a result produced without the
    primary classifier must be flagged for a human."""
    from autoace.config import REVIEW_THRESHOLD

    best_case = ConfidenceInputs(
        tone_voters_agree=True, intensity_voters_agree=True,
        ser_consistent=True, llm_self_confidence=0.99,
        evidence_conflict=False, near_threshold=False,
        tier=Tier.A, diarization_degraded=False, asr_avg_logprob=0.0,
        llm_unavailable=True,
    )
    assert compute_confidence(best_case) < REVIEW_THRESHOLD


def test_llm_available_is_the_default_and_unpenalised():
    """Existing call sites that omit the flag must be unaffected."""
    inputs = ConfidenceInputs(
        tone_voters_agree=True, intensity_voters_agree=True,
        ser_consistent=True, llm_self_confidence=0.9,
        evidence_conflict=False, near_threshold=False,
        tier=Tier.A, diarization_degraded=False, asr_avg_logprob=0.0,
    )
    assert inputs.llm_unavailable is False
    assert compute_confidence(inputs) > 0.85
