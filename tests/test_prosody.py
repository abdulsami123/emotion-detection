import numpy as np
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.prosody import (
    ActivationProfile,
    Tier,
    activation_profile,
    baseline_statistics,
    combine,
    discretise,
    extract_features,
    select_tier,
    z_score,
)
from autoace.vad import Segment, speech_segments


def test_tier_selection_from_customer_speech_duration():
    assert select_tier(20.0) == Tier.A
    assert select_tier(6.0) == Tier.B
    assert select_tier(1.0) == Tier.C


def test_minimal_speech_is_tier_c():
    """call_002's customer says two words - about a second. There is no
    baseline to establish and no trajectory to fit."""
    assert select_tier(1.2) == Tier.C


def test_discretise_produces_relative_tags_not_numbers():
    tags = discretise(
        {"loudness_range": 2.4, "f0_elevation": 1.1, "rate_deviation": 0.2}
    )
    assert "much louder" in tags
    assert all(not any(ch.isdigit() for ch in tag) for tag in tags)


def test_discretise_returns_baseline_when_nothing_deviates():
    assert discretise({"loudness_range": 0.0, "f0_elevation": 0.0}) == ["baseline"]


def test_rising_trajectory_is_detected():
    profile = activation_profile([0.1, 0.9, 2.2])
    assert profile.slope_rising is True
    assert profile.peak_z == pytest.approx(2.2)


def test_flat_trajectory_is_not_rising():
    profile = activation_profile([1.0, 1.05, 0.95])
    assert profile.slope_rising is False


def test_activation_profile_handles_empty_input():
    profile = activation_profile([])
    assert profile.activation_z == 0.0
    assert profile.slope_rising is False


def test_extract_features_returns_all_five_fields():
    """All five eGeMAPS column names are verified present, so a zero here
    means extraction silently failed rather than the feature being absent."""
    y, _ = load_mono(reference_call("call_001.ogg"))
    segments = speech_segments(y)
    features = extract_features(y, segments[0])
    assert set(features) == {
        "loudness_range",
        "f0_elevation",
        "f0_range",
        "rate_deviation",
        "jitter_shimmer",
    }
    assert any(v != 0.0 for v in features.values()), "all features zero - extraction failed"


def test_baseline_statistics_gives_nonzero_spread():
    """z-scoring divides by the baseline std, so a zero std would blow up or
    silently clamp. It must be guarded."""
    y, _ = load_mono(reference_call("call_003.ogg"))
    segments = speech_segments(y)
    means, stds = baseline_statistics(y, segments)
    assert set(means) == set(stds)
    assert all(s > 0.0 for s in stds.values())


def test_z_score_of_the_baseline_itself_is_near_zero():
    means = {"loudness_range": 5.0}
    stds = {"loudness_range": 2.0}
    assert z_score({"loudness_range": 5.0}, means, stds)["loudness_range"] == 0.0
    assert z_score({"loudness_range": 7.0}, means, stds)["loudness_range"] == 1.0


def test_combine_is_a_weighted_sum_using_config_weights():
    from autoace.config import ACTIVATION_WEIGHTS

    score = combine({"loudness_range": 2.0})
    assert score == pytest.approx(2.0 * ACTIVATION_WEIGHTS["loudness_range"])


def test_combine_accepts_extra_non_vocal_channels():
    """Interactional and lexical channels arrive separately from eGeMAPS."""
    from autoace.config import ACTIVATION_WEIGHTS

    score = combine({}, {"ser_arousal": 1.0})
    assert score == pytest.approx(ACTIVATION_WEIGHTS["ser_arousal"])
