import numpy as np
import pytest

from autoace.config import NOISE_FLOOR_PRESENT, reference_call
from autoace.io_audio import load_mono
from autoace.quality import assess_quality, noise_floor_dbfs, speech_level_dbfs
from autoace.schema import AudioQuality
from autoace.vad import non_speech_segments, speech_segments

CALLS = ("call_001.ogg", "call_002.ogg", "call_003.ogg")


@pytest.fixture(scope="module")
def audio():
    return {name: load_mono(reference_call(name))[0] for name in CALLS}


@pytest.mark.parametrize("name", CALLS)
def test_all_provided_calls_are_clear(audio, name):
    """All three labels say `clear`, including the one whose noise type is
    'sharp static'. audio_quality degrades only when intelligibility suffers."""
    assert assess_quality(audio[name]).quality == AudioQuality.CLEAR


@pytest.mark.parametrize("name", CALLS)
def test_telephony_bandwidth_is_not_treated_as_muffling(audio, name):
    """4-6% of energy above 3.4 kHz is normal narrowband telephony. A detector
    baselined on wideband speech flags all three."""
    assert assess_quality(audio[name]).detectors["muffled"] is False


@pytest.mark.parametrize("name", CALLS)
def test_no_clipping_on_provided_calls(audio, name):
    assert assess_quality(audio[name]).detectors["clipping"] is False


def test_noise_floor_ordering_matches_severity_labels(audio):
    """Floor tracks severity; SNR does not. Ground truth is none / medium /
    medium for calls 001 / 002 / 003."""
    floors = {
        name: noise_floor_dbfs(audio[name], non_speech_segments(audio[name]))
        for name in CALLS
    }
    assert floors["call_001.ogg"] < floors["call_002.ogg"]
    assert floors["call_002.ogg"] < floors["call_003.ogg"]


@pytest.mark.parametrize(
    "name,expected_floor",
    [("call_001.ogg", -56.3), ("call_002.ogg", -52.1), ("call_003.ogg", -47.0)],
)
def test_production_floor_is_on_the_calibration_scale(audio, name, expected_floor):
    """NOISE_SEVERITY_BANDS was calibrated on levels relative to each call's
    PEAK frame. If this estimator drifts onto absolute dBFS, every threshold in
    that table silently breaks. Wide tolerance because the estimators differ
    (Silero non-speech vs a percentile split) - this checks the SCALE."""
    floor = noise_floor_dbfs(audio[name], non_speech_segments(audio[name]))
    assert floor == pytest.approx(expected_floor, abs=8.0)


def test_floor_separates_the_presence_threshold(audio):
    """The clean call must fall below NOISE_FLOOR_PRESENT and both noisy calls
    above it - this is what makes the floor the presence detector."""
    assert noise_floor_dbfs(audio["call_001.ogg"], non_speech_segments(audio["call_001.ogg"])) < NOISE_FLOOR_PRESENT
    assert noise_floor_dbfs(audio["call_002.ogg"], non_speech_segments(audio["call_002.ogg"])) > NOISE_FLOOR_PRESENT
    assert noise_floor_dbfs(audio["call_003.ogg"], non_speech_segments(audio["call_003.ogg"])) > NOISE_FLOOR_PRESENT


def test_speech_is_louder_than_the_floor(audio):
    """Sanity check that both levels are on the same scale: speech must exceed
    the noise floor on every call."""
    for name in CALLS:
        y = audio[name]
        assert speech_level_dbfs(y, speech_segments(y)) > noise_floor_dbfs(
            y, non_speech_segments(y)
        )


def test_empty_region_returns_a_floor_sentinel(audio):
    """No non-speech at all must not raise or return 0.0, which would read as
    an extremely loud floor and force `high` severity."""
    assert noise_floor_dbfs(audio["call_001.ogg"], []) <= -80.0


# ---------------------------------------------------------------------------
# The test that was missing, and its absence let two real bugs through.
#
# The suite previously checked only that floors were ORDERED and that they
# straddled NOISE_FLOOR_PRESENT. Both held while severity was wrong on two of
# three calls: first because a minimum-statistics estimator drifted the floor
# 5 dB (collapsing the two noisy calls into one band), then because
# NOISE_SEVERITY_BANDS placed the low/medium boundary at -50.0, above the
# -52.1 `medium` anchor. Ordering assertions cannot catch either. Assert the
# label the schema actually emits.
# ---------------------------------------------------------------------------

from autoace.config import NOISE_SEVERITY_BANDS


def _severity_for(floor_db: float) -> str:
    for upper_bound, name in NOISE_SEVERITY_BANDS:
        if floor_db <= upper_bound:
            return name
    return "high"


@pytest.mark.parametrize(
    "name,expected_severity",
    [("call_001.ogg", "none"), ("call_002.ogg", "medium"), ("call_003.ogg", "medium")],
)
def test_floor_maps_to_the_labelled_severity(audio, name, expected_severity):
    """End-to-end on the field that ships: floor -> severity band -> label."""
    y = audio[name]
    floor = noise_floor_dbfs(y, non_speech_segments(y))
    assert _severity_for(floor) == expected_severity, (
        f"{name}: floor {floor:.1f} dB mapped to {_severity_for(floor)!r}, "
        f"expected {expected_severity!r}"
    )


def test_severity_bands_are_monotonic_and_bracket_the_anchors():
    """A band table whose bounds are out of order, or which places a boundary
    between two same-class anchors, is silently broken. -52.1 and -47.0 are
    both `medium`, so no boundary may sit between them."""
    bounds = [b for b, _ in NOISE_SEVERITY_BANDS]
    assert bounds == sorted(bounds), "severity bounds must be ascending"
    for boundary in bounds[:-1]:
        assert not (-52.1 < boundary < -47.0), (
            f"boundary {boundary} sits between the two `medium` anchors "
            f"(-52.1, -47.0) and would split them across bands"
        )
