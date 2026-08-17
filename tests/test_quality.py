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
