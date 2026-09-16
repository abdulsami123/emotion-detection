import numpy as np
import pytest

from emotion_detection.config import reference_call
from emotion_detection.io_audio import load_mono
from emotion_detection.ser import Dimensions, predict_dimensions

CALLS = ("call_001.ogg", "call_002.ogg", "call_003.ogg")


@pytest.fixture(scope="module")
def dims():
    return {
        name: predict_dimensions(load_mono(reference_call(name))[0]) for name in CALLS
    }


@pytest.mark.parametrize("name", CALLS)
def test_returns_three_bounded_dimensions(dims, name):
    result = dims[name]
    assert isinstance(result, Dimensions)
    for axis in ("arousal", "dominance", "valence"):
        value = getattr(result, axis)
        assert 0.0 <= value <= 1.0, f"{axis}={value} out of range"


def test_dimensions_differ_across_calls(dims):
    """A model returning the same triple for every input carries no signal and
    would make the SER evidence useless to the tone classifier."""
    signatures = {
        (round(d.arousal, 3), round(d.valence, 3)) for d in dims.values()
    }
    assert len(signatures) == len(CALLS)


def test_valence_ranks_the_satisfied_call_above_the_upset_call(dims):
    """Ground truth: call_003 is `satisfied` (warm, polite caller), call_001 is
    `upset` (caller escalating at an unresponsive bot). If valence does not
    order them that way, the output axes are probably mislabelled - which
    would silently invert fuse.coherence_conflict."""
    assert dims["call_003.ogg"].valence > dims["call_001.ogg"].valence


def test_as_prompt_line_is_human_readable(dims):
    line = dims["call_001.ogg"].as_prompt_line()
    assert "arousal" in line and "dominance" in line and "valence" in line
    assert "0." in line


def test_short_audio_does_not_raise():
    """A one-second clip is shorter than the analysis window."""
    result = predict_dimensions(np.zeros(16000, dtype=np.float32))
    assert 0.0 <= result.arousal <= 1.0
