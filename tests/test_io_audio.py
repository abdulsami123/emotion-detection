import numpy as np
import pytest

from autoace.config import LABELS_CSV, reference_call
from autoace.io_audio import UnsupportedAudio, channel_layout, load_mono


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_load_mono_returns_16k_float32(name):
    y, sr = load_mono(reference_call(name))
    assert sr == 16000
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) > 0


@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_provided_calls_are_duplicated_mono(name):
    """The .ogg files are 2-channel but byte-identical, so there is no free
    speaker separation and diarization is required."""
    layout = channel_layout(reference_call(name))
    assert layout["channels"] == 2
    assert layout["correlation"] == pytest.approx(1.0, abs=1e-4)
    assert layout["separated"] is False


def test_no_loudness_normalization_is_applied():
    """Absolute level is load-bearing for the noise floor. The three calls
    have materially different levels; decoding must preserve that."""
    levels = [
        float(np.sqrt(np.mean(load_mono(reference_call(n))[0] ** 2)))
        for n in ("call_001.ogg", "call_002.ogg", "call_003.ogg")
    ]
    # Measured true ratio is 2.653. Peak normalization collapses it to 1.616
    # and RMS normalization to 1.000, so the threshold must sit above 1.616
    # to catch either. 2.0 leaves headroom on both sides.
    assert max(levels) / min(levels) > 2.0, (
        "levels are suspiciously uniform - has normalization crept in?"
    )


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedAudio):
        load_mono(str(LABELS_CSV))
