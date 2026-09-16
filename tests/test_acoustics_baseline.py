"""Pins the published acoustic characterisation. These exact figures justify
NOISE_FLOOR_PRESENT, LONG_SILENCE_SEC, and the telephony muffling baseline.
If this test fails, the calibration argument no longer holds and must be
re-derived from scratch."""

import pytest

from emotion_detection.acoustics import baseline_characterisation
from emotion_detection.config import reference_call

# (file, snr_db, floor_dbfs, max_gap_s, clip_pct, hf_fraction)
EXPECTED = [
    ("call_001.ogg", 42.2, -56.3, 2.87, 0.00, 0.044),
    ("call_002.ogg", 29.9, -52.1, 3.29, 0.00, 0.043),
    ("call_003.ogg", 34.9, -47.0, 7.35, 0.00, 0.060),
]


@pytest.mark.parametrize("name,snr,floor,gap,clip,hf", EXPECTED)
def test_baseline_matches_published_characterisation(name, snr, floor, gap, clip, hf):
    m = baseline_characterisation(reference_call(name))
    assert m["snr_db"] == pytest.approx(snr, abs=0.1)
    assert m["floor_dbfs"] == pytest.approx(floor, abs=0.1)
    assert m["max_nonspeech_gap_s"] == pytest.approx(gap, abs=0.01)
    assert m["clip_pct"] == pytest.approx(clip, abs=0.01)
    assert m["hf_fraction"] == pytest.approx(hf, abs=0.001)


def test_long_silence_threshold_exceeds_measured_gap():
    """call_003 has a 7.35s gap and is labelled long_silence_present=false,
    so the threshold must sit above it."""
    from emotion_detection.config import LONG_SILENCE_SEC
    worst = max(
        baseline_characterisation(reference_call(n))["max_nonspeech_gap_s"]
        for n, *_ in EXPECTED
    )
    assert LONG_SILENCE_SEC > worst


def test_noise_floor_threshold_separates_the_labels():
    """The one no-noise call must fall below NOISE_FLOOR_PRESENT and both
    noisy calls above it."""
    from emotion_detection.config import NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_001.ogg"))["floor_dbfs"] < NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_002.ogg"))["floor_dbfs"] > NOISE_FLOOR_PRESENT
    assert baseline_characterisation(reference_call("call_003.ogg"))["floor_dbfs"] > NOISE_FLOOR_PRESENT


def test_telephony_bandwidth_is_narrow_on_all_calls():
    """4-6% of energy above 3.4 kHz is normal telephony, not muffling. A
    detector baselined on wideband speech would flag all three."""
    for name, *_ in EXPECTED:
        hf = baseline_characterisation(reference_call(name))["hf_fraction"]
        assert 0.03 < hf < 0.08
