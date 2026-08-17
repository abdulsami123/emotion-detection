import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.schema import AudioQuality, NoiseSeverity
from autoace.signal_branch import analyse_signal

CALLS = ("call_001.ogg", "call_002.ogg", "call_003.ogg")

EXPECTED = {
    "call_001.ogg": dict(
        present=False, severity=NoiseSeverity.NONE, overlap=False, silence=False
    ),
    "call_002.ogg": dict(
        present=True, severity=NoiseSeverity.MEDIUM, overlap=True, silence=False
    ),
    "call_003.ogg": dict(
        present=True, severity=NoiseSeverity.MEDIUM, overlap=True, silence=False
    ),
}


@pytest.fixture(scope="module")
def results():
    return {
        name: analyse_signal(load_mono(reference_call(name))[0]) for name in CALLS
    }


@pytest.mark.parametrize("name", CALLS)
def test_noise_presence_matches_labels(results, name):
    assert results[name].background_noise_present is EXPECTED[name]["present"]


@pytest.mark.parametrize("name", CALLS)
def test_noise_severity_matches_labels(results, name):
    assert results[name].background_noise_severity == EXPECTED[name]["severity"]


@pytest.mark.parametrize("name", CALLS)
def test_audio_quality_is_clear_on_all_provided_calls(results, name):
    assert results[name].audio_quality == AudioQuality.CLEAR


@pytest.mark.parametrize("name", CALLS)
def test_long_silence_is_false_on_all_provided_calls(results, name):
    """Longest non-speech gap is 7.35s (call_003) and the label is still
    false, so the threshold must exceed it and require true dead air."""
    assert results[name].long_silence_present is EXPECTED[name]["silence"]


@pytest.mark.parametrize("name", CALLS)
def test_speaker_overlap_matches_labels(results, name):
    """Intra-segment speaker change. Measured rates: 6.3% / 45.8% / 22.4% for
    calls 001 / 002 / 003 against labels false / true / true."""
    assert results[name].speaker_overlap_present is EXPECTED[name]["overlap"]


def test_type_is_empty_when_no_noise_present(results):
    assert results["call_001.ogg"].background_noise_type == ""


def test_type_is_populated_when_noise_present(results):
    assert results["call_002.ogg"].background_noise_type == "TV"
    assert results["call_003.ogg"].background_noise_type != ""


def test_floor_and_snr_are_reported_for_confidence(results):
    """Both feed the confidence calculation and the dashboard detail view."""
    for name in CALLS:
        assert -90.0 < results[name].noise_floor_dbfs < 0.0
        assert results[name].snr_db > 0.0
