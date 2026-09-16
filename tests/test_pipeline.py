import pytest

from emotion_detection.config import LABELS_CSV, reference_call
from emotion_detection.pipeline import analyse_file
from emotion_detection.schema import CallAnalysis


def test_malformed_file_returns_an_error_not_an_exception():
    """A single bad file must never fail a batch."""
    result = analyse_file(str(LABELS_CSV))
    assert result.error is not None
    assert result.analysis is None


@pytest.mark.slow
def test_signal_fields_survive_a_failing_tone_branch():
    """The API key is invalid here, so the tone branch takes its fallback path.
    The six signal fields must still be correct - they are the reliable-points
    block and must not depend on an API."""
    result = analyse_file(reference_call("call_001.ogg"))
    assert result.error is None
    assert isinstance(result.analysis, CallAnalysis)
    assert result.analysis.background_noise_present is False
    assert result.analysis.background_noise_severity.value == "none"
    assert result.analysis.audio_quality.value == "clear"
    assert result.analysis.speaker_overlap_present is False
    assert result.analysis.long_silence_present is False


@pytest.mark.slow
def test_confidence_is_lowered_when_the_llm_is_unavailable():
    """With only one tone voter, confidence must not read as if two agreed."""
    result = analyse_file(reference_call("call_001.ogg"))
    assert 0.05 <= result.analysis.confidence <= 0.98
    assert result.analysis.confidence < 0.85


@pytest.mark.slow
def test_reasoning_records_which_tone_path_was_used():
    """Operationally essential: a silent fallback looks identical to success."""
    result = analyse_file(reference_call("call_001.ogg"))
    assert result.reasoning
    assert any(k in result.reasoning.lower() for k in ("nli", "fallback", "haiku", "llm"))
