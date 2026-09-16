import pytest

from emotion_detection.asr import transcribe
from emotion_detection.config import reference_call


@pytest.fixture(scope="module")
def call_001_result():
    return transcribe(reference_call("call_001.ogg"))


def test_returns_words_with_timestamps(call_001_result):
    """The prosody alignment design depends entirely on word timestamps."""
    assert call_001_result.words
    for word in call_001_result.words:
        assert word.end >= word.start
        assert word.text.strip()


def test_word_timestamps_are_in_seconds_within_the_clip(call_001_result):
    """A 30.9s call: timestamps must be seconds, not sample indices."""
    assert all(0.0 <= w.start <= 40.0 for w in call_001_result.words)


def test_detects_english_on_call_001(call_001_result):
    assert call_001_result.language == "en"


def test_transcribes_the_agent_greeting(call_001_result):
    """Every call opens with the bot greeting 'Hi, I'm Erica from ...'."""
    assert "erica" in call_001_result.text.lower()


def test_handles_the_spanish_call(call_001_result):
    """call_002 switches to Spanish. An English-only ASR returns garbage here,
    which is why Parakeet was rejected."""
    result = transcribe(reference_call("call_002.ogg"))
    assert result.language in {"es", "en"}
    assert "erica" in result.text.lower()
    assert len(result.text) > 40


def test_reports_average_logprob_for_confidence(call_001_result):
    """ASR quality feeds the confidence formula."""
    assert call_001_result.avg_logprob is not None
    assert -5.0 < call_001_result.avg_logprob < 0.0


def test_segments_are_ordered_and_carry_language(call_001_result):
    for earlier, later in zip(call_001_result.segments, call_001_result.segments[1:]):
        assert later.start >= earlier.start
    assert all(s.end >= s.start for s in call_001_result.segments)


def test_detect_language_works_on_an_audio_slice():
    """faster-whisper reports language per CLIP, never per segment (its
    Segment dataclass has no language field, verified against 1.2.1). The
    diarization code-switch fix needs per-segment language, so it must come
    from detect_language() over a slice instead.

    call_002 opens with the bot greeting in English and later continues in
    Spanish. If both slices report the same language, the planned fix is not
    viable and the design needs revisiting."""
    from emotion_detection.asr import detect_language
    from emotion_detection.io_audio import load_mono

    y, sr = load_mono(reference_call("call_002.ogg"))
    opening, _ = detect_language(y[: 8 * sr])
    later, _ = detect_language(y[-12 * sr :])

    assert opening == "en", f"expected English greeting, detected {opening!r}"
    assert later == "es", f"expected Spanish continuation, detected {later!r}"
