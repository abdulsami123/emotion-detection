import numpy as np
import pytest

from autoace.config import reference_call
from autoace.io_audio import load_mono
from autoace.vad import Segment, concatenate, non_speech_segments, speech_segments


@pytest.fixture(scope="module")
def call_001():
    return load_mono(reference_call("call_001.ogg"))[0]


def test_returns_ordered_non_overlapping_segments(call_001):
    segments = speech_segments(call_001)
    assert segments, "expected at least one speech segment in a 31s call"
    for seg in segments:
        assert isinstance(seg, Segment)
        assert seg.end > seg.start
    for earlier, later in zip(segments, segments[1:]):
        assert later.start >= earlier.end


def test_speech_covers_a_plausible_share_of_the_call(call_001):
    """A 31s two-party call should be mostly-but-not-entirely speech."""
    segments = speech_segments(call_001)
    total = sum(s.duration for s in segments)
    duration = len(call_001) / 16000
    assert 0.2 < total / duration < 0.95


def test_non_speech_is_the_complement(call_001):
    duration = len(call_001) / 16000
    speech = sum(s.duration for s in speech_segments(call_001))
    gaps = sum(s.duration for s in non_speech_segments(call_001))
    assert speech + gaps == pytest.approx(duration, abs=0.05)


def test_concatenate_splices_only_the_named_regions(call_001):
    segments = speech_segments(call_001)
    spliced = concatenate(call_001, segments)
    expected = int(sum(s.duration for s in segments) * 16000)
    assert abs(len(spliced) - expected) < 1000
    assert spliced.dtype == np.float32


def test_concatenate_of_nothing_is_empty(call_001):
    empty = concatenate(call_001, [])
    assert len(empty) == 0
    assert empty.dtype == np.float32
