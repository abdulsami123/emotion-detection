import pytest

from autoace.config import reference_call
from autoace.diarize import assign_speakers
from autoace.io_audio import load_mono
from autoace.vad import Segment, speech_segments


@pytest.fixture(scope="module")
def call_003():
    y, _ = load_mono(reference_call("call_003.ogg"))
    return y, speech_segments(y)


def test_assigns_every_segment_to_a_role(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert set(result.assignments) == set(range(len(segments)))
    assert set(result.assignments.values()) <= {"agent", "customer"}


def test_agent_speaks_first(call_003):
    """The bot always greets first ('Hi, I'm Erica from...') - true on all
    three provided calls, and the basis of the fallback heuristic."""
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert result.assignments[0] == "agent"


def test_both_roles_present_in_a_two_party_call(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert set(result.assignments.values()) == {"agent", "customer"}
    assert result.degraded is False


def test_customer_speech_duration_is_reported_and_bounded(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    total = sum(s.duration for s in segments)
    assert 0 < result.customer_speech_seconds < total


def test_customer_segments_match_the_assignment_map(call_003):
    y, segments = call_003
    result = assign_speakers(y, segments)
    expected = [s for i, s in enumerate(segments) if result.assignments[i] == "customer"]
    assert result.customer_segments == expected


def test_too_few_segments_is_marked_degraded(call_003):
    """A clip with one segment cannot be split into two speakers; the
    pipeline must know so it can cap confidence."""
    y, _ = call_003
    result = assign_speakers(y, [Segment(0.0, 1.0)])
    assert result.degraded is True
    assert set(result.assignments.values()) == {"customer"}
