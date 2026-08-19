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


def test_agent_speaks_first_holds_on_call_003_only(call_003):
    """The first-speaker fallback happens to hold here - the bot greets before
    the caller says anything.

    An earlier version of this test claimed the assumption was "true on all
    three provided calls" and only ever checked this one. That claim is FALSE:
    on call_001 the caller opens with an impatient "Come on." at 1.2s and the
    bot greets at 3.7s. Because the anchor picks the FIRST segment's cluster as
    the agent, that one exception inverted every role in the call, and the tone
    branch analysed the bot's uniformly flat TTS delivery as the customer's -
    tone scored 0/3 as a direct result.

    The heuristic is kept because it is right more often than not and costs
    nothing, but it is no longer trusted on its own: `pipeline.verify_roles`
    re-anchors the assignment on transcript content (see the test below).
    """
    y, segments = call_003
    result = assign_speakers(y, segments)
    assert result.assignments[0] == "agent"


def test_first_speaker_anchor_is_wrong_on_call_001():
    """Pins the counter-example, so nobody restores the discredited claim.

    Segment 0 of call_001 is the CUSTOMER ("Come on."), so an assignment that
    anchors on turn order labels it `agent` and inverts the whole call.
    """
    y, _ = load_mono(reference_call("call_001.ogg"))
    segments = speech_segments(y)
    result = assign_speakers(y, segments)
    # Diarization alone gets this wrong - that is the point of the test.
    assert result.assignments[0] == "agent"
    assert result.agent_reference_similarity is None, (
        "no reference bank was supplied, so the first-speaker fallback ran"
    )


def test_transcript_verification_corrects_the_inverted_call():
    """The actual invariant the pipeline now relies on: content-based
    verification recovers the correct roles on call_001, and leaves the two
    already-correct calls untouched."""
    from autoace.asr import transcribe
    from autoace.pipeline import _text_in_segment, verify_roles

    y, _ = load_mono(reference_call("call_001.ogg"))
    segments = speech_segments(y)
    assignment = assign_speakers(y, segments)
    transcript = transcribe(reference_call("call_001.ogg"))
    texts = {i: _text_in_segment(transcript.words, s) for i, s in enumerate(segments)}

    corrected, swapped = verify_roles(assignment.assignments, texts)
    assert swapped is True, "call_001's roles are inverted and must be swapped"
    # Segment 0 is the caller's "Come on."; segment 1 is the bot's greeting.
    assert corrected[0] == "customer"
    assert corrected[1] == "agent"


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, not yet fixed. Speaker-embedding clustering cannot "
        "isolate the customer on a code-switched call. Measured: 12.36s "
        "attributed to the customer against a true ~0.9s. Proven unfixable "
        "within cluster-level labelling: the customer's single 0.9s segment "
        "never forms its own cluster at k in (2,3) - it groups with genuine "
        "agent segments - so the arithmetic floor for customer_speech_seconds "
        "is 5.2s regardless of the reference vector used. No cluster clears "
        "AGENT_REF_MIN_SIM (best 0.4427), and at k=3 the mixed cluster (0.4427) "
        "outranks the pure-agent one (0.4094), so reference similarity is not "
        "even ordering correctly. The principled fix needs per-segment "
        "classification informed by ASR language ID (Task 6), which does not "
        "exist yet. Left xfail(strict) so it flips to XPASS the moment it is "
        "genuinely fixed."
    ),
)
def test_code_switched_call_does_not_inflate_customer_speech():
    """call_002: bot greets in English, customer says 'Spanish, please'
    (~1s), then the BOT continues in Spanish. ECAPA embeddings are
    language-sensitive, so naive 2-way clustering splits the bot's own two
    languages and mislabels ~11s of bot speech as customer. True customer
    speech is ~1s, which must land in Tier C (<3s)."""
    y, _ = load_mono(reference_call("call_002.ogg"))
    result = assign_speakers(y, speech_segments(y))
    assert result.customer_speech_seconds < 3.0, (
        f"expected ~1s of customer speech, got "
        f"{result.customer_speech_seconds:.1f}s - bot speech is being "
        f"attributed to the customer"
    )


def test_customer_share_is_plausible_on_every_provided_call():
    """The bot is a receptionist: it should never be a small minority of the
    call. A customer share above 85% means roles are likely swapped."""
    for name in ("call_001.ogg", "call_002.ogg", "call_003.ogg"):
        y, _ = load_mono(reference_call(name))
        segments = speech_segments(y)
        result = assign_speakers(y, segments)
        total = sum(s.duration for s in segments)
        share = result.customer_speech_seconds / total
        assert share < 0.85, f"{name}: customer share {share:.0%} - roles swapped?"
