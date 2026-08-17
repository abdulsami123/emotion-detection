import pytest

from autoace.schema import EmotionalTone
from autoace.tone_llm import ToneRequest, build_prompt, parse_response
from autoace.tone_nli import classify_tone_nli


def _request(**overrides):
    base = dict(
        duration_s=30.9,
        customer_speech_s=6.2,
        tier="B",
        language="en",
        ser_line="arousal 0.65 | dominance 0.64 | valence 0.53",
        agent_behavior="greeted; failed to respond to 5 consecutive utterances",
        annotated_lines=['[00:12-00:18] (baseline) "Are you a real person?"'],
        trajectory="rising through 00:27; peak activation z=2.3",
    )
    base.update(overrides)
    return ToneRequest(**base)


def test_prompt_contains_verbatim_label_definitions():
    """The brief's definitions ARE the classifier spec - paraphrasing loses the
    frustrated/upset/distressed boundaries."""
    prompt = build_prompt(_request())
    assert "Frustrated means annoyed, impatient" in prompt
    assert "Distressed means highly emotional" in prompt
    assert "Upset means clearly angry" in prompt


def test_prompt_states_prosody_is_corroborating_for_tone():
    prompt = build_prompt(_request())
    assert "CORROBORATING" in prompt
    assert "loudness alone" in prompt


def test_prompt_carries_the_politeness_masking_rule():
    """call_003 reads as frustrated but is labelled satisfied."""
    prompt = build_prompt(_request())
    assert "thanks the agent is SATISFIED" in prompt


def test_prompt_carries_the_escalating_repetition_rule():
    """call_001 reads as neutral but is labelled upset."""
    assert "escalating repetition" in build_prompt(_request()).lower()


def test_prompt_includes_the_payload_values():
    prompt = build_prompt(_request(language="es", tier="C"))
    assert "es" in prompt
    assert "tier C" in prompt or "C)" in prompt


def test_parse_response_rejects_an_invalid_tone():
    with pytest.raises(ValueError):
        parse_response(
            '{"emotional_tone": "furious", "emotional_intensity": "high", '
            '"self_confidence": 0.9, "reasoning": "x", '
            '"lexical_intensity_markers": [], "agent_failed": false, '
            '"evidence_conflict": false}'
        )


def test_parse_response_accepts_a_valid_payload():
    result = parse_response(
        '{"emotional_tone": "upset", "emotional_intensity": "high", '
        '"self_confidence": 0.9, "reasoning": "escalating repetition", '
        '"lexical_intensity_markers": ["repetition"], "agent_failed": true, '
        '"evidence_conflict": false}'
    )
    assert result.emotional_tone == EmotionalTone.UPSET
    assert result.agent_failed is True
    assert result.self_confidence == 0.9


def test_self_confidence_is_named_distinctly_from_the_schema_field():
    """The schema's `confidence` is computed from voter agreement, not from the
    model rating itself. Distinct naming prevents accidental passthrough."""
    from autoace.tone_llm import ToneResponse
    import dataclasses

    fields = {f.name for f in dataclasses.fields(ToneResponse)}
    assert "self_confidence" in fields
    assert "confidence" not in fields


def test_nli_returns_a_normalised_distribution_over_all_five_tones():
    """The local second approach required by the brief."""
    scores = classify_tone_nli(
        "I have been waiting three weeks and nobody has called me back."
    )
    assert set(scores) == {t.value for t in EmotionalTone}
    assert abs(sum(scores.values()) - 1.0) < 0.01


def test_nli_handles_empty_transcript():
    scores = classify_tone_nli("")
    assert abs(sum(scores.values()) - 1.0) < 0.01
