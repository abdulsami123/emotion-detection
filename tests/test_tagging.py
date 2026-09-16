"""Background-noise typing.

Two of the original design claims were refuted by measurement; see the
`emotion_detection/tagging.py` module docstring. These tests pin what actually holds,
and mark what does not with strict xfail so it cannot quietly pass unnoticed.
"""

import numpy as np
import pytest

from emotion_detection.config import reference_call
from emotion_detection.io_audio import load_mono
from emotion_detection.tagging import classify_noise_type, spectral_artifact, tag_audio
from emotion_detection.vad import non_speech_segments


@pytest.fixture(scope="module")
def call_001():
    return load_mono(reference_call("call_001.ogg"))[0]


@pytest.fixture(scope="module")
def call_002():
    return load_mono(reference_call("call_002.ogg"))[0]


@pytest.fixture(scope="module")
def call_003():
    return load_mono(reference_call("call_003.ogg"))[0]


def test_ast_identifies_television_on_call_002(call_002):
    """Ground truth is 'TV'. Whole-clip AST maps to it correctly, where PANNs
    said 'Radio' and non-speech-only tagging says 'music'.

    Asserted on the mapped GROUP, not the top raw class: the highest raw
    non-speech class is `Telephone` (measured), which is true but useless -
    every clip here is a phone call. The group mapping is what the schema
    field consumes."""
    result = tag_audio(call_002)
    assert result.dominant_group == "TV"
    assert result.group_probabilities["TV"] > result.group_probabilities["music"]


def test_relative_dominance_separates_clean_from_noisy(call_001, call_002, call_003):
    """Measured relative dominance: 3.7 on the clean call against 47.3 and 8.6
    on the two noisy ones. Absolute probabilities cannot do this (all sit
    between 0.006 and 0.023), so this is the usable secondary presence signal
    alongside the noise floor."""
    clean = tag_audio(call_001).relative_dominance
    for noisy in (tag_audio(call_002), tag_audio(call_003)):
        assert noisy.relative_dominance > clean


def test_dominance_is_relative_not_absolute(call_002):
    """Absolute AudioSet probabilities span only 0.001-0.144 on telephony
    audio, so gating on them would call every clip noise-free. Dominance is
    top-group / median-group instead."""
    result = tag_audio(call_002)
    assert result.relative_dominance > 1.0
    assert result.probability < 0.2, (
        "absolute probability is expected to be tiny here - if it is large, "
        "the calibration assumption behind relative dominance has changed"
    )


def test_speech_classes_are_suppressed(call_003):
    """The clip is mostly speech, so an unsuppressed tagger would return a
    speech class and never surface the background at all."""
    result = tag_audio(call_003)
    assert "speech" not in result.top_class.lower()
    assert "conversation" not in result.top_class.lower()


def test_short_audio_yields_no_tag():
    """Under the minimum taggable duration there is nothing to judge, and a
    confident-looking label would be worse than none."""
    result = tag_audio(np.zeros(int(0.5 * 16000), dtype=np.float32))
    assert result.dominant_group is None
    assert result.relative_dominance == 0.0


def test_spectral_artifact_is_disabled_and_never_false_fires(call_002, call_003):
    """All three proposed transmission-noise discriminators were refuted, and
    a mains-hum test fired on all three calls. Disabled rather than shipped:
    a detector that fires arbitrarily is worse than no detector. This test
    fails if someone re-enables it without validating it first."""
    for audio in (call_002, call_003):
        assert spectral_artifact(audio, non_speech_segments(audio)) is None


def test_classify_returns_a_label_for_a_noisy_call(call_002):
    assert classify_noise_type(call_002, non_speech_segments(call_002)) == "TV"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION. call_003's ground-truth type is 'sharp static', a "
        "line/codec artifact. AudioSet cannot see it (the `static` group scores "
        "0.0013-0.0059 on every call, including this one), and no spectral "
        "discriminator tested separates it in the correct direction: over the "
        "non-speech regions, flatness is 0.0426 here vs 0.0531 on the CLEAN "
        "call (lowest, should be highest), and transient rate 0.29/s vs 0.47/s "
        "on the clean call (also inverted). Crest factor and kurtosis invert "
        "too. Fixing this needs labelled line-noise examples, which the three "
        "provided calls do not supply. Strict xfail so it flips to XPASS the "
        "moment a real discriminator lands."
    ),
)
def test_static_is_typed_correctly_on_call_003(call_003):
    label = classify_noise_type(call_003, non_speech_segments(call_003))
    assert "static" in label.lower()
