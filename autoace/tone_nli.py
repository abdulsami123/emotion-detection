"""Local zero-shot NLI tone classifier.

This is the second, materially different approach the brief requires
alongside the Haiku LLM classifier (autoace.tone_llm). Two roles:

  1. Satisfies the "compare two materially different approaches" requirement
     - this is a local zero-shot entailment model (facebook/bart-large-mnli),
     not an LLM completion, so its errors are decorrelated from the LLM's
     rather than being a second sample from the same failure modes.
  2. Doubles as the no-API-key fallback and a vote in the confidence
     ensemble (see autoace.config CONF_TONE_AGREE) - if the LLM and this
     classifier agree on the label, that raises the ensemble's confidence.

Text-only, unlike the LLM classifier: this module never sees prosody or
SER, so on its own it reproduces the exact failure the LLM prompt was
written to avoid (call_003's words alone read as frustrated, not the
ground-truth satisfied). It is one voter among several, never a standalone
classifier.
"""

from __future__ import annotations

from functools import lru_cache

from autoace.config import NLI_MODEL
from autoace.schema import EmotionalTone

_HYPOTHESIS_TEMPLATE = "This caller sounds {}."

# Hypothesis phrasing per tone - kept close to the schema's own definitions
# (autoace.tone_llm.LABEL_DEFINITIONS) so the two classifiers are judging the
# same boundaries, just via different mechanisms.
_LABEL_HYPOTHESES: dict[str, str] = {
    EmotionalTone.NEUTRAL.value: "neutral, with no clear positive or negative emotion",
    EmotionalTone.SATISFIED.value: "satisfied, pleased, or appreciative",
    EmotionalTone.FRUSTRATED.value: "frustrated, annoyed, or impatient",
    EmotionalTone.UPSET.value: "upset, clearly angry, or agitated",
    EmotionalTone.DISTRESSED.value: "distressed, overwhelmed, or panicked",
}

_TONE_VALUES = [t.value for t in EmotionalTone]


@lru_cache(maxsize=1)
def _load_classifier():
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=NLI_MODEL)


def classify_tone_nli(transcript: str) -> dict[str, float]:
    """Zero-shot tone distribution over the five schema tones.

    Returns a normalised distribution (sums to ~1.0) keyed by EmotionalTone
    value. An empty or whitespace-only transcript (no ASR text - e.g. a
    near-silent call, or the ~2-word call_002) carries no lexical signal at
    all, so it returns the uniform distribution rather than an arbitrary
    guess dressed up as a confident score.
    """
    if not transcript or not transcript.strip():
        uniform = 1.0 / len(_TONE_VALUES)
        return {label: uniform for label in _TONE_VALUES}

    classifier = _load_classifier()
    candidate_labels = [_LABEL_HYPOTHESES[label] for label in _TONE_VALUES]
    result = classifier(
        transcript,
        candidate_labels=candidate_labels,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    hypothesis_to_label = {v: k for k, v in _LABEL_HYPOTHESES.items()}
    scores = {
        hypothesis_to_label[label]: float(score)
        for label, score in zip(result["labels"], result["scores"])
    }
    total = sum(scores.values()) or 1.0
    return {label: scores.get(label, 0.0) / total for label in _TONE_VALUES}
