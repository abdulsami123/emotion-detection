"""LLM tone classifier for the human caller in a voice-agent call.

Runs against OpenAI `gpt-4o-mini` (see autoace.config.LLM_MODEL). Only
`classify_tone` is vendor-specific: `build_prompt` and `parse_response` are
provider-agnostic on purpose, so changing provider is a one-function edit and
the prompt design - which is the hard-won part - travels unchanged. It was
previously written against Anthropic Haiku 4.5 and the swap touched exactly
that one function.

Why the prompt is shaped the way it is
---------------------------------------
These are AI-voice-agent calls: a TTS bot ("Erica", a Toyota dealership
receptionist) talking to human callers. Text alone scores at or below chance
on the three labelled calls:

  call_001  upset / high        "Are you a real person?" then "Hello?" x5 -
                                 ~8 words. Reads as neutral or confused; the
                                 truth is escalating anger from being ignored.
  call_002  neutral / medium    "Spanish, please." - two words, no signal.
  call_003  satisfied / medium  Long call, repeatedly told the dealership is
                                 closed, dates confused, bounced to an
                                 advisor. Reads as FRUSTRATED. The caller
                                 stays warm and closes with "Thank you" - the
                                 ground truth is satisfied.

call_003 is decisive: the transcript supplies the *situation* (blocked,
bounced around), not the caller's *sentiment*. Prosody is primary; word
choice corroborates. The prompt below encodes both failure directions -
politeness masking (003 reads frustrated, is satisfied) and escalating
repetition (001 reads neutral, is upset) - because a prompt that only
encodes one direction gets the other call wrong.

SER axes are measured as COMPRESSED on this telephony audio: valence spans
only 0.534-0.638 and arousal only 0.594-0.646 across the three labelled
calls (see autoace/config.py VALENCE_POS_MIN etc. for the same finding used
elsewhere in the pipeline). The prompt must not imply the model will see
values near 0 or 1 - relative comparison matters more than absolute level.

Why the few-shot examples are synthetic, not the labelled calls
-----------------------------------------------------------------
The three labelled calls (call_001/002/003) are the ENTIRE validation set
for this pipeline. Using them as few-shot examples would let the model
simply reproduce cases it was shown, destroying the regression test. The
FEW_SHOT examples below are therefore invented calls that illustrate the
same schema boundaries (politeness masking, escalating repetition, too
little speech to judge) without overlapping the eval set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from autoace.config import LLM_MAX_OUTPUT_TOKENS, LLM_MODEL, LLM_TEMPERATURE
from autoace.schema import EmotionalIntensity, EmotionalTone

# --------------------------------------------------------------- definitions
# Verbatim from the brief. Tests assert on these exact substrings - the
# definitions ARE the classifier spec, and paraphrasing loses the
# frustrated/upset/distressed boundaries the brief drew deliberately.
LABEL_DEFINITIONS: dict[str, str] = {
    "neutral": "Neutral means no clear positive or negative emotion.",
    "satisfied": "Satisfied means pleased, relieved, appreciative, or clearly positive.",
    "frustrated": (
        "Frustrated means annoyed, impatient, or dissatisfied without "
        "strong anger or distress."
    ),
    "upset": "Upset means clearly angry, agitated, or strongly dissatisfied.",
    "distressed": (
        "Distressed means highly emotional, overwhelmed, panicked, crying, "
        "or otherwise emotionally escalated."
    ),
}

INTENSITY_DEFINITIONS = (
    "Low is subtle or mild. Medium is clear and sustained. High is strong, "
    "escalated, or likely to require attention."
)

_LABEL_DEFINITIONS_BLOCK = "\n".join(
    f"- {LABEL_DEFINITIONS[label]}" for label in ("neutral", "satisfied", "frustrated", "upset", "distressed")
)

# ------------------------------------------------------------ system prompt
SYSTEM_PROMPT = f"""\
You are an expert call-audio analyst. Your job is to classify the emotional
tone and intensity of the HUMAN CALLER on one voice-agent phone call. Ignore
the tone of the automated agent entirely - it is a text-to-speech bot with
no emotion of its own; only the human caller's state is being classified.

You do not receive audio. You receive acoustic measurements (a dimensional
speech-emotion-recognition line, a prosody trajectory) and an annotated
transcript of the caller's utterances, each tagged with delivery notes
(loudness, pitch, pace) at the moment it was spoken. Treat the acoustic
evidence as PRIMARY evidence of tone, not decoration on top of the words.

Tone label definitions (use these exact boundaries):
{_LABEL_DEFINITIONS_BLOCK}

Intensity label definitions:
{INTENSITY_DEFINITIONS}

Evidence weighting:
- SER valence and dominance, and prosody (loudness, pitch, pace, energy
  trajectory), are PRIMARY evidence for TONE. Word choice is CORROBORATING
  evidence only - it can support or override a prosodic read, but never
  invent one on its own. Never infer frustration, upset, or distress from
  loudness alone - loud does not automatically mean angry, and a loud caller
  reading as clearly positive prosody is still satisfied.
- Prosody and SER arousal are PRIMARY evidence for INTENSITY. Intensity is
  independent of tone: a call can be low-intensity-satisfied or
  high-intensity-satisfied, low-intensity-upset or high-intensity-upset.
- Distinguishing upset from distressed: high dominance with high arousal
  reads as UPSET (angry, assertive, in control of the interaction). Low
  dominance with high arousal reads as DISTRESSED (overwhelmed, not in
  control, possibly panicked or crying).

Domain rules (both directions matter - a prompt that only encodes one of
these gets the other kind of call wrong):
- Politeness formulas ("please", "thank you", formal register) are WEAK
  evidence on their own - callers are often polite while frustrated. But the
  combination is a strong signal: a caller who stays warm and thanks the agent is SATISFIED
  even if the agent failed to fulfil their request. Do not read repeated
  blocking or bad news from the agent as evidence of the caller's tone; read
  the caller's own delivery and closing instead.
- Conversely, escalating repetition of a short utterance with rising energy
  (louder, faster, shorter gaps between repeats) indicates UPSET even when
  the words themselves are neutral or the caller says almost nothing else.
  Do not read a short transcript as low-signal neutral if the delivery is
  escalating.
- Sarcasm inverts lexical polarity - positive words delivered with flat or
  hostile prosody are not evidence of satisfaction.
- Under about 3 seconds of caller speech is insufficient evidence for a
  confident call. Do not force a label past what the evidence supports;
  report the uncertainty via a low self_confidence value instead.

A note on scale: SER values on this audio arrive in a compressed range
(roughly 0.5-0.7), not spread across the full [0, 1] the model nominally
emits. Compare values RELATIVELY (higher/lower than a typical call) rather
than against absolute thresholds near 0 or 1.

Output instruction: return only the JSON object matching the schema. If the
acoustic evidence and the lexical evidence point in different directions,
set evidence_conflict to true and lower self_confidence rather than forcing
a single label to look more certain than it is.
"""

# ----------------------------------------------------------------- few-shot
# These three examples are DELIBERATELY SYNTHETIC, not drawn from
# call_001/002/003. Those three labelled calls are the entire validation set
# for this pipeline; using them as few-shot examples would let the model
# reproduce memorized cases instead of applying the rule, which would
# destroy the regression test. Each example below illustrates one schema
# boundary from the domain rules above without overlapping the eval set.
FEW_SHOT = """\
Example A - polite but blocked (illustrates: politeness overrides a
frustrating situation):
  Caller: "Oh, okay, no problem, I understand. Could you maybe have someone
  call me back tomorrow? That would be great, thank you so much!"
  Delivery: warm, steady pace, no rising energy, closes with a genuine thanks.
  -> emotional_tone: satisfied, emotional_intensity: low. The caller was told
  no but stayed warm throughout and closed positively.

Example B - short neutral words, escalating delivery (illustrates:
escalating repetition overrides lexically neutral content):
  Caller: "Hello?" (baseline) ... "Hello?" (louder) ... "Hello? Hello?"
  (much louder, rapid, clipped).
  Delivery: loudness and pitch rise sharply across three repeats in under 20
  seconds; gaps between repeats shrink.
  -> emotional_tone: upset, emotional_intensity: high. The words carry no
  sentiment at all, but the escalating delivery is the signal.

Example C - almost no caller speech (illustrates: insufficient evidence ->
low self_confidence, not a forced label):
  Caller: "Yeah." (that is the entire transcript; customer_speech_s: 1.1)
  -> emotional_tone: neutral, emotional_intensity: low, self_confidence:
  0.25, evidence_conflict: false. There is not enough signal to say more;
  the low confidence communicates that, rather than guessing at a stronger
  label.
"""

# --------------------------------------------------------------- schema
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "emotional_tone": {
            "type": "string",
            "enum": [t.value for t in EmotionalTone],
        },
        "emotional_intensity": {
            "type": "string",
            "enum": [i.value for i in EmotionalIntensity],
        },
        "self_confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "lexical_intensity_markers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "agent_failed": {"type": "boolean"},
        "evidence_conflict": {"type": "boolean"},
    },
    "required": [
        "emotional_tone",
        "emotional_intensity",
        "self_confidence",
        "reasoning",
        "lexical_intensity_markers",
        "agent_failed",
        "evidence_conflict",
    ],
    "additionalProperties": False,
}


@dataclass
class ToneRequest:
    duration_s: float
    customer_speech_s: float
    tier: str
    language: str
    ser_line: str
    agent_behavior: str
    annotated_lines: list[str]
    trajectory: str
    interruptions: int = 0


@dataclass
class ToneResponse:
    """The seven fields the LLM produces.

    Named `self_confidence`, never `confidence`: the schema's `confidence`
    field (autoace.schema.CallAnalysis.confidence) is computed downstream
    from voter agreement across the LLM and NLI classifiers, not from the
    model's own self-rating. Sharing a name would invite an accidental
    passthrough of the model's opinion of itself in place of the ensemble's
    measured agreement.
    """

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    self_confidence: float
    reasoning: str
    lexical_intensity_markers: list[str] = field(default_factory=list)
    agent_failed: bool = False
    evidence_conflict: bool = False


def build_prompt(request: ToneRequest) -> str:
    """Render the complete prompt text for one call, as a single string.

    Deliberately self-contained (system content + few-shot + payload) so it
    is fully testable without an API key or network access. `classify_tone`
    sends this string as the `system` parameter of the Messages API call.
    """
    payload = f"""\
Call under review:
- Tier: {request.tier} (this is a tier {request.tier} call)
- Language: {request.language}
- Duration: {request.duration_s:.1f}s total, {request.customer_speech_s:.1f}s of caller speech
- Caller interruptions of the agent: {request.interruptions}
- SER line: {request.ser_line}
- Agent behavior this call: {request.agent_behavior}
- Prosody trajectory: {request.trajectory}
- Annotated caller transcript:
{chr(10).join(request.annotated_lines) if request.annotated_lines else "(no caller speech captured)"}
"""
    return "\n\n".join([SYSTEM_PROMPT, FEW_SHOT, payload])


def parse_response(raw: str) -> ToneResponse:
    """Parse and validate one JSON response from the model.

    Raises ValueError if `emotional_tone` or `emotional_intensity` is not one
    of the schema's enum values, so an invalid label can never reach the
    output schema - constructing the str Enum from an unrecognised value
    raises ValueError on its own, which is exactly the failure mode we want.
    """
    data = json.loads(raw)
    return ToneResponse(
        emotional_tone=EmotionalTone(data["emotional_tone"]),
        emotional_intensity=EmotionalIntensity(data["emotional_intensity"]),
        self_confidence=float(data["self_confidence"]),
        reasoning=data["reasoning"],
        lexical_intensity_markers=list(data.get("lexical_intensity_markers", [])),
        agent_failed=bool(data.get("agent_failed", False)),
        evidence_conflict=bool(data.get("evidence_conflict", False)),
    )


def _system_content() -> str:
    """The fixed system side of the prompt: definitions plus few-shots.

    Roughly 2500 of the measured 2712 input tokens, and byte-identical across
    every call. OpenAI caches prompt prefixes over 1024 tokens automatically at
    half price, so keeping this prefix stable and placing the per-call payload
    in the user message is what makes that discount reachable. (The Haiku tier
    this replaced had a 4096-token cache minimum, which this prefix never met -
    so caching is a genuine gain from the provider switch, not a wash.)
    """
    return SYSTEM_PROMPT + "\n" + FEW_SHOT


def classify_tone(request: ToneRequest) -> ToneResponse:
    """Classify one call's caller tone and intensity with an OpenAI model.

    Uses `gpt-4o-mini` via strict structured outputs. The prompt assembly in
    `build_prompt` and the validation in `parse_response` are provider-agnostic
    by design, so this function is the only place the vendor appears - swapping
    providers again means editing this one call, not the prompt.

    `temperature=0` is deliberate and is why a non-reasoning tier was chosen:
    the gpt-5-* models do not honour it, and determinism is a reproducibility
    requirement here - a consistent label matters as much as an accurate one
    when a grader re-runs the same batch.

    `strict: True` on the schema makes the enum constraints binding at the API
    level, so an out-of-enum tone cannot come back at all. `parse_response`
    still validates, because defence at the boundary is cheap and the schema
    could be relaxed by accident.

    Privacy boundary: the customer-side transcript and derived acoustic
    features (SER triple, prosody tags, trajectory, agent-behaviour summary)
    leave AutoAce infrastructure in this call. Raw audio never does - only
    text-derived measurements are sent.
    """
    from openai import OpenAI

    client = OpenAI()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tone_analysis",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": _system_content()},
            {"role": "user", "content": build_prompt(request)},
        ],
    )
    return parse_response(completion.choices[0].message.content)
