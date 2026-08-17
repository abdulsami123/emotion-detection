"""Background-noise typing.

The shape of this module is dictated by measurement, and two of the original
design's claims were refuted. Both refutations are recorded here because the
code is otherwise surprising.

REFUTED 1 - "tag non-speech regions only".
  The reasoning was that speech masks background noise, so restricting the
  tagger to non-speech regions should improve typing. Measured on the three
  labelled calls it does the opposite: whole-clip tagging identifies call_002's
  ground-truth "TV" correctly (top AudioSet class `Television`), while
  non-speech-only tagging returns "music". In telephony the non-speech gaps are
  near-SILENT - that is exactly why the noise floor works as a presence
  detector - whereas the television is audible underneath the speech. Removing
  the speech removed the evidence. So AST runs on the whole clip, with speech
  classes suppressed after the fact.

REFUTED 2 - "a spectral classifier catches the transmission noise AudioSet
  cannot see". AudioSet genuinely is blind to it: the `static` group scores
  0.0013-0.0059 on every call including the one whose ground truth IS "sharp
  static". But none of the three proposed spectral discriminators works either,
  measured over the non-speech regions:

    call            truth           flatness   transients/s   crest   kurtosis
    call_001.ogg    (none)            0.0531           0.47   36.48      541.6
    call_002.ogg    TV                0.1157           0.28   12.88       81.0
    call_003.ogg    sharp static      0.0426           0.29   21.00      108.8

  Flatness is LOWEST on the static call (broadband hiss should be flattest).
  Transient rate, crest factor and kurtosis are all HIGHEST on the clean call.
  A mains-hum detector at 6 dB prominence fired on all three. No threshold
  separates the static call in the correct direction on any of these features,
  so `spectral_artifact` is disabled rather than shipped as a detector that
  fires arbitrarily. Finding the right feature needs labelled examples of
  line noise, which the three provided calls do not supply.

Net position: `background_noise_present` and `_severity` come from the noise
floor (measured, monotonic with the labels - see acoustics/quality), and only
`_type` depends on this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from autoace.config import AST_MODEL, SAMPLE_RATE
from autoace.vad import Segment

# AST's native input is 10.24s (1024 mel frames at 10ms). Shorter windows get
# padded to that length, which wastes compute and shifts the input away from
# the training distribution. An earlier draft used 1.0s/0.5s and needed ~97
# forward passes per call against ~10 here.
_WINDOW_S = 10.0
_HOP_S = 5.0
_MIN_TAGGABLE_S = 2.0

# AudioSet class name -> the brief's short informal vocabulary. The labeller
# writes "TV", not "television" - keep outputs short and conventional.
AUDIOSET_TO_LABEL: dict[str, list[str]] = {
    "office chatter": [
        "Hubbub, speech noise, speech babble",
        "Chatter",
        "Crowd",
        "Babbling",
    ],
    "music": ["Music", "Musical instrument", "Singing"],
    "road noise": ["Vehicle", "Car", "Traffic noise, roadway noise", "Engine"],
    "TV": ["Television"],
    "radio": ["Radio"],
    "keyboard typing": ["Typing", "Computer keyboard", "Typewriter"],
    "wind": ["Wind", "Wind noise (microphone)", "Rustling leaves"],
    "mechanical noise": [
        "Mechanisms",
        "Machine",
        "Motor vehicle (road)",
        "Engine knocking",
    ],
}

SPEECH_CLASSES = {
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Speech synthesizer",
    "Narration, monologue",
}


@dataclass
class TagResult:
    dominant_group: str | None
    probability: float
    relative_dominance: float
    group_probabilities: dict[str, float] = field(default_factory=dict)
    top_class: str = ""


@lru_cache(maxsize=1)
def _load_ast():
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL)
    model.eval()
    return extractor, model


def tag_audio(y: np.ndarray) -> TagResult:
    """Tag the whole clip with AST, suppressing speech classes afterwards.

    Whole-clip rather than non-speech-only: see REFUTED 1 in the module
    docstring. Reported confidence is `relative_dominance` (top group divided
    by median group), not absolute probability - absolute AudioSet
    probabilities are uncalibrated on telephony audio, spanning only
    0.001-0.144 across the labelled calls, so an absolute gate would classify
    every clip as noise-free.
    """
    extractor, model = _load_ast()
    window = int(_WINDOW_S * SAMPLE_RATE)
    hop = int(_HOP_S * SAMPLE_RATE)

    if len(y) < _MIN_TAGGABLE_S * SAMPLE_RATE:
        return TagResult(None, 0.0, 0.0)

    audio = y if len(y) >= window else np.pad(y, (0, window - len(y)))

    logit_sum = None
    count = 0
    for start in range(0, len(audio) - window + 1, hop):
        inputs = extractor(
            audio[start : start + window],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        logit_sum = logits if logit_sum is None else logit_sum + logits
        count += 1

    probs = torch.sigmoid(logit_sum / count).numpy()
    id2label = model.config.id2label

    per_class = {
        id2label[i]: float(p)
        for i, p in enumerate(probs)
        if id2label[i] not in SPEECH_CLASSES
    }
    group_probs = {
        group: max((per_class.get(name, 0.0) for name in names), default=0.0)
        for group, names in AUDIOSET_TO_LABEL.items()
    }

    dominant = max(group_probs, key=group_probs.get)
    top_probability = group_probs[dominant]
    median = float(np.median(list(group_probs.values()))) or 1e-9
    top_class = max(per_class, key=per_class.get) if per_class else ""

    return TagResult(
        dominant_group=dominant,
        probability=top_probability,
        relative_dominance=top_probability / median,
        group_probabilities=group_probs,
        top_class=top_class,
    )


def spectral_artifact(y: np.ndarray, segments: list[Segment]) -> None:
    """Transmission-noise detection: DISABLED, deliberately.

    See REFUTED 2 in the module docstring for the measurements. Flatness,
    transient rate, crest factor and kurtosis all order the three labelled
    calls in the wrong direction, and a mains-hum test fired on all three.
    Shipping any of them would emit an arbitrary label rather than an
    informative one.

    Kept as a named seam so the call sites and the intent stay visible, and so
    a validated detector can be dropped in when labelled line-noise examples
    exist. Always returns None; the signature is intentionally typed to say so.
    """
    return None


def classify_noise_type(y: np.ndarray, segments: list[Segment]) -> str:
    """The `background_noise_type` string.

    Environmental typing from AST over the whole clip. Transmission noise
    (static, hum, crackle) is currently undetectable here - a call whose real
    noise is a line artifact will receive the nearest environmental label
    instead, which is a known and documented inaccuracy.
    """
    artifact = spectral_artifact(y, segments)
    if artifact is not None:
        return artifact
    return tag_audio(y).dominant_group or ""
