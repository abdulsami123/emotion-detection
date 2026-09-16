"""Dimensional speech emotion recognition.

SER_MODEL (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) is trained
on MSP-Podcast - naturalistic speech, not an acted studio corpus - which is
why it was chosen over categorical SER for this telephony audio.

Output axis order - VERIFIED, not assumed:
  1. `model.config.id2label` on the downloaded checkpoint reads
     {0: "arousal", 1: "dominance", 2: "valence"}, and `label2id` agrees.
  2. The model card's own usage example labels its printed output columns
     "Arousal    dominance valence" in that order.
  3. Empirically on the three reference calls, raw output index 2 (valence)
     is higher on call_003 (labelled `satisfied`) than on call_001 (labelled
     `upset`), and index 0 (arousal) is higher on call_001 than call_003 -
     both consistent with [arousal, dominance, valence].
All three sources agree, so the output is consumed in that order with no
reordering.

The checkpoint ships only config.json + weights, no custom modeling code, and
its `architectures` field names `Wav2Vec2ForSpeechClassification`, which
`AutoModel` does not recognise. The `_EmotionModel` class below - a mean-
pooled Wav2Vec2Model plus a small regression head - is copied from the model
card's own usage example, the standard way this checkpoint is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from emotion_detection.config import SAMPLE_RATE, SER_MAX_WINDOWS, SER_MODEL, SER_WINDOW_S


@dataclass(frozen=True)
class Dimensions:
    """Arousal / dominance / valence in [0, 1]. Always consumed as a
    distribution, never reduced to an argmax."""

    arousal: float
    dominance: float
    valence: float

    def as_prompt_line(self) -> str:
        return (
            f"arousal {self.arousal:.2f} | "
            f"dominance {self.dominance:.2f} | "
            f"valence {self.valence:.2f}"
        )


class _RegressionHead(nn.Module):
    """Per-dimension regression head, as shipped by the model author."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """Mean-pooled Wav2Vec2Model + regression head -> [arousal, dominance,
    valence]. See the module docstring for why this hand-rolled class is
    needed instead of an Auto* class."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)


@lru_cache(maxsize=1)
def _load_model():
    processor = AutoProcessor.from_pretrained(SER_MODEL)
    model = _EmotionModel.from_pretrained(SER_MODEL)
    model.eval()
    return processor, model


def predict_dimensions(y: np.ndarray) -> Dimensions:
    """Predict call-level arousal/dominance/valence.

    Cost scales with call duration unless bounded: wav2vec2-large is a
    300M-parameter model, and running it over a whole multi-minute call risks
    the same O(duration) blow-up measured on SQUIM (28.6s/audio-minute) and
    AST (30.9s/audio-minute) elsewhere in this pipeline. Emotion is a
    call-level summary here, not a frame-level signal, so - as with those two
    - a fixed number of evenly-spaced SER_WINDOW_S windows (capped at
    SER_MAX_WINDOWS) is sampled and averaged instead of scanning the whole
    signal, making the cost O(1) per call.
    """
    processor, model = _load_model()
    window = int(SER_WINDOW_S * SAMPLE_RATE)

    audio = y if len(y) >= window else np.pad(y, (0, window - len(y)))

    starts = list(range(0, len(audio) - window + 1, window))
    if not starts:
        starts = [0]
    if len(starts) > SER_MAX_WINDOWS:
        picks = np.linspace(0, len(starts) - 1, SER_MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks)]

    outputs = []
    for start in starts:
        chunk = audio[start : start + window]
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).squeeze(0)
        outputs.append(logits.numpy())

    mean = np.mean(outputs, axis=0)
    arousal, dominance, valence = (float(np.clip(v, 0.0, 1.0)) for v in mean)
    return Dimensions(arousal=arousal, dominance=dominance, valence=valence)
