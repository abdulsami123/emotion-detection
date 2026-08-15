import json

import pytest
from pydantic import ValidationError

from autoace.config import LABELS_CSV
from autoace.schema import CallAnalysis, EmotionalTone, EmotionalIntensity


def test_valid_analysis_round_trips():
    payload = {
        "emotional_tone": "upset",
        "emotional_intensity": "high",
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
        "confidence": 0.82,
    }
    result = CallAnalysis(**payload)
    assert result.emotional_tone == EmotionalTone.UPSET
    assert json.loads(result.model_dump_json()) == payload


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        CallAnalysis(
            emotional_tone="furious",  # not in the enum
            emotional_intensity="high",
            background_noise_present=False,
            background_noise_type="",
            background_noise_severity="none",
            audio_quality="clear",
            speaker_overlap_present=False,
            long_silence_present=False,
            confidence=0.82,
        )


def test_confidence_bounds_enforced():
    base = {
        "emotional_tone": "neutral",
        "emotional_intensity": "low",
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
    }
    with pytest.raises(ValidationError):
        CallAnalysis(**base, confidence=1.5)
    with pytest.raises(ValidationError):
        CallAnalysis(**base, confidence=-0.1)


def test_every_labels_csv_row_validates():
    """The three provided labels must satisfy our schema exactly."""
    import csv
    with open(LABELS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    for row in rows:
        CallAnalysis(**json.loads(row["result_json"]))
