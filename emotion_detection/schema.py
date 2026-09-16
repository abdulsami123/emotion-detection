"""The output contract. Enum values are fixed by the trial brief and must
not be renamed — the grader matches on these exact strings."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


class CallAnalysis(BaseModel):
    """The 9-field result for one audio clip."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


# Tone groupings used by the coherence check in a later task.
POSITIVE_TONES = {EmotionalTone.SATISFIED}
NEGATIVE_TONES = {
    EmotionalTone.FRUSTRATED,
    EmotionalTone.UPSET,
    EmotionalTone.DISTRESSED,
}
