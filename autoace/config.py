"""Every threshold in the system. Nothing numeric belongs anywhere else.

Provenance of each value is marked:
  MEASURED  — derived from the three labelled calls (spec 2.5)
  DERIVED   — reasoned from the brief's definitions or physics
  UNFITTED  — interpolated with no supporting example; highest risk
"""

from pathlib import Path

# ---------------------------------------------------------------- models
WHISPER_MODEL = "large-v3-turbo"
WHISPER_COMPUTE_CPU = "int8"
WHISPER_COMPUTE_GPU = "float16"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SER_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
NLI_MODEL = "facebook/bart-large-mnli"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
LLM_MODEL = "claude-haiku-4-5"
LLM_TEMPERATURE = 0.0

SAMPLE_RATE = 16000

# ------------------------------------------------------------------ VAD
VAD_THRESHOLD = 0.5                    # DERIVED — Silero default
VAD_MIN_SPEECH_MS = 250                # DERIVED
VAD_MIN_SILENCE_MS = 300               # DERIVED

# --------------------------------------------------------- diarization
STEREO_SEPARATE_MAX_CORR = 0.98        # DERIVED — above this, duplicated mono
AGENT_REF_MIN_SIM = 0.55               # DERIVED — below, fall back to first-speaker
DIARIZATION_MIN_SILHOUETTE = 0.10      # DERIVED — below, mark degraded

# --------------------------------------------------------------- noise
# MEASURED: the one no-noise call sits at -56.3 dBFS; both medium calls at
# -52.1 and -47.0. Boundaries between them are UNFITTED.
NOISE_FLOOR_PRESENT = -55.0            # MEASURED (anchor at -56.3 / -52.1)
NOISE_SEVERITY_BANDS = [               # (upper bound dBFS, severity)
    (-55.0, "none"),                   # MEASURED
    (-50.0, "low"),                    # UNFITTED — no `low` example exists
    (-44.0, "medium"),                 # MEASURED (anchors -52.1, -47.0)
    (999.0, "high"),                   # UNFITTED — no `high` example exists
]
TAG_MIN_DOM = 1.8                      # UNFITTED — top group / median group ratio
TYPE_AST_MIN_DOM = 2.5                 # UNFITTED — below this, prefer spectral label

# spectral line-artifact detection
STATIC_FLATNESS_MIN = 0.40             # DERIVED — broadband hiss is spectrally flat
HUM_FREQS_HZ = (50.0, 60.0)            # DERIVED — mains frequencies
HUM_PROMINENCE_DB = 6.0                # DERIVED
CRACKLE_MAX_MS = 20.0                  # DERIVED
CRACKLE_MIN_COUNT = 3                  # DERIVED

# ------------------------------------------------------------- quality
# MEASURED: all three calls are `clear` with 4-6% energy above 3.4 kHz and
# 0.00% clipping. `clear` is the strong prior.
SQUIM_STOI_SLIGHT = 0.75               # UNFITTED — no impaired example exists
SQUIM_STOI_SEVERE = 0.55               # UNFITTED
CLIP_FRACTION_THRESHOLD = 0.001        # DERIVED — 0.1% of samples near full scale
DROPOUT_MIN_MS = 30.0                  # DERIVED
ECHO_LAG_RANGE_MS = (20.0, 200.0)      # DERIVED
ECHO_PEAK_MIN = 0.30                   # DERIVED
# Telephony baseline: do NOT treat narrowband as muffling.
TELEPHONY_HF_BASELINE = 0.04           # MEASURED — 4-6% above 3.4 kHz is normal
MUFFLE_HF_MIN = 0.015                  # DERIVED — well below the telephony floor
LOW_VOLUME_LUFS = -35.0                # DERIVED

# --------------------------------------------------------------- other
OVERLAP_MIN_SEGMENT_S = 0.5            # DERIVED — ignore backchannels
OVERLAP_MIN_TOTAL_S = 1.0              # DERIVED
# MEASURED: call_003 has a 7.35 s non-speech gap and is labelled false.
LONG_SILENCE_SEC = 10.0                # MEASURED — must exceed 7.35

# ----------------------------------------------------------- intensity
TIER_A_MIN_SPEECH_S = 15.0             # DERIVED
TIER_C_MAX_SPEECH_S = 3.0              # DERIVED
BASELINE_WINDOW_S = 25.0               # DERIVED
ACTIVATION_HIGH_Z = 1.0                # UNFITTED
ACTIVATION_LOW_Z = 0.3                 # UNFITTED — no `low` example exists
ACTIVATION_PEAK_Z = 2.0                # UNFITTED
ACTIVATION_WEIGHTS = {                 # UNFITTED — uniform within each channel
    "loudness_range": 0.15,
    "f0_elevation": 0.15,
    "f0_range": 0.15,
    "rate_deviation": 0.10,
    "jitter_shimmer": 0.05,
    "interruption_rate": 0.10,
    "turn_latency": 0.05,
    "lexical_markers": 0.10,
    "ser_arousal": 0.15,
}

# ---------------------------------------------------------- confidence
CONF_BASE = 0.55
CONF_TONE_AGREE = 0.15
CONF_INTENSITY_AGREE = 0.10
CONF_SER_CONSISTENT = 0.10
CONF_LLM_SELF_HIGH = 0.05
CONF_LLM_SELF_HIGH_THRESHOLD = 0.80
CONF_EVIDENCE_CONFLICT = -0.15
CONF_NEAR_THRESHOLD = -0.05
CONF_TIER_C = -0.20
CONF_DEGRADED = -0.15
CONF_ASR_POOR = -0.10
CONF_MIN, CONF_MAX = 0.05, 0.98
TIER_C_MAX_CONF = 0.45
DEGRADED_MAX_CONF = 0.50
NEAR_THRESHOLD_FRACTION = 0.10         # within 10% of a boundary
ASR_MIN_LOGPROB = -1.0                 # DERIVED
REVIEW_THRESHOLD = 0.60                # dashboard review queue

# coherence check
VALENCE_POS_MIN = 0.45                 # UNFITTED
VALENCE_NEG_MAX = 0.55                 # UNFITTED
AROUSAL_HIGH_MIN = 0.50                # UNFITTED

# ------------------------------------------------------------- fixtures
REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "reference"   # provided trial assets, not code we write
LABELS_CSV = REFERENCE_DIR / "labels.csv"


def reference_call(name: str) -> str:
    """Absolute path to a provided trial call.

    The manifest's `name` column holds a BARE filename (fixed by the brief's
    batch format), so audio is always resolved as directory + name. Never
    assume the audio sits in the working directory.
    """
    return str(REFERENCE_DIR / name)
