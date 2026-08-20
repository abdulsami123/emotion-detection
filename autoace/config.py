"""Every threshold in the system. Nothing numeric belongs anywhere else.

Provenance of each value is marked:
  MEASURED  — derived from the three labelled calls (spec 2.5)
  DERIVED   — reasoned from the brief's definitions or physics
  UNFITTED  — interpolated with no supporting example; highest risk
"""

import os
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
# Tone classifier. OpenAI rather than Anthropic per the trial owner's choice.
# gpt-4o-mini is the closest analogue to the Haiku tier this replaced: cheap,
# fast, supports strict structured outputs, and still accepts temperature=0.
# The gpt-5-* reasoning tiers do NOT honour temperature, and determinism is a
# reproducibility requirement here - a consistent label matters as much as an
# accurate one when the grader re-runs the batch.
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"
# Upgrade candidate, ~3x the input cost. Both got call_001 right in a live
# smoke test (upset/high, matching ground truth); gpt-4.1-mini returned a
# longer rationale and higher self-confidence. Worth an A/B on a larger
# labelled set, not decidable on one call.
LLM_MODEL_UPGRADE = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_OUTPUT_TOKENS = 1024

# MEASURED 2026-08-17 against the live API, not estimated. The full prompt
# (system + few-shot + payload) is 2712 input tokens on a 30.9s call, against
# an earlier ESTIMATE of 830 - a 3.3x error. Roughly 2500 of those are the
# fixed system+few-shot prefix, so cost per AUDIO MINUTE is dominated by fixed
# prompt overhead on short calls and amortises on long ones.
# Output measured at 124 tokens (gpt-4o-mini) / 205 (gpt-4.1-mini).
LLM_MEASURED_INPUT_TOKENS = 2712
LLM_MEASURED_OUTPUT_TOKENS = 124

SAMPLE_RATE = 16000

# ------------------------------------------------------------------- SER
SER_WINDOW_S = 6.0                     # DERIVED — long enough for stable arousal/valence
SER_MAX_WINDOWS = 4                    # DERIVED — caps cost at O(1) per call

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
NOISE_SEVERITY_BANDS = [               # (inclusive upper bound, severity)
    (-55.0, "none"),                   # MEASURED — the one `none` call is -56.3
    (-53.0, "low"),                    # UNFITTED — no `low` example exists, but
                                       #   the bound MUST sit below -52.1 or the
                                       #   lower `medium` anchor falls into `low`.
                                       #   An earlier -50.0 did exactly that and
                                       #   mislabelled call_002 as `low`.
    (-44.0, "medium"),                 # MEASURED — anchors -52.1 and -47.0
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
# MEASURED: longest true-digital-silence run on any of the three (all `clear`)
# calls is 165 ms (call_003, a single occurrence — ordinary silence-suppressed
# pause, not a dropout); call_001's longest is 38.9 ms, call_002's is 82.6 ms.
# 30 ms fired on every call (natural inter-word pauses routinely exceed it).
# Set comfortably above the largest observed natural gap so only sustained
# dead air (network packet loss, not a conversational pause) fires.
DROPOUT_MIN_MS = 200.0                 # MEASURED (largest natural gap 165 ms)
ECHO_LAG_RANGE_MS = (20.0, 200.0)      # DERIVED
ECHO_PEAK_MIN = 0.30                   # DERIVED
# Telephony baseline: do NOT treat narrowband as muffling.
TELEPHONY_HF_BASELINE = 0.04           # MEASURED — 4-6% above 3.4 kHz is normal
MUFFLE_HF_MIN = 0.015                  # DERIVED — well below the telephony floor
LOW_VOLUME_LUFS = -35.0                # DERIVED

# --------------------------------------------------------------- other
# The pyannote overlapped-speech-detection pipeline this was originally
# sized for is gated on HuggingFace and unavailable here (see OVERLAP_* below
# for the ECAPA-based replacement that shipped instead). Confirmed via grep
# that nothing else in the codebase references OVERLAP_MIN_SEGMENT_S /
# OVERLAP_MIN_TOTAL_S, so they are removed rather than kept as dead weight —
# this comment is the record of that decision.
# MEASURED: call_003 has a 7.35 s non-speech gap and is labelled false.
LONG_SILENCE_SEC = 10.0                # MEASURED — must exceed 7.35

# --------------------------------------------------------------- overlap
# Speaker change WITHIN one VAD segment implies two concurrent voices.
# pyannote's overlap model is gated on HuggingFace and unavailable, so this
# reuses the ECAPA encoder already loaded for diarization.
OVERLAP_SUBWINDOW_S = 0.8              # DERIVED — long enough for stable ECAPA
OVERLAP_HOP_S = 0.4                    # DERIVED
OVERLAP_MIN_SEGMENT_S = 1.6            # DERIVED — need >=2 sub-windows
OVERLAP_CHANGE_COS = 0.5               # DERIVED — below this is a speaker change
# MEASURED: rate of sub-window pairs below OVERLAP_CHANGE_COS is
#   0.063 (call_001, overlap=false) / 0.224 (call_003, true) / 0.458 (call_002, true)
OVERLAP_RATE_MIN = 0.15                # MEASURED — sits between 0.063 and 0.224

# ----------------------------------------------------------- intensity
TIER_A_MIN_SPEECH_S = 15.0             # DERIVED
TIER_C_MAX_SPEECH_S = 3.0              # DERIVED
BASELINE_WINDOW_S = 25.0               # DERIVED
# MEASURED 2026-08-17: when the baseline window consumes most of the customer
# speech it is scoring, the z-scores are degenerate BY CONSTRUCTION - a set
# z-scored against its own mean has mean exactly zero, which is arithmetic, not
# signal. Measured baseline coverage and resulting mean activation:
#     call_001  7.1s customer speech -> 100% consumed -> z = -0.000
#     call_002 12.4s customer speech -> 100% consumed -> z = +0.000
#     call_003 73.8s customer speech ->  37% consumed -> z = +0.240  (usable)
# Zero activation trips ACTIVATION_LOW_Z and forces `low`, which is why
# intensity scored 1/3 - BELOW the 2/3 constant-`medium` baseline. Above this
# coverage there is no measurement, so the call is routed to the Tier C path
# (emit `medium`, cap confidence) rather than reporting a manufactured `low`.
BASELINE_MAX_COVERAGE = 0.80           # MEASURED — 0.37 usable vs 1.00 degenerate
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
# MEASURED 2026-08-17: distinct from "the two tone voters disagreed". A missing
# voter is not the same as two conflicting ones, and tone_voters_agree=False
# only WITHHOLDS a bonus - it applies no penalty. On call_001 with the Haiku key
# invalid, the NLI-only path produced `distressed/low` (truth: `upset/high`) at
# confidence 0.75, i.e. above REVIEW_THRESHOLD, so a wrong answer from a
# degraded path would not have been queued for review. Penalised and hard-capped
# below the review threshold so the primary classifier being absent always
# reaches a human.
CONF_LLM_UNAVAILABLE = -0.20           # MEASURED
LLM_UNAVAILABLE_MAX_CONF = 0.55        # MEASURED — sits below REVIEW_THRESHOLD 0.60
CONF_MIN, CONF_MAX = 0.05, 0.98
TIER_C_MAX_CONF = 0.45
DEGRADED_MAX_CONF = 0.50
NEAR_THRESHOLD_FRACTION = 0.10         # within 10% of a boundary
ASR_MIN_LOGPROB = -1.0                 # DERIVED
REVIEW_THRESHOLD = 0.60                # dashboard review queue

# coherence check (fuse.coherence_conflict)
# MEASURED 2026-08-17: the SER axes are heavily COMPRESSED on this telephony
# audio. Across the three labelled calls, valence spans only 0.534-0.638 and
# arousal only 0.594-0.646 - nothing like the full [0,1] the model nominally
# emits. Earlier values (0.45 / 0.55) sat INSIDE the observed valence range, so
# the conflict check would have fired on roughly half of all calls regardless
# of whether the label was right, penalising correct answers. call_001 (upset,
# valence 0.534) escaped a spurious flag by only 0.017.
# These bounds are deliberately set OUTSIDE the observed range, so the check
# fires only on genuinely extreme disagreement rather than on normal variation.
# Re-derive if a larger labelled set ever shows a wider spread.
VALENCE_POS_MIN = 0.40                 # MEASURED — below the 0.534 minimum
VALENCE_NEG_MAX = 0.70                 # MEASURED — above the 0.638 maximum
AROUSAL_HIGH_MIN = 0.45                # MEASURED — below the 0.594 minimum

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


# ------------------------------------------------------- hosted dashboard
# Job store and staged uploads. AUTOACE_DATA_DIR lets the tests point this at
# a tmp_path and lets the VM point it at the boot volume.
DATA_DIR = Path(os.environ.get("AUTOACE_DATA_DIR", REPO_ROOT / "_data"))
JOBS_DB = DATA_DIR / "jobs.db"
UPLOAD_DIR = DATA_DIR / "uploads"

JOB_TTL_SECONDS = 7 * 24 * 3600   # DERIVED — results kept until download. Audio is
                                  # unlinked per file regardless, so this governs
                                  # metadata and results only.
MAX_ATTEMPTS = 2                  # DERIVED — breaks the OOM crash loop. reconcile()
                                  # alone would requeue an OOM-killed file forever.
STALE_RUNNING_SECONDS = 900.0     # DERIVED — 5.6x the slowest measured file (160.8s)
WORKER_POLL_SECONDS = 2.0
UI_POLL_SECONDS = 5.0
MEAN_SECONDS_PER_FILE = 105.0     # MEASURED — mean of the three provided calls
                                  # (58.2/58.7/129.8s) x the 1.18 two-thread penalty
MAX_UPLOAD_MB = 500               # DERIVED — 50 files at the largest provided call
                                  # (2.8 MB) is ~140 MB; 500 MB is generous headroom
JOBS_CONNECT_TIMEOUT_S = 30.0     # DERIVED — how long a writer waits for the WAL
                                  # lock before raising. Generously above the
                                  # longest single-row write.
JOBS_BUSY_TIMEOUT_MS = 5000       # DERIVED — SQLite-level retry window for the
                                  # same contention, in milliseconds.
EXPIRY_SWEEP_SECONDS = 3600.0      # DERIVED — hourly is frequent enough against a
                                  # 7-day TTL, and cheap enough to run inline in
                                  # the worker's drain loop.
