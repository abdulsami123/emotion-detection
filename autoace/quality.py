"""Technical audio quality, measured independently of background noise.

The brief warns: "do not infer background noise solely from poor audio
quality". These are separate code paths reading separate evidence. Default is
`clear` - all three labelled calls are clear, including the one with audible
static - so positive evidence is required to move off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np
import torch

from autoace.acoustics import (
    clip_percentage,
    frame_db,
    high_frequency_fraction,
    stft_magnitude,
)
from autoace.config import (
    CLIP_FRACTION_THRESHOLD,
    DROPOUT_MIN_MS,
    ECHO_LAG_RANGE_MS,
    ECHO_PEAK_MIN,
    LOW_VOLUME_LUFS,
    MUFFLE_HF_MIN,
    SAMPLE_RATE,
    SQUIM_STOI_SEVERE,
    SQUIM_STOI_SLIGHT,
)
from autoace.schema import AudioQuality
from autoace.vad import Segment

_FLOOR_SENTINEL = -90.0
_FRAME_HOP = SAMPLE_RATE // 100  # 10 ms, matching acoustics.stft_magnitude's hop

# Minimum-statistics ("MCRA-lite") floor tracker parameters. Ambient noise
# persists whether or not a frame is speech-active, but Silero VAD can and
# does misclassify a loud, broadband noise burst (TV audio, static) as
# speech - so averaging only over VAD-labelled non-speech frames *understates*
# noise severity for exactly the calls where it matters. Minimum-statistics
# noise estimation (Martin, 2001) sidesteps this: track the minimum of a
# lightly-smoothed power envelope in a sliding window across the WHOLE
# signal. Speech pushes energy up transiently; persistent noise does not dip
# below its own floor, so the sliding minimum recovers the ambient level
# under speech too, without relying on the VAD label being correct.
# Must match acoustics.baseline_characterisation, which produced the
# NOISE_SEVERITY_BANDS anchors. Changing it invalidates those thresholds.
_FLOOR_PERCENTILE = 60


@dataclass
class QualityResult:
    quality: AudioQuality
    estimated_stoi: float | None
    detectors: dict[str, bool]


def _frame_times(n_frames: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=SAMPLE_RATE, hop_length=_FRAME_HOP)


def _segment_mask(times: np.ndarray, segments: list[Segment]) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for seg in segments:
        mask |= (times >= seg.start) & (times < seg.end)
    return mask


def _is_silent(y: np.ndarray) -> bool:
    """True when the signal has no real energy at all.

    `frame_db`'s ref=np.max reference degenerates on an all-zero signal:
    librosa.amplitude_to_db floors both the numerator and the (zero) peak
    reference at its `amin`, so 0/0 resolves to 0 dB per frame - i.e. silence
    would otherwise read back as "as loud as the peak". Guard against that
    degenerate case before it ever reaches frame_db.
    """
    return bool(np.abs(y).max() < 1e-7) if len(y) else True


def noise_floor_dbfs(y: np.ndarray, non_speech: list[Segment]) -> float:
    """Ambient noise floor, in dB relative to the call's peak frame.

    CRITICAL: the scale must match the calibration. NOISE_SEVERITY_BANDS and
    NOISE_FLOOR_PRESENT were measured peak-relative, NOT as absolute dBFS.
    `frame_db` already expresses every frame that way (ref=np.max), so this
    function never rescales the audio - it only aggregates.

    `non_speech` gates whether there is any established quiet reference at
    all in this call: an empty list (nothing ever judged non-speech) means no
    evidence to estimate from, so we return a sentinel rather than 0.0, which
    would read as an implausibly loud floor and force `high` severity.
    Given at least one such region exists, the estimate uses the SAME
    percentile-split aggregation that produced the calibration anchors in
    `acoustics.baseline_characterisation`: mean level of frames below the 60th
    percentile of frame energy.

    This is deliberately the crude estimator rather than a better one, and the
    reason is scale consistency. Two alternatives were implemented and measured:

      estimator                     call_001   call_002   call_003
      ------------------------------ --------  ---------  ---------
      calibration anchors (truth)      -56.3     -52.1      -47.0
                                       none     medium     medium
      percentile split (SHIPPED)       -56.3     -52.1      -47.0
                                       none     medium     medium   OK
      minimum-statistics / MCRA        -56.8     -52.6      -52.0
                                       none        low        low   WRONG
      mean over Silero non-speech    ordering inverts (noise bursts land
                                     inside VAD's speech label, leaving only
                                     the quietest pauses in the gap bucket)

    Minimum-statistics tracking is the more principled estimator in the
    abstract - it is what speech-enhancement literature uses - but it seeks the
    QUIETEST moments, which under-reports a continuous noise. It compressed
    call_003 by 5 dB and collapsed the spread between the two noisy calls from
    5.1 dB to 0.6 dB, putting both in the `low` band against a `medium` ground
    truth. The `none`/`medium` boundary here is only 3 dB wide, so a 5 dB drift
    is fatal.

    The lesson: swapping the estimator without re-deriving the thresholds is
    what breaks severity. With three labelled calls there is no headroom to
    re-derive, so the estimator that produced the anchors is the one that
    ships. If a better floor tracker is ever adopted, NOISE_SEVERITY_BANDS must
    be re-measured against it in the same commit.
    """
    if not non_speech or _is_silent(y):
        return _FLOOR_SENTINEL
    db = frame_db(stft_magnitude(y))
    if db.size == 0:
        return _FLOOR_SENTINEL
    quiet = db <= np.percentile(db, _FLOOR_PERCENTILE)
    if not quiet.any():
        return _FLOOR_SENTINEL
    return float(db[quiet].mean())


def speech_level_dbfs(y: np.ndarray, speech: list[Segment]) -> float:
    """Mean speech level, on the same peak-relative scale as the floor.

    Unlike the floor, this is a straightforward mean over VAD-labelled speech
    frames: it only needs to sit safely above the floor (SNR = the
    difference), not hit a calibrated severity boundary, so it is not
    sensitive to the VAD-misclassification failure mode above.
    """
    if not speech or _is_silent(y):
        return _FLOOR_SENTINEL
    db = frame_db(stft_magnitude(y))
    mask = _segment_mask(_frame_times(len(db)), speech)
    if not mask.any():
        return _FLOOR_SENTINEL
    return float(db[mask].mean())


@lru_cache(maxsize=1)
def _load_squim():
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    return SQUIM_OBJECTIVE.get_model()


# SQUIM's transformer encoder runs full self-attention over the sample
# sequence: memory is O(n^2) in call length. Measured on this machine, a
# whole ~172 s call (call_003) tries to allocate ~118 GB and raises
# RuntimeError; ~31-35 s calls (call_001/002) run fine whole. Call-centre
# calls of that length are routine, not an edge case, so SQUIM is run in
# fixed windows and the per-window scores averaged, rather than passing the
# whole signal and silently losing STOI evidence on exactly the longest calls.
_SQUIM_CHUNK_S = 20.0


def _estimate_stoi(y: np.ndarray) -> float | None:
    """Non-intrusive quality estimate - no clean reference needed.

    Chunk scores are combined duration-weighted, not by a plain mean: the
    final chunk of a call is routinely much shorter than the rest (e.g. a
    ~15 s remainder after 20 s chunks on a 35 s call), and a short,
    boundary-truncated chunk gets a noticeably less reliable score from this
    model than a full-length one (measured: 0.96 on a 20 s chunk vs. 0.51 on
    the ~15 s remainder of the same call, whose whole-call unchunked score is
    0.83) - an unweighted mean lets that one short, unreliable chunk swing
    the call-level estimate as much as a full chunk covering more audio.
    """
    try:
        model = _load_squim()
        chunk = int(_SQUIM_CHUNK_S * SAMPLE_RATE)
        scores: list[tuple[float, int]] = []
        with torch.no_grad():
            for start in range(0, len(y), chunk):
                segment = y[start : start + chunk]
                if len(segment) < SAMPLE_RATE:  # too short to score meaningfully
                    continue
                stoi, _pesq, _sisdr = model(torch.from_numpy(segment).unsqueeze(0))
                scores.append((float(stoi.item()), len(segment)))
        if not scores:
            stoi, _pesq, _sisdr = model(torch.from_numpy(y).unsqueeze(0))
            return float(stoi.item())
        total = sum(n for _, n in scores)
        return sum(s * n for s, n in scores) / total
    except Exception:
        return None  # SQUIM unavailable: fall back to deterministic detectors


def _has_dropouts(y: np.ndarray) -> bool:
    """Vectorised longest-silent-run check.

    A Python per-sample loop over a 172 s call is ~2.75M iterations (measured
    at ~0.3 s on this machine, vs. ~0.01 s below) - cheap next to SQUIM, but
    still an unforced cost and a less obvious implementation. This computes
    the same "longest run of near-silent samples" via a vectorised
    run-length scan (find where the boolean silence mask changes value, then
    take the longest silent run) instead of a per-sample Python loop.
    """
    silent = np.abs(y) < 1e-4
    if not silent.any():
        return False
    min_run = int(DROPOUT_MIN_MS / 1000 * SAMPLE_RATE)
    # Indices where the boolean value changes, to get run boundaries.
    change = np.flatnonzero(np.diff(silent.astype(np.int8)))
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(silent)]))
    run_lengths = ends - starts
    run_values = silent[starts]
    silent_runs = run_lengths[run_values]
    return bool(silent_runs.size > 0 and silent_runs.max() >= min_run)


def _has_echo(y: np.ndarray) -> bool:
    """Detect a strong self-similar peak at a plausible echo lag.

    `np.correlate(y, y, mode="full")` on a 2.75M-sample call builds the
    entire lag range (O(n) lags, each an O(n) dot product) - effectively
    never returns. Echo only matters at the handful of lags in
    ECHO_LAG_RANGE_MS, so this evaluates a sampled grid of dot products
    directly over just that window (measured: ~0.03 s on the 172 s call)
    instead of computing every lag out to +/-n.
    """
    lo = int(ECHO_LAG_RANGE_MS[0] / 1000 * SAMPLE_RATE)
    hi = int(ECHO_LAG_RANGE_MS[1] / 1000 * SAMPLE_RATE)
    if len(y) <= hi:
        return False
    energy0 = float(np.dot(y, y))
    if energy0 <= 0:
        return False
    # Only the lags we care about (20-200 ms) are ever evaluated.
    best = 0.0
    for lag in range(lo, hi, max(1, (hi - lo) // 200)):
        shifted = y[lag:]
        base = y[: len(shifted)]
        corr = float(np.dot(base, shifted)) / energy0
        best = max(best, corr)
    return best >= ECHO_PEAK_MIN


def _is_low_volume(y: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(y**2)))
    return 20.0 * np.log10(max(rms, 1e-9)) < LOW_VOLUME_LUFS


def assess_quality(y: np.ndarray) -> QualityResult:
    spec = stft_magnitude(y)
    detectors = {
        "clipping": clip_percentage(y) / 100.0 > CLIP_FRACTION_THRESHOLD,
        "dropouts": _has_dropouts(y),
        "echo": _has_echo(y),
        # Baselined against telephony, NOT wideband speech: 4-6% of energy
        # above 3.4 kHz is normal here and must not count as muffling.
        "muffled": high_frequency_fraction(spec) < MUFFLE_HF_MIN,
        "low_volume": _is_low_volume(y),
    }

    stoi = _estimate_stoi(y)
    fired = sum(detectors.values())

    if (stoi is not None and stoi < SQUIM_STOI_SEVERE) or fired >= 2:
        quality = AudioQuality.SEVERELY_IMPAIRED
    elif (stoi is not None and stoi < SQUIM_STOI_SLIGHT) or fired == 1:
        quality = AudioQuality.SLIGHTLY_IMPAIRED
    else:
        quality = AudioQuality.CLEAR

    return QualityResult(quality=quality, estimated_stoi=stoi, detectors=detectors)
