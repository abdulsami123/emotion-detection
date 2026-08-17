"""End-to-end orchestration for one audio file.

Two invariants drive every decision here:

  1. Per-file fail isolation. One malformed file must never fail a batch.
     Every stage is wrapped; failures return a `FileResult` with `.error`
     set and `.analysis = None` (for a total failure) or a degraded-but-
     complete `CallAnalysis` (for a tone-branch-only failure - see below).

  2. The six signal fields (background noise / audio quality / overlap /
     long silence) must survive total tone-branch failure. They come from
     `analyse_signal`, which touches no LLM and no network, and are
     computed and locked in BEFORE the tone branch is attempted at all.

Tone path (two-tier, per task 14):
  1. Try `classify_tone` (Haiku). On success, use its tone, its intensity
     as the LLM voter, and its reasoning.
  2. On ANY exception (e.g. the Haiku call fails), fall back to
     `classify_tone_nli` over the customer transcript text and take the
     argmax as the tone. `tone_voters_agree` is forced False - there is
     only one working tone voter, and confidence must reflect that.
  3. Only if BOTH fail does the tone degrade to `EmotionalTone.NEUTRAL`.

Whichever path ran is recorded in `FileResult.reasoning` ("haiku" /
"nli fallback" / "both unavailable") - a silent fallback is operationally
indistinguishable from success, which is exactly the failure mode an
on-call engineer cannot afford.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoace.asr import Word, transcribe
from autoace.config import REVIEW_THRESHOLD, SAMPLE_RATE
from autoace.diarize import assign_speakers
from autoace.fuse import (
    ConfidenceInputs,
    coherence_conflict,
    compute_confidence,
    intensity_from_activation,
    reconcile_intensity,
)
from autoace.io_audio import load_mono
from autoace.prosody import (
    Tier,
    activation_profile,
    baseline_statistics,
    combine,
    discretise,
    extract_features,
    select_tier,
    z_score,
)
from autoace.schema import CallAnalysis, EmotionalIntensity, EmotionalTone
from autoace.ser import Dimensions, predict_dimensions
from autoace.signal_branch import analyse_signal
from autoace.tone_llm import ToneRequest, classify_tone
from autoace.tone_nli import classify_tone_nli
from autoace.vad import Segment, speech_segments


@dataclass
class FileResult:
    name: str
    analysis: CallAnalysis | None
    error: str | None = None
    reasoning: str = ""
    review_flagged: bool = False


def _fmt_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _text_in_segment(words: list[Word], segment: Segment) -> str:
    return " ".join(
        w.text.strip() for w in words if w.start < segment.end and w.end > segment.start
    ).strip()


def _thirds_means(values: list[float]) -> list[float]:
    """Split `values` into three consecutive, roughly-equal groups and
    return each group's mean - the trajectory input for Tier A calls."""
    n = len(values)
    size = n // 3
    thirds = [values[0:size], values[size : 2 * size], values[2 * size :]]
    return [float(np.mean(third)) for third in thirds if third]


def _agent_behavior(text: str) -> str:
    """A few string checks over the full call transcript. Not a substitute
    for a real agent-action classifier - just enough signal for the tone
    LLM's `agent_behavior` field to say something more useful than nothing."""
    lower = text.lower()
    if "advisor" in lower or "transfer" in lower:
        return "transferred to an advisor"
    if lower.count("hello?") >= 3:
        return "repeated-greeting dead-air loop"
    if any(
        phrase in lower
        for phrase in ("closed", "cannot", "can't help", "unable to", "denied", "no longer")
    ):
        return "request denied"
    return "handled the call without notable failure"


def _trajectory_description(per_values: list[float]) -> str:
    if not per_values:
        return "no customer speech to trend"
    return "activation by window: " + ", ".join(f"{v:.2f}" for v in per_values)


def analyse_file(path: str) -> FileResult:
    """Run the full pipeline on one audio file. Never raises: every stage
    is wrapped, and a failure at any point returns a `FileResult` describing
    what happened rather than propagating."""
    name = Path(path).name

    try:
        y, _sr = load_mono(path)
    except Exception as exc:  # noqa: BLE001 - fail isolation boundary
        return FileResult(name=name, analysis=None, error=f"audio load failed: {exc}")

    try:
        signal = analyse_signal(y)
    except Exception as exc:  # noqa: BLE001 - fail isolation boundary
        return FileResult(name=name, analysis=None, error=f"signal branch failed: {exc}")

    # ---- tone branch: fully wrapped so a signal-branch success is never
    # thrown away because VAD/diarization/ASR/SER/the LLM had a bad day.
    tone = EmotionalTone.NEUTRAL
    rule_intensity = EmotionalIntensity.MEDIUM
    llm_intensity = EmotionalIntensity.MEDIUM
    tier = Tier.C
    tone_voters_agree = False
    llm_self_confidence = 0.0
    diarization_degraded = True
    asr_avg_logprob = 0.0
    dims = Dimensions(arousal=0.5, dominance=0.5, valence=0.5)
    tone_path = "both unavailable"
    tone_path_detail = "tone branch did not run"

    try:
        speech = speech_segments(y)
        assignment = assign_speakers(y, speech)
        diarization_degraded = assignment.degraded
        customer_segments = assignment.customer_segments
        tier = select_tier(assignment.customer_speech_seconds)

        transcript = transcribe(path)
        asr_avg_logprob = transcript.avg_logprob

        dims = predict_dimensions(y)
        # Call-level SER arousal has no per-segment trajectory of its own
        # (predict_dimensions summarises the whole call), so it contributes
        # the same offset to every window - a constant term in the weighted
        # activation sum, scaled onto roughly the same footing as the
        # eGeMAPS z-scores using the compressed 0.5-0.7 range measured
        # elsewhere in this pipeline (see config.py AROUSAL_HIGH_MIN notes).
        ser_arousal_z = (dims.arousal - 0.6) / 0.1

        means, stds = baseline_statistics(y, customer_segments)
        per_segment_activation: list[float] = []
        annotated_lines: list[str] = []
        for seg in customer_segments:
            features = extract_features(y, seg)
            z = z_score(features, means, stds)
            tags = discretise(z)
            activation = combine(z, {"ser_arousal": ser_arousal_z})
            per_segment_activation.append(activation)
            text = _text_in_segment(transcript.words, seg)
            annotated_lines.append(
                f'[{_fmt_timestamp(seg.start)}-{_fmt_timestamp(seg.end)}] '
                f'({", ".join(tags)}) "{text}"'
            )

        if tier is Tier.A and len(per_segment_activation) >= 3:
            trajectory_values = _thirds_means(per_segment_activation)
        else:
            trajectory_values = per_segment_activation

        profile = activation_profile(trajectory_values)
        rule_intensity = intensity_from_activation(profile)

        customer_text = " ".join(
            _text_in_segment(transcript.words, seg) for seg in customer_segments
        ).strip()

        request = ToneRequest(
            duration_s=len(y) / SAMPLE_RATE,
            customer_speech_s=assignment.customer_speech_seconds,
            tier=tier.value,
            language=transcript.language,
            ser_line=dims.as_prompt_line(),
            agent_behavior=_agent_behavior(transcript.text),
            annotated_lines=annotated_lines,
            trajectory=_trajectory_description(trajectory_values),
        )

        # ---- two-tier tone classification ----
        try:
            response = classify_tone(request)
            tone = response.emotional_tone
            llm_intensity = response.emotional_intensity
            llm_self_confidence = response.self_confidence
            tone_voters_agree = True
            tone_path = "haiku"
            tone_path_detail = f"haiku: {response.reasoning}"
        except Exception as haiku_exc:  # noqa: BLE001 - two-tier fallback boundary
            try:
                nli_scores = classify_tone_nli(customer_text)
                tone = EmotionalTone(max(nli_scores, key=nli_scores.get))
                # No independent intensity read from the NLI fallback - the
                # rule-based (prosody) measurement is the only real signal,
                # so it is kept rather than manufacturing a disagreement.
                llm_intensity = rule_intensity
                tone_voters_agree = False
                tone_path = "nli fallback"
                tone_path_detail = (
                    f"nli fallback: haiku unavailable ({haiku_exc}); "
                    "used NLI argmax over the customer transcript"
                )
            except Exception as nli_exc:  # noqa: BLE001
                tone = EmotionalTone.NEUTRAL
                llm_intensity = rule_intensity
                tone_voters_agree = False
                tone_path = "both unavailable"
                tone_path_detail = (
                    f"both unavailable: haiku failed ({haiku_exc}) and NLI failed "
                    f"({nli_exc}); degraded to neutral"
                )
    except Exception as branch_exc:  # noqa: BLE001 - the top-level fail-isolation boundary
        tone_path = "both unavailable"
        tone_path_detail = f"both unavailable: tone branch failed entirely ({branch_exc})"

    intensity, intensity_voters_agree = reconcile_intensity(rule_intensity, llm_intensity, tier)
    conflict = coherence_conflict(tone, intensity, dims)
    ser_consistent = not conflict

    near_threshold = False  # no distance-to-boundary measurement is produced upstream

    confidence = compute_confidence(
        ConfidenceInputs(
            tone_voters_agree=tone_voters_agree,
            intensity_voters_agree=intensity_voters_agree,
            ser_consistent=ser_consistent,
            llm_self_confidence=llm_self_confidence,
            evidence_conflict=conflict,
            near_threshold=near_threshold,
            tier=tier,
            diarization_degraded=diarization_degraded,
            asr_avg_logprob=asr_avg_logprob,
            # The primary tone classifier never ran. Distinct from the two
            # voters disagreeing: this is a MISSING voter, and it hard-caps
            # confidence below REVIEW_THRESHOLD so the call always reaches a
            # human rather than shipping as if it were confidently classified.
            llm_unavailable=(tone_path != "haiku"),
        )
    )

    analysis = CallAnalysis(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=signal.background_noise_present,
        background_noise_type=signal.background_noise_type,
        background_noise_severity=signal.background_noise_severity,
        audio_quality=signal.audio_quality,
        speaker_overlap_present=signal.speaker_overlap_present,
        long_silence_present=signal.long_silence_present,
        confidence=confidence,
    )

    review_flagged = confidence < REVIEW_THRESHOLD or conflict

    return FileResult(
        name=name,
        analysis=analysis,
        error=None,
        reasoning=f"tone path: {tone_path}. {tone_path_detail}",
        review_flagged=review_flagged,
    )
