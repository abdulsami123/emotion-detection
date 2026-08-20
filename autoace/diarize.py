"""Two-speaker assignment via ECAPA embeddings and agglomerative clustering.

Cheaper than full pyannote diarization and more robust for the 2-speaker
case. Degrades predictably: reference-bank match, then first-speaker
heuristic, then a `degraded` flag that caps downstream confidence.

Getting this wrong fails silently: swapping the roles substitutes the bot's
flat TTS delivery for the customer's voice, dragging tone toward `neutral`
and intensity toward `low` with no error raised. Hence the layered fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from autoace.config import (
    AGENT_REF_MIN_SIM,
    DIARIZATION_MIN_SILHOUETTE,
    ECAPA_MODEL,
    MODELS_DIR,
    SAMPLE_RATE,
)
from autoace.vad import Segment

_MIN_EMBED_SAMPLES = int(0.4 * SAMPLE_RATE)  # ECAPA needs ~0.4s to be stable


@dataclass
class SpeakerAssignment:
    assignments: dict[int, str]           # segment index -> "agent" | "customer"
    customer_speech_seconds: float
    degraded: bool
    agent_reference_similarity: float | None = None
    customer_segments: list[Segment] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # Windows symlinks require developer mode / admin privileges we cannot
    # assume the runner has; copy the cached weights into savedir instead.
    return EncoderClassifier.from_hparams(
        source=ECAPA_MODEL,
        savedir=str(MODELS_DIR / "ecapa"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )


def _embed(y: np.ndarray, segments: list[Segment]) -> np.ndarray:
    encoder = _load_encoder()
    vectors = []
    for seg in segments:
        chunk = y[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
        if len(chunk) < _MIN_EMBED_SAMPLES:
            chunk = np.pad(chunk, (0, _MIN_EMBED_SAMPLES - len(chunk)))
        with torch.no_grad():
            vec = encoder.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
        vectors.append(vec.squeeze().cpu().numpy())
    return np.vstack(vectors)


def _degraded_result(segments: list[Segment]) -> SpeakerAssignment:
    """One effective speaker: treat all speech as customer and flag it."""
    return SpeakerAssignment(
        assignments={i: "customer" for i in range(len(segments))},
        customer_speech_seconds=sum(s.duration for s in segments),
        degraded=True,
        customer_segments=list(segments),
    )


def assign_speakers(
    y: np.ndarray,
    segments: list[Segment],
    agent_reference: np.ndarray | None = None,
) -> SpeakerAssignment:
    if len(segments) < 2:
        return _degraded_result(segments)

    embeddings = _embed(y, segments)
    labels = AgglomerativeClustering(
        n_clusters=2, metric="cosine", linkage="average"
    ).fit_predict(embeddings)

    if len(set(labels)) < 2:
        return _degraded_result(segments)

    score = silhouette_score(embeddings, labels, metric="cosine")
    if score < DIARIZATION_MIN_SILHOUETTE:
        return _degraded_result(segments)

    agent_cluster, similarity = _identify_agent(embeddings, labels, agent_reference)

    assignments = {
        i: ("agent" if label == agent_cluster else "customer")
        for i, label in enumerate(labels)
    }
    customer_segments = [
        seg for i, seg in enumerate(segments) if assignments[i] == "customer"
    ]
    return SpeakerAssignment(
        assignments=assignments,
        customer_speech_seconds=sum(s.duration for s in customer_segments),
        degraded=False,
        agent_reference_similarity=similarity,
        customer_segments=customer_segments,
    )


def _identify_agent(
    embeddings: np.ndarray, labels: np.ndarray, reference: np.ndarray | None
) -> tuple[int, float | None]:
    """Match against the stored agent voice; fall back to first-speaker."""
    if reference is not None:
        best_cluster, best_similarity = 0, -1.0
        for cluster in (0, 1):
            centroid = embeddings[labels == cluster].mean(axis=0)
            similarity = float(
                centroid
                @ reference
                / (np.linalg.norm(centroid) * np.linalg.norm(reference))
            )
            if similarity > best_similarity:
                best_cluster, best_similarity = cluster, similarity
        if best_similarity >= AGENT_REF_MIN_SIM:
            return best_cluster, best_similarity

    # Fallback: the bot greets first on every observed call.
    return int(labels[0]), None


def build_agent_reference(calls: list[tuple[np.ndarray, list[Segment]]]) -> np.ndarray:
    """Mean embedding of the first (agent) segment across known calls.

    Speaker identity only — carries no label information, so it cannot leak
    tone or noise ground truth into the evaluation.
    """
    vectors = [_embed(y, segments[:1])[0] for y, segments in calls if segments]
    return np.vstack(vectors).mean(axis=0)
