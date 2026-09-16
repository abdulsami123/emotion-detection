"""Batch scoring harness for the three labelled reference calls.

Reads the brief's manifest format (`name,result_json`) and scores predictions
against it. This module never reaches into the audio pipeline itself - it is
pure comparison of two `CallAnalysis`-shaped things, so it can be exercised
without any model loaded.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

from emotion_detection.schema import CallAnalysis

# Fixed by the brief's batch format - the manifest and any batch output must
# use exactly these two column names.
BATCH_COLUMNS = ("name", "result_json")


@dataclass
class ManifestRow:
    name: str
    expected: CallAnalysis | None


def load_manifest(path: str) -> list[ManifestRow]:
    """Read the brief's CSV manifest.

    An empty `result_json` cell - or an entirely absent `result_json` column -
    means the row is unlabelled (audio provided, no ground truth yet), so
    `expected` is None rather than raising. Brief section 7 allows exactly
    that for an unlabeled hidden test set.

    Opened as `utf-8-sig` because a manifest saved from Excel carries a BOM,
    which under plain `utf-8` turns the first fieldname into '﻿name' and
    fails EVERY row lookup - taking down the whole batch before any inference
    runs. The manifest is a single point of failure in a way one bad audio file
    is not, so its errors name what went wrong and which row it was.
    """
    rows: list[ManifestRow] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: manifest is empty")
        if "name" not in reader.fieldnames:
            raise ValueError(
                f"{path}: manifest has no 'name' column - found "
                f"{reader.fieldnames}. The brief requires a 'name' column "
                f"holding the exact audio filename."
            )

        for record in reader:
            name = (record.get("name") or "").strip()
            if not name:
                continue  # blank or trailing row

            raw = (record.get("result_json") or "").strip()
            if not raw:
                rows.append(ManifestRow(name=name, expected=None))
                continue

            try:
                expected = CallAnalysis(**json.loads(raw))
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ValueError(
                    f"{path}: row '{name}' has an unreadable result_json: {exc}"
                ) from exc
            rows.append(ManifestRow(name=name, expected=expected))
    return rows


def _norm(value: str) -> str:
    return value.strip().lower()


_MISSING = "__missing__"


def _accuracy(pairs: list[tuple]) -> float:
    if not pairs:
        return 0.0
    correct = sum(1 for truth, pred in pairs if truth == pred)
    return correct / len(pairs)


def _confusion(pairs: list[tuple]) -> dict:
    matrix: dict = {}
    for truth, pred in pairs:
        matrix.setdefault(truth, {}).setdefault(pred, 0)
        matrix[truth][pred] += 1
    return matrix


def _macro_f1(pairs: list[tuple]) -> float:
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs}, key=str)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores.append(f1)
    return sum(scores) / len(scores)


def score_batch(rows: list[ManifestRow], predictions: dict[str, CallAnalysis]) -> dict:
    """Score `predictions` (name -> CallAnalysis) against the manifest's
    ground truth. Rows with no ground truth are ignored entirely - they
    cannot be scored either way. A labelled row with no matching prediction
    counts as an error AND as a miss on every metric (never silently
    dropped, which would flatter the accuracy figures)."""
    labelled = [row for row in rows if row.expected is not None]
    n_labelled = len(labelled)
    errors = sum(1 for row in labelled if row.name not in predictions)

    tone_pairs: list[tuple] = []
    intensity_pairs: list[tuple] = []
    noise_present_pairs: list[tuple] = []
    noise_severity_pairs: list[tuple] = []
    noise_type_pairs: list[tuple] = []
    technical_fields = ("audio_quality", "speaker_overlap_present", "long_silence_present")
    technical_pairs: dict[str, list[tuple]] = {field: [] for field in technical_fields}

    for row in labelled:
        expected = row.expected
        predicted = predictions.get(row.name)

        if predicted is None:
            tone_pairs.append((expected.emotional_tone.value, _MISSING))
            intensity_pairs.append((expected.emotional_intensity.value, _MISSING))
            noise_present_pairs.append((expected.background_noise_present, _MISSING))
            noise_severity_pairs.append((expected.background_noise_severity.value, _MISSING))
            noise_type_pairs.append((_norm(expected.background_noise_type), _MISSING))
            for field in technical_fields:
                technical_pairs[field].append((getattr(expected, field), _MISSING))
            continue

        tone_pairs.append((expected.emotional_tone.value, predicted.emotional_tone.value))
        intensity_pairs.append(
            (expected.emotional_intensity.value, predicted.emotional_intensity.value)
        )
        noise_present_pairs.append(
            (expected.background_noise_present, predicted.background_noise_present)
        )
        noise_severity_pairs.append(
            (expected.background_noise_severity.value, predicted.background_noise_severity.value)
        )
        noise_type_pairs.append(
            (_norm(expected.background_noise_type), _norm(predicted.background_noise_type))
        )
        for field in technical_fields:
            technical_pairs[field].append((getattr(expected, field), getattr(predicted, field)))

    medium_count = sum(1 for row in labelled if row.expected.emotional_intensity.value == "medium")
    constant_medium_baseline = medium_count / n_labelled if n_labelled else 0.0

    return {
        "n_labelled": n_labelled,
        "errors": errors,
        "tone": {
            "accuracy": _accuracy(tone_pairs),
            "macro_f1": _macro_f1(tone_pairs),
            "confusion": _confusion(tone_pairs),
        },
        "intensity": {
            "accuracy": _accuracy(intensity_pairs),
            "macro_f1": _macro_f1(intensity_pairs),
            "confusion": _confusion(intensity_pairs),
            "constant_medium_baseline": constant_medium_baseline,
        },
        "noise": {
            "present_accuracy": _accuracy(noise_present_pairs),
            "severity_macro_f1": _macro_f1(noise_severity_pairs),
            "type_accuracy": _accuracy(noise_type_pairs),
        },
        "technical": {
            field: _accuracy(pairs) for field, pairs in technical_pairs.items()
        },
    }
