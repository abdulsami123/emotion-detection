"""End-to-end regression over the three labelled calls.

This is a REGRESSION test, not a validation set. With n=3 it establishes that
the system reproduces known-good behaviour and nothing more; it cannot support
an accuracy claim. The assertions are therefore split by how much confidence
the evidence actually supports:

  - The six deterministic signal fields are asserted EXACTLY. They come from
    measured, calibrated thresholds and all six are correct on all three calls.
  - Intensity is asserted only against the constant-`medium` baseline. A metric
    that does not beat the majority class is not evidence of anything, so the
    bar is "at least as good as guessing", not a fixed number.
  - Tone is NOT asserted on accuracy at all. The Anthropic key is currently
    invalid, so tone runs on the NLI fallback; pinning a number here would pin
    the fallback's performance and would break the moment the primary path
    starts working. What IS asserted is that the degraded path is visible.
"""

import pytest

from emotion_detection.config import LABELS_CSV, REVIEW_THRESHOLD, reference_call
from emotion_detection.eval import load_manifest, score_batch
from emotion_detection.pipeline import analyse_file


@pytest.fixture(scope="module")
def outcome():
    rows = load_manifest(str(LABELS_CSV))
    results = {row.name: analyse_file(reference_call(row.name)) for row in rows}
    predictions = {
        name: result.analysis for name, result in results.items() if result.analysis
    }
    return rows, results, predictions, score_batch(rows, predictions)


@pytest.mark.slow
def test_every_call_produces_a_result(outcome):
    _rows, results, predictions, report = outcome
    assert len(predictions) == 3
    assert report["errors"] == 0
    for name, result in results.items():
        assert result.error is None, f"{name}: {result.error}"


@pytest.mark.slow
def test_deterministic_signal_fields_are_exact(outcome):
    """These five are the reliable-points block: measured thresholds, no API,
    no network. All correct on all three calls."""
    _rows, _results, _predictions, report = outcome
    assert report["noise"]["present_accuracy"] == 1.0
    assert report["noise"]["severity_macro_f1"] == 1.0
    assert report["technical"]["audio_quality"] == 1.0
    assert report["technical"]["speaker_overlap_present"] == 1.0
    assert report["technical"]["long_silence_present"] == 1.0


@pytest.mark.slow
def test_noise_type_is_right_wherever_the_source_is_environmental(outcome):
    """2 of 3. The miss is call_003, whose ground truth is 'sharp static' - a
    line artifact that AudioSet cannot see and that no tested spectral feature
    discriminates. Tracked separately as a strict xfail in test_tagging.py."""
    _rows, _results, _predictions, report = outcome
    assert report["noise"]["type_accuracy"] >= 2 / 3


@pytest.mark.slow
def test_intensity_is_no_worse_than_guessing_the_majority_class(outcome):
    """The bar is the constant-`medium` baseline, not a fixed number. Intensity
    previously scored BELOW it (0.33 vs 0.67) because a degenerate self-baseline
    manufactured `low` on short calls from zero evidence."""
    _rows, _results, _predictions, report = outcome
    assert (
        report["intensity"]["accuracy"]
        >= report["intensity"]["constant_medium_baseline"]
    )


@pytest.mark.slow
def test_a_degraded_tone_path_is_always_visible(outcome):
    """The failure mode to prevent is a silent fallback: a result produced
    without the primary classifier must never look confident. It is hard-capped
    below the review threshold and the path is recorded in the reasoning."""
    _rows, results, _predictions, _report = outcome
    for name, result in results.items():
        assert result.reasoning, f"{name}: no tone path recorded"
        used_primary = "haiku" in result.reasoning.lower() and "fallback" not in (
            result.reasoning.lower()
        )
        if not used_primary:
            assert result.analysis.confidence < REVIEW_THRESHOLD, (
                f"{name}: degraded path reported confidence "
                f"{result.analysis.confidence} at or above the review threshold"
            )
            assert result.review_flagged, f"{name}: degraded path not flagged"


@pytest.mark.slow
def test_report_carries_every_group_the_brief_scores(outcome):
    _rows, _results, _predictions, report = outcome
    for group in ("tone", "intensity", "noise", "technical"):
        assert group in report
    assert report["tone"]["confusion"]
    assert report["intensity"]["confusion"]
