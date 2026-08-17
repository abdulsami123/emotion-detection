import pytest

from autoace.config import LABELS_CSV
from autoace.eval import BATCH_COLUMNS, load_manifest, score_batch


def test_load_manifest_reads_the_brief_format():
    rows = load_manifest(str(LABELS_CSV))
    assert len(rows) == 3
    assert rows[0].name == "call_001.ogg"
    assert rows[0].expected.emotional_tone.value == "upset"


def test_manifest_columns_match_the_brief():
    assert BATCH_COLUMNS == ("name", "result_json")


def test_perfect_predictions_score_one():
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {row.name: row.expected for row in rows})
    assert report["tone"]["accuracy"] == 1.0
    assert report["intensity"]["accuracy"] == 1.0
    assert report["errors"] == 0


def test_intensity_reports_the_constant_medium_baseline():
    """A metric that does not beat the majority-class baseline is not evidence.
    Ground truth is high/medium/medium, so the baseline is 2/3."""
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {row.name: row.expected for row in rows})
    assert report["intensity"]["constant_medium_baseline"] == pytest.approx(2 / 3)


def test_missing_predictions_are_counted_as_errors():
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {})
    assert report["errors"] == 3
    assert report["tone"]["accuracy"] == 0.0


def test_report_groups_match_the_brief_scoring_categories():
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {row.name: row.expected for row in rows})
    for group in ("tone", "intensity", "noise", "technical"):
        assert group in report


def test_confusion_matrix_is_produced_for_tone():
    rows = load_manifest(str(LABELS_CSV))
    report = score_batch(rows, {row.name: row.expected for row in rows})
    assert report["tone"]["confusion"]
