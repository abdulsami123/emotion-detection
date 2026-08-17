import json
import zipfile
from pathlib import Path

import pytest

from autoace.app import (
    SCHEMA_COLUMNS,
    BatchValidation,
    results_to_csv,
    results_to_json,
    validate_batch,
)
from autoace.pipeline import FileResult
from autoace.schema import CallAnalysis


def _analysis(**overrides):
    base = dict(
        emotional_tone="neutral",
        emotional_intensity="medium",
        background_noise_present=False,
        background_noise_type="",
        background_noise_severity="none",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.7,
    )
    base.update(overrides)
    return CallAnalysis(**base)


def test_validation_reports_manifest_rows_with_no_audio(tmp_path: Path):
    """Reported BEFORE processing, per the brief."""
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,\ncall_999.ogg,\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "call_999.ogg" in report.missing_audio
    assert report.matched == ["call_001.ogg"]


def test_validation_reports_audio_with_no_manifest_row(tmp_path: Path):
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    (tmp_path / "stray.wav").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text(
        'name,result_json\ncall_001.ogg,\n', encoding="utf-8"
    )
    report = validate_batch(tmp_path)
    assert "stray.wav" in report.unlisted_audio


def test_validation_with_no_manifest_still_lists_the_audio(tmp_path: Path):
    (tmp_path / "call_001.ogg").write_bytes(b"x")
    report = validate_batch(tmp_path)
    assert report.ok is False
    assert "call_001.ogg" in report.unlisted_audio


def test_csv_export_preserves_original_filenames():
    csv_text = results_to_csv([FileResult(name="call_042.mp3", analysis=_analysis())])
    assert "call_042.mp3" in csv_text
    assert "neutral" in csv_text
    for column in SCHEMA_COLUMNS:
        assert column in csv_text


def test_csv_export_includes_failed_files_with_a_reason():
    """A failed file must appear in the output, not vanish."""
    csv_text = results_to_csv(
        [FileResult(name="broken.wav", analysis=None, error="unsupported extension")]
    )
    assert "broken.wav" in csv_text
    assert "unsupported extension" in csv_text


def test_json_export_round_trips_and_marks_errors():
    payload = json.loads(
        results_to_json(
            [
                FileResult(name="ok.ogg", analysis=_analysis()),
                FileResult(name="bad.ogg", analysis=None, error="boom"),
            ]
        )
    )
    assert payload[0]["name"] == "ok.ogg"
    assert payload[0]["result"]["emotional_tone"] == "neutral"
    assert payload[1]["result"] is None
    assert payload[1]["error"] == "boom"


def test_review_flagged_rows_are_identifiable_in_the_table():
    """A degraded result must be visible, not buried - with the API key invalid
    every call currently takes the NLI fallback and flags for review."""
    from autoace.app import results_to_table

    rows = results_to_table(
        [
            FileResult(name="low.ogg", analysis=_analysis(confidence=0.55),
                       review_flagged=True),
            FileResult(name="high.ogg", analysis=_analysis(confidence=0.9)),
        ]
    )
    assert rows[0][0] == "low.ogg", "flagged rows must sort to the top"
    assert any("REVIEW" in str(cell) for cell in rows[0])


def test_app_builds_without_launching():
    """Import and construction must work headlessly; never call launch() here."""
    from autoace.app import build_app

    assert build_app() is not None
