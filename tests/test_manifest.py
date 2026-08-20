"""Manifest parsing robustness.

The manifest is a single point of failure for a whole batch: unlike one bad
audio file, an unreadable manifest means nothing gets processed at all. Every
case here is something an evaluator-supplied CSV plausibly does.
"""

import pytest

from autoace.eval import load_manifest


def test_manifest_with_a_utf8_bom_is_readable(tmp_path):
    """Excel writes UTF-8 with a BOM by default. Opened as plain utf-8 the
    first fieldname becomes '\\ufeffname', so every row lookup raises KeyError
    and the entire batch fails before any inference runs."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\ncall_001.ogg,\n", encoding="utf-8-sig")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]
    assert rows[0].expected is None


def test_manifest_without_a_result_json_column_is_readable(tmp_path):
    """Brief section 7: for an unlabeled hidden test set result_json 'may be
    empty or omitted'. An omitted COLUMN must work, not just an empty cell."""
    path = tmp_path / "m.csv"
    path.write_text("name\ncall_001.ogg\ncall_002.ogg\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg", "call_002.ogg"]
    assert all(r.expected is None for r in rows)


def test_manifest_missing_the_name_column_raises_a_clear_error(tmp_path):
    """A bare KeyError('name') surfacing as an unhandled traceback tells the
    evaluator nothing. Name the problem and what was found instead."""
    path = tmp_path / "m.csv"
    path.write_text("filename,result_json\ncall_001.ogg,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="name"):
        load_manifest(str(path))


def test_manifest_that_is_empty_raises_a_clear_error(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_manifest(str(path))


def test_manifest_ignores_blank_trailing_rows(tmp_path):
    """A trailing newline or a row of empty cells is common in hand-edited
    CSVs and must not become a row named ''."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\ncall_001.ogg,\n,\n\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]


def test_manifest_strips_whitespace_around_names(tmp_path):
    """A name with a stray space would not match the audio file on disk, and
    would be reported as 'missing audio' for a file that is right there."""
    path = tmp_path / "m.csv"
    path.write_text("name,result_json\n  call_001.ogg  ,\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert [r.name for r in rows] == ["call_001.ogg"]


def test_manifest_reports_which_row_has_bad_json(tmp_path):
    """A malformed result_json cell must name the offending row rather than
    surfacing a bare JSONDecodeError with no context."""
    path = tmp_path / "m.csv"
    path.write_text(
        'name,result_json\ncall_001.ogg,"{not json}"\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="call_001.ogg"):
        load_manifest(str(path))


def test_manifest_reports_a_row_whose_json_violates_the_schema(tmp_path):
    """A syntactically valid object with a bad enum value must also be
    attributed to its row, not raise an opaque pydantic error."""
    path = tmp_path / "m.csv"
    path.write_text(
        'name,result_json\ncall_001.ogg,"{""emotional_tone"":""ecstatic""}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="call_001.ogg"):
        load_manifest(str(path))


def test_labelled_manifest_still_parses(tmp_path):
    """The happy path must not regress: a fully labelled row still yields a
    CallAnalysis."""
    from autoace.schema import CallAnalysis

    expected = CallAnalysis.model_validate(
        {
            "emotional_tone": "frustrated",
            "emotional_intensity": "medium",
            "background_noise_present": True,
            "background_noise_type": "office chatter",
            "background_noise_severity": "low",
            "audio_quality": "clear",
            "speaker_overlap_present": False,
            "long_silence_present": False,
            "confidence": 0.82,
        }
    )
    cell = '"' + expected.model_dump_json().replace('"', '""') + '"'
    path = tmp_path / "m.csv"
    path.write_text(f"name,result_json\ncall_001.ogg,{cell}\n", encoding="utf-8")

    rows = load_manifest(str(path))

    assert len(rows) == 1
    assert rows[0].expected == expected
