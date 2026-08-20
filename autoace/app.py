"""The hosted dashboard (brief section 7).

Every requirement in that section maps to one function here, kept separate
from the Gradio wiring so each can be unit-tested without a browser or a
running server:

  - `validate_batch`   -> validation BEFORE processing (missing/unlisted/
                           unsupported files reported as a list).
  - `run_batch`         -> visible progress + per-file failure isolation.
  - `results_to_csv` /
    `results_to_json`   -> downloadable exports, original filenames preserved,
                           failures included rather than dropped.
  - `results_to_table`  -> the review queue. Confidence below
                           `REVIEW_THRESHOLD` is common right now (every call
                           takes the NLI fallback while the Anthropic key is
                           invalid - see pipeline.py), so flagged rows are
                           sorted to the top rather than left to blend in.
  - `build_app`         -> the Blocks UI. Never calls `.launch()` - that only
                           happens in `__main__`, and only with a password.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from autoace import jobs
from autoace.config import MAX_UPLOAD_MB, REVIEW_THRESHOLD, UI_POLL_SECONDS
from autoace.eval import load_manifest, score_batch
from autoace.io_audio import SUPPORTED_SUFFIXES
from autoace.pipeline import FileResult, analyse_file
from autoace.schema import CallAnalysis

# "name" first, then the nine schema fields, in schema declaration order.
SCHEMA_COLUMNS = ("name",) + tuple(CallAnalysis.model_fields.keys())

TABLE_HEADERS = [
    "file",
    "tone",
    "intensity",
    "noise type",
    "severity",
    "quality",
    "confidence",
    "flag",
    "notes",
]


@dataclass
class BatchValidation:
    matched: list[str] = field(default_factory=list)
    missing_audio: list[str] = field(default_factory=list)
    unlisted_audio: list[str] = field(default_factory=list)
    manifest_path: Path | None = None

    @property
    def ok(self) -> bool:
        """True only when there IS a manifest and at least one row of it
        matched a real audio file - anything else means there is nothing
        safe to run inference on yet."""
        return self.manifest_path is not None and len(self.matched) > 0


def _iter_audio_files(folder: Path):
    """All files under `folder` (recursively - a ZIP upload may nest its
    contents inside one top-level directory rather than putting audio at
    the literal root) whose extension is decodable."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def validate_batch(folder: str | Path) -> BatchValidation:
    """Compare the manifest's `name` column against the audio actually
    present, in both directions, BEFORE any inference runs.

    Searched recursively: a ZIP that unpacks into one wrapper folder must
    validate the same as one with audio literally at the root.
    """
    folder = Path(folder)

    csv_files = sorted(folder.rglob("*.csv"))
    manifest_path = csv_files[0] if csv_files else None

    audio_by_name: dict[str, Path] = {}
    for path in _iter_audio_files(folder):
        # First match wins - duplicate basenames in different subfolders are
        # a batch-authoring problem, not something to silently overwrite.
        audio_by_name.setdefault(path.name, path)

    if manifest_path is not None:
        manifest_names = [row.name for row in load_manifest(str(manifest_path))]
    else:
        manifest_names = []

    matched = [name for name in manifest_names if name in audio_by_name]
    missing_audio = [name for name in manifest_names if name not in audio_by_name]
    listed = set(manifest_names)
    unlisted_audio = sorted(name for name in audio_by_name if name not in listed)

    return BatchValidation(
        matched=matched,
        missing_audio=missing_audio,
        unlisted_audio=unlisted_audio,
        manifest_path=manifest_path,
    )


def _unsupported_files(folder: Path) -> list[str]:
    """Non-audio, non-manifest files in the upload - reported so the
    evaluator sees them called out rather than silently ignored."""
    names = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_SUFFIXES or suffix == ".csv":
            continue
        names.append(path.name)
    return sorted(names)


def _row_dict(result: FileResult) -> dict:
    row = {col: "" for col in SCHEMA_COLUMNS}
    row["name"] = result.name
    if result.analysis is not None:
        data = result.analysis.model_dump(mode="json")
        for col in SCHEMA_COLUMNS[1:]:
            row[col] = data[col]
    row["error"] = result.error or ""
    return row


def results_to_csv(results: list[FileResult]) -> str:
    """One row per file, failures included (with `.error` in an `error`
    column) so a malformed file shows up as a row, never a silent drop."""
    fieldnames = list(SCHEMA_COLUMNS) + ["error"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(_row_dict(result))
    return buf.getvalue()


def results_to_json(results: list[FileResult]) -> str:
    payload = []
    for result in results:
        payload.append(
            {
                "name": result.name,
                "result": (
                    json.loads(result.analysis.model_dump_json())
                    if result.analysis is not None
                    else None
                ),
                "error": result.error,
            }
        )
    return json.dumps(payload, indent=2)


def results_to_table(results: list[FileResult]) -> list[list]:
    """Display rows for the review queue. Review-flagged (or outright
    failed) rows sort to the top, then ascending confidence - a degraded
    result must be visible, not buried at the bottom of a 50-row batch."""
    rows: list[list] = []
    for result in results:
        if result.analysis is not None:
            data = result.analysis.model_dump(mode="json")
            confidence = data["confidence"]
            flagged = bool(result.review_flagged) or confidence < REVIEW_THRESHOLD
            rows.append(
                [
                    result.name,
                    data["emotional_tone"],
                    data["emotional_intensity"],
                    data["background_noise_type"],
                    data["background_noise_severity"],
                    data["audio_quality"],
                    round(confidence, 3),
                    "REVIEW" if flagged else "",
                    result.reasoning,
                ]
            )
        else:
            rows.append(
                [result.name, "", "", "", "", "", None, "ERROR", result.error or ""]
            )

    def sort_key(row):
        flagged = row[7] in ("REVIEW", "ERROR")
        confidence = row[6] if isinstance(row[6], (int, float)) else -1.0
        return (0 if flagged else 1, confidence)

    rows.sort(key=sort_key)
    return rows


def _resolve_path(item) -> Path:
    if hasattr(item, "name"):  # gradio's uploaded-file wrapper
        return Path(item.name)
    return Path(item)


def _prepare_workdir(upload) -> Path:
    """Normalise whatever Gradio hands back (a single ZIP path, a single
    folder path, or a list of individual file paths from a multi-file/
    folder picker) into one directory holding the batch."""
    if isinstance(upload, (list, tuple)):
        workdir = Path(tempfile.mkdtemp(prefix="autoace_batch_"))
        for item in upload:
            src = _resolve_path(item)
            dest = workdir / src.name
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
        return workdir

    path = _resolve_path(upload)
    if path.is_dir():
        return path
    if path.suffix.lower() == ".zip":
        workdir = Path(tempfile.mkdtemp(prefix="autoace_batch_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(workdir)
        return workdir
    # A single non-zip file (e.g. one CSV dropped alone): its parent
    # directory is the closest thing to "the batch".
    return path.parent


def _describe_validation(workdir: Path, validation: BatchValidation) -> list[str]:
    """Human-readable problems found before any inference runs.

    Extracted from the old `run_batch` unchanged in substance: the brief
    requires unmatched files to be reported rather than silently skipped, and
    that has to happen at enqueue time, not when the worker gets there.
    """
    problems: list[str] = []
    if validation.manifest_path is None:
        problems.append("No CSV manifest found in the upload.")
    if validation.missing_audio:
        problems.append(
            "Manifest rows with no matching audio file: "
            + ", ".join(validation.missing_audio)
        )
    if validation.unlisted_audio:
        problems.append(
            "Audio files with no manifest row (will not be processed): "
            + ", ".join(validation.unlisted_audio)
        )
    unsupported = _unsupported_files(workdir)
    if unsupported:
        problems.append(
            "Unsupported file formats (ignored): " + ", ".join(unsupported)
        )
    return problems


def enqueue_batch(upload) -> tuple[str, str]:
    """Validate the upload and queue it. Returns `(job_id, status_markdown)`.

    Returns as soon as the rows are written - the worker does the work. An
    empty `job_id` means nothing was queued and there is nothing to poll.
    """
    if upload is None:
        return "", "No file uploaded."

    workdir = _prepare_workdir(upload)
    validation = validate_batch(workdir)
    problems = _describe_validation(workdir, validation)
    validation_json = json.dumps(
        {
            "matched": validation.matched,
            "missing_audio": validation.missing_audio,
            "unlisted_audio": validation.unlisted_audio,
            "problems": problems,
        }
    )

    conn = jobs.connect()
    try:
        if not validation.matched:
            reason = "Validation failed - nothing to process.\n" + "\n".join(
                f"- {p}" for p in problems
            )
            job_id = jobs.enqueue_failed(
                conn, workdir=workdir, validation_json=validation_json, error=reason
            )
            return job_id, reason

        audio_by_name: dict[str, Path] = {}
        for path in _iter_audio_files(workdir):
            audio_by_name.setdefault(path.name, path)
        matched_audio = {name: audio_by_name[name] for name in validation.matched}

        manifest_labelled = False
        if validation.manifest_path is not None:
            manifest_labelled = any(
                row.expected is not None
                for row in load_manifest(str(validation.manifest_path))
            )

        job_id = jobs.enqueue(
            conn,
            workdir=workdir,
            audio_by_name=matched_audio,
            validation_json=validation_json,
            manifest_labelled=manifest_labelled,
        )
    finally:
        conn.close()

    lines = [
        f"**Queued {len(matched_audio)} file(s).** Job ID: `{job_id}`",
        "",
        "Keep this job ID. You can close the page and paste it into the "
        "**Look up a job** box to come back to these results.",
    ]
    if problems:
        lines.append("")
        lines.append("Validation warnings:")
        lines.extend(f"- {p}" for p in problems)
    return job_id, "\n".join(lines)


def _format_eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"~{minutes} min"
    return f"~{minutes // 60}h {minutes % 60:02d}m"


def poll_job(job_id: str):
    """Read-only projection of the store for the UI.

    Returns `(status_markdown, table_rows, csv_path, json_path, scoring_text)`
    - the same five outputs the old synchronous `run_batch` returned, so the
    Blocks wiring keeps the same shape.
    """
    empty = ("", [], None, None, "")
    if not job_id:
        return empty

    conn = jobs.connect()
    try:
        status = jobs.job_status(conn, job_id)
        if status is None:
            return (
                f"Job `{job_id}` not found. It may have expired.", [], None, None, "",
            )

        results = jobs.job_results(conn, job_id)
        validation = json.loads(status.validation_json or "{}")
    finally:
        conn.close()

    if status.status == "failed" and status.total == 0:
        return (status.error or "Validation failed.", [], None, None, "")

    lines = [
        f"**Job** `{job_id}` — **{status.status}**",
        "",
        f"{status.finished} of {status.total} files finished "
        f"({status.done} analysed, {status.failed} failed).",
    ]
    if status.status in ("pending", "running"):
        if status.queue_position:
            lines.append(
                f"Queued behind {status.queue_position} other job(s). "
                f"Estimated wait: {_format_eta(status.eta_seconds)}."
            )
        else:
            lines.append(
                f"Estimated time remaining: {_format_eta(status.eta_seconds)}."
            )
    for problem in validation.get("problems", []):
        lines.append(f"- {problem}")

    # `results_to_table`'s ERROR branch fires on `analysis is None` alone, with
    # no way to tell "actually failed" from "not processed yet" - correct for
    # a finished batch, wrong here: an unclaimed file would rank as an error
    # ahead of a genuinely REVIEW-flagged completed row. So the still-
    # outstanding (pending/running) rows are kept out of that sort and
    # appended after, tagged PENDING rather than ERROR.
    finished = [r for r in results if r.analysis is not None or r.error]
    outstanding = [r for r in results if r.analysis is None and not r.error]
    table = results_to_table(finished)
    table.extend(
        [r.name, "", "", "", "", "", None, "PENDING", ""] for r in outstanding
    )

    out_dir = Path(tempfile.mkdtemp(prefix="autoace_out_"))
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    csv_path.write_text(results_to_csv(results), encoding="utf-8")
    json_path.write_text(results_to_json(results), encoding="utf-8")

    scoring_text = ""
    if status.manifest_labelled and status.status == "complete":
        # The manifest lives in the workdir, which is removed when the job
        # finishes - so metrics cannot be computed here yet. Task 12 persists
        # the manifest on the job row and replaces this with real scoring. An
        # explicit message beats a silently missing feature.
        scoring_text = (
            "Scoring for a labelled batch is computed once the manifest is "
            "persisted with the job (see the implementation plan, Task 12)."
        )

    return "\n".join(lines), table, str(csv_path), str(json_path), scoring_text


def build_app() -> gr.Blocks:
    """The UI. Construction only - never calls `.launch()`."""
    with gr.Blocks(title="AutoAce - Call Tone Review") as demo:
        gr.Markdown(
            "# AutoAce batch review\n"
            "Upload a folder or ZIP containing audio files at the root plus "
            "one CSV manifest (`name,result_json`). Validation runs before "
            "any inference, and rows flagged for human review (low "
            "confidence or a conflicting signal) sort to the top of the "
            "table below."
        )
        upload = gr.File(
            label="Audio files + manifest CSV, or a single ZIP",
            file_count="multiple",
            type="filepath",
        )
        run_button = gr.Button("Validate & run batch", variant="primary")
        status = gr.Textbox(label="Status", lines=6, interactive=False)
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            label="Results (review-flagged rows first)",
            interactive=False,
        )
        with gr.Row():
            csv_out = gr.File(label="Download results.csv")
            json_out = gr.File(label="Download results.json")
        scoring = gr.Textbox(
            label="Scoring metrics (shown when the manifest has labelled rows)",
            lines=12,
            interactive=False,
        )

        # Minimal wiring so construction stays valid now that run_batch is
        # gone. This mapping is wrong (enqueue_batch returns (job_id, status),
        # not the five outputs below) - Task 15 rewires the Blocks UI for the
        # async job queue properly.
        run_button.click(
            fn=enqueue_batch,
            inputs=[upload],
            outputs=[status, status],
        )

    return demo


if __name__ == "__main__":
    user = os.environ.get("AUTOACE_USER", "admin")
    password = os.environ.get("AUTOACE_PASSWORD")
    if not password:
        raise SystemExit(
            "AUTOACE_PASSWORD must be set in the environment - refusing to "
            "start a hosted dashboard with no login credential."
        )

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(user, password),
    )
