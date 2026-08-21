"""The hosted dashboard (brief section 7).

Every requirement in that section maps to one function here, kept separate
from the Gradio wiring so each can be unit-tested without a browser or a
running server:

  - `validate_batch`   -> validation BEFORE processing (missing/unlisted/
                           unsupported files reported as a list).
  - `enqueue_batch` /
    `poll_job`          -> visible progress + per-file failure isolation, via
                           an async job queue rather than a blocking request.
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
from autoace.config import (
    DATA_DIR,
    EXPORTS_DIR,
    MAX_UPLOAD_MB,
    REVIEW_THRESHOLD,
    UI_POLL_SECONDS,
)
from autoace.eval import load_manifest, score_batch
from autoace.io_audio import SUPPORTED_SUFFIXES
from autoace.pipeline import FileResult, analyse_file
from autoace.schema import CallAnalysis

# "name" first, then the nine schema fields, in schema declaration order.
SCHEMA_COLUMNS = ("name",) + tuple(CallAnalysis.model_fields.keys())

# One column per schema field plus the filename, the review flag, and the tone
# path. Brief section 7 requires the DISPLAYED prediction to use the required
# output schema, so a field that is exported but not shown does not count -
# background_noise_present, speaker_overlap_present, and long_silence_present
# were previously missing here.
TABLE_HEADERS = [
    "file",
    "tone",
    "intensity",
    "noise?",
    "noise type",
    "severity",
    "quality",
    "overlap?",
    "long silence?",
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


def _yn(value: bool) -> str:
    """Booleans render as yes/no. A bare True next to enum strings like
    `slightly_impaired` is hard to scan in a 50-row table."""
    return "yes" if value else "no"


def results_to_table(results: list[FileResult]) -> list[list]:
    """Display rows for the review queue.

    Review-flagged (or outright failed) rows sort to the top, then ascending
    confidence - a degraded result must be visible, not buried at the bottom of
    a 50-row batch.

    Every row has exactly `len(TABLE_HEADERS)` cells, including error and
    pending rows: Gradio mangles a dataframe with ragged rows rather than
    complaining. A pending file is labelled `queued`, not `ERROR` - it has not
    failed, it has not run.
    """
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
                    _yn(data["background_noise_present"]),
                    data["background_noise_type"],
                    data["background_noise_severity"],
                    data["audio_quality"],
                    _yn(data["speaker_overlap_present"]),
                    _yn(data["long_silence_present"]),
                    round(confidence, 3),
                    "REVIEW" if flagged else "",
                    result.reasoning,
                ]
            )
        else:
            state = "ERROR" if result.error else "queued"
            # len - 4: filename, then the confidence/flag/notes trio are supplied
            # explicitly. Derived so adding a column cannot silently produce
            # a ragged row.
            blanks = [""] * (len(TABLE_HEADERS) - 4)
            rows.append([result.name] + blanks + [None, state, result.error or ""])

    def sort_key(row):
        flagged = row[-2] in ("REVIEW", "ERROR")
        confidence = row[-3] if isinstance(row[-3], (int, float)) else -1.0
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
    """Human-readable problems (and, for a missing manifest, plain
    information) found before any inference runs.

    Extracted from the old `run_batch` unchanged in substance: the brief
    requires unmatched files to be reported rather than silently skipped, and
    that has to happen at enqueue time, not when the worker gets there.

    A missing manifest is no longer treated as an error - the brief never
    says to reject a batch for lacking one, and `enqueue_batch` now processes
    every discovered clip in that case. So this reports it as information
    (what will happen, and that scoring needs a manifest), not a problem.
    """
    problems: list[str] = []
    if validation.manifest_path is None:
        clip_count = sum(1 for _ in _iter_audio_files(workdir))
        if clip_count:
            problems.append(
                f"No CSV manifest found - processing all {clip_count} "
                "discovered audio clip(s). Nothing to cross-check, so no "
                "scoring is available; include a manifest with a populated "
                "`result_json` column to get scoring metrics."
            )
        else:
            problems.append(
                "No CSV manifest found, and no supported audio files were "
                "found either - nothing to process."
            )
    if validation.missing_audio:
        problems.append(
            "Manifest rows with no matching audio file: "
            + ", ".join(validation.missing_audio)
        )
    if validation.unlisted_audio and validation.manifest_path is not None:
        # With no manifest at all, `unlisted_audio` is every audio file (there
        # is nothing to list it against) - and unlike the manifest-present
        # case, those files ARE processed, so the "will not be processed"
        # wording would be wrong. The no-manifest message above already
        # covers it.
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

    The manifest is optional, not mandatory: the brief's format includes one,
    and validation always checks against it when present, but nothing in the
    brief says to reject an upload for lacking one - and for an unlabeled
    hidden test set the manifest may carry nothing scoring needs anyway.

      - Manifest present: only `validation.matched` is queued, exactly as
        before. Zero matches means every row references audio that is not in
        the upload - a genuine batch-authoring error, so that still fails.
      - No manifest: every supported audio file discovered is queued,
        unlabelled, with nothing to cross-check. Only an upload with no
        audio either fails.
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
        if validation.manifest_path is not None:
            if not validation.matched:
                reason = "Validation failed - nothing to process.\n" + "\n".join(
                    f"- {p}" for p in problems
                )
                job_id = jobs.enqueue_failed(
                    conn, workdir=workdir, validation_json=validation_json,
                    error=reason,
                )
                return job_id, reason

            audio_by_name: dict[str, Path] = {}
            for path in _iter_audio_files(workdir):
                audio_by_name.setdefault(path.name, path)
            queued_audio = {name: audio_by_name[name] for name in validation.matched}

            manifest_csv = validation.manifest_path.read_text(encoding="utf-8")
            manifest_labelled = any(
                row.expected is not None
                for row in load_manifest(str(validation.manifest_path))
            )
        else:
            queued_audio = {}
            for path in _iter_audio_files(workdir):
                queued_audio.setdefault(path.name, path)

            if not queued_audio:
                reason = "Validation failed - nothing to process.\n" + "\n".join(
                    f"- {p}" for p in problems
                )
                job_id = jobs.enqueue_failed(
                    conn, workdir=workdir, validation_json=validation_json,
                    error=reason,
                )
                return job_id, reason

            manifest_csv = None
            manifest_labelled = False

        job_id = jobs.enqueue(
            conn,
            workdir=workdir,
            audio_by_name=queued_audio,
            validation_json=validation_json,
            manifest_labelled=manifest_labelled,
            manifest_csv=manifest_csv,
        )
    finally:
        conn.close()

    lines = [
        f"**Queued {len(queued_audio)} file(s).** Job ID: `{job_id}`",
        "",
        "Keep this job ID. You can close the page and paste it into the "
        "**Look up a job** box to come back to these results.",
    ]
    if problems:
        lines.append("")
        lines.append("Validation notes:")
        lines.extend(f"- {p}" for p in problems)
    return job_id, "\n".join(lines)


def _write_exports(job_id: str, results: list[FileResult]) -> tuple[str, str]:
    """Write results.csv / results.json to a stable per-job directory.

    Deliberately NOT `tempfile.mkdtemp`. `poll_job` runs on every `gr.Timer`
    tick, so a fresh temp directory per call leaked one every
    UI_POLL_SECONDS for as long as a tab stayed open - more than a thousand
    across an 87-minute batch, and still growing afterwards because the timer
    keeps firing once a job is complete. Overwriting one directory per job
    bounds it, and `jobs.expire` removes it with the job row.
    """
    out_dir = EXPORTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    csv_path.write_text(results_to_csv(results), encoding="utf-8")
    json_path.write_text(results_to_json(results), encoding="utf-8")
    return str(csv_path), str(json_path)


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

    table = results_to_table(results)

    csv_path, json_path = _write_exports(job_id, results)

    scoring_text = ""
    if status.manifest_labelled and status.status == "complete" and status.manifest_csv:
        # load_manifest reads a path, and the original manifest went away with
        # the workdir - so round-trip the persisted text through a temp file
        # rather than duplicating the CSV parsing here.
        manifest_dir = Path(tempfile.mkdtemp(prefix="autoace_manifest_"))
        manifest_file = manifest_dir / "manifest.csv"
        manifest_file.write_text(status.manifest_csv, encoding="utf-8")
        try:
            manifest_rows = load_manifest(str(manifest_file))
            predictions = {
                r.name: r.analysis for r in results if r.analysis is not None
            }
            if predictions:
                metrics = score_batch(manifest_rows, predictions)
                scoring_text = json.dumps(metrics, indent=2, default=str)
        finally:
            shutil.rmtree(manifest_dir, ignore_errors=True)

    return "\n".join(lines), table, str(csv_path), str(json_path), scoring_text


def build_app() -> gr.Blocks:
    """The UI. Construction only - never calls `.launch()`.

    Asynchronous by necessity: a 50-file batch is ~2.7 hours on the deployment
    hardware, so the upload handler only enqueues and a `gr.Timer` polls the
    job store. Closing the tab stops the timer but not the worker, which is
    why the job-ID box exists.
    """
    with gr.Blocks(title="AutoAce - Call Tone Review") as demo:
        gr.Markdown(
            "# AutoAce batch review\n"
            "Upload a folder or ZIP containing audio files. A CSV manifest "
            "(`name,result_json`) is optional - include one to cross-check "
            "filenames and get scoring metrics when `result_json` is "
            "populated; without one, every supported clip found is "
            "processed. Validation runs before any inference. Processing "
            "happens in the background at roughly "
            "**192 s per file**, so a 50-file batch takes about 2.7 hours - "
            "**keep the job ID**, close the page if you like, and paste the ID "
            "back in to return to your results. Rows flagged for human review "
            "(low confidence or conflicting signals) sort to the top."
        )

        job_state = gr.State("")

        upload = gr.File(
            label="Audio files + manifest CSV, or a single ZIP",
            file_count="multiple",
            type="filepath",
        )
        run_button = gr.Button("Validate & queue batch", variant="primary")

        with gr.Row():
            job_box = gr.Textbox(
                label="Look up a job",
                placeholder="Paste a job ID to resume watching it",
                scale=4,
            )
            lookup_button = gr.Button("Look up", scale=1)

        status = gr.Markdown()
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            label="Results (review-flagged rows first)",
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            csv_out = gr.File(label="Download results.csv")
            json_out = gr.File(label="Download results.json")
        scoring = gr.Textbox(
            label="Scoring metrics (shown when the manifest has labelled rows)",
            lines=12,
            interactive=False,
        )

        timer = gr.Timer(UI_POLL_SECONDS)
        poll_outputs = [status, table, csv_out, json_out, scoring]

        def _submit(files):
            """Queue the batch and seed both the polling state and the lookup
            box, so the ID is visible and selectable rather than only mentioned
            in the status text."""
            job_id, message = enqueue_batch(files)
            return job_id, job_id, message

        run_button.click(
            fn=_submit,
            inputs=[upload],
            outputs=[job_state, job_box, status],
        )

        lookup_button.click(
            fn=lambda jid: (jid or "").strip(),
            inputs=[job_box],
            outputs=[job_state],
        ).then(fn=poll_job, inputs=[job_state], outputs=poll_outputs)

        timer.tick(fn=poll_job, inputs=[job_state], outputs=poll_outputs)

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
        # 127.0.0.1, not 0.0.0.0: Caddy terminates TLS on the VM and is the
        # only listener reachable from off-box, so a wrong firewall rule
        # cannot expose the app over plaintext HTTP.
        server_name=os.environ.get("AUTOACE_BIND", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        auth=(user, password),
        max_file_size=f"{MAX_UPLOAD_MB}mb",
        # Exports live under DATA_DIR now (see _write_exports), so gradio has
        # to be allowed to serve from there.
        allowed_paths=[str(DATA_DIR)],
    )
