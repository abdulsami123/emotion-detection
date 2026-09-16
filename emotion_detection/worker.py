"""The drain loop.

Runs as its own systemd unit rather than a thread or a forked child. A thread
is ruled out because openSMILE and the DSP paths do not reliably release the
GIL, so a worker thread would stall the UI. A separate unit additionally buys
independent restarts and independent journald streams.

Model weights load lazily on the first call and stay resident, which is why
this process - not the web process - is the one that needs the memory.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from typing import Callable

from emotion_detection import jobs
from emotion_detection.config import EXPIRY_SWEEP_SECONDS, MAX_ATTEMPTS, WORKER_POLL_SECONDS

log = logging.getLogger("emotion_detection.worker")

Analyser = Callable[[str], object]


def _default_analyser(path: str):
    """Imported lazily so `import emotion_detection.worker` stays cheap - the tests that
    inject a fake analyser must not pay for loading torch."""
    from emotion_detection.pipeline import analyse_file

    return analyse_file(path)


def drain_once(conn: sqlite3.Connection, analyse: Analyser | None = None) -> bool:
    """Process at most one file. Returns True if work was claimed.

    `analyse_file` is documented never to raise, so an exception escaping it is
    unexpected rather than routine - it earns a retry, bounded by MAX_ATTEMPTS,
    because the alternative to a bound is a crash loop: an OOM-killed file gets
    requeued and dies again on the same file forever.
    """
    analyse = analyse or _default_analyser

    claimed = jobs.claim_next(conn)
    if claimed is None:
        return False

    log.info(
        "analysing %s (job %s, attempt %d)",
        claimed.name, claimed.job_id, claimed.attempts,
    )
    try:
        result = analyse(claimed.audio_path)
    except Exception as exc:  # noqa: BLE001 - failure isolation is the point
        if claimed.attempts >= MAX_ATTEMPTS:
            log.exception(
                "giving up on %s after %d attempts", claimed.name, claimed.attempts
            )
            jobs.fail(
                conn, claimed,
                f"unexpected failure after {claimed.attempts} attempts: {exc}",
            )
        else:
            log.exception("retrying %s after unexpected failure", claimed.name)
            jobs.reconcile(conn, stale_after=0.0)
        return True

    jobs.complete(conn, claimed, result)
    log.info("recorded %s", claimed.name)
    return True


def run_forever(conn: sqlite3.Connection | None = None) -> None:
    """Reconcile, then drain, then sweep expired jobs, forever.

    `reconcile(stale_after=0.0)` on startup is safe because this is the only
    worker: nothing can legitimately be `running` when it begins.
    """
    conn = conn or jobs.connect()

    requeued, failed, orphaned = jobs.reconcile(conn, stale_after=0.0)
    if requeued or failed or orphaned:
        log.warning(
            "startup reconcile: %d requeued, %d abandoned past the attempt cap, "
            "%d orphaned job(s) finalised",
            requeued, failed, orphaned,
        )

    last_sweep = 0.0
    while True:
        did_work = drain_once(conn)

        now = time.time()
        if now - last_sweep > EXPIRY_SWEEP_SECONDS:
            removed = jobs.expire(conn)
            if removed:
                log.info("expired %d job(s) past the retention window", removed)
            last_sweep = now

        if not did_work:
            time.sleep(WORKER_POLL_SECONDS)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
