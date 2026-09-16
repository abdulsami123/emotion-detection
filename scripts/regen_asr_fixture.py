"""Regenerate tests/fixtures/asr_baseline.json.

Recreates the ASR evidence that several design decisions rest on: the
transcripts behind the prosody-primary architecture, the per-call detected
language, and the measured CPU wall-clock used in the latency model.

Explicit UTF-8 on write - the Spanish call contains characters that the
Windows default codepage cannot encode.
"""

import json
import time
from pathlib import Path

from emotion_detection.asr import transcribe
from emotion_detection.config import reference_call

CALLS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "asr_baseline.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in CALLS:
        started = time.time()
        transcript = transcribe(reference_call(name))
        elapsed = time.time() - started
        results[name] = {
            "language": transcript.language,
            "elapsed_s": round(elapsed, 2),
            "avg_logprob": round(transcript.avg_logprob, 3),
            "n_words": len(transcript.words),
            "n_segments": len(transcript.segments),
            "first_word_start": transcript.words[0].start if transcript.words else None,
            "last_word_end": transcript.words[-1].end if transcript.words else None,
            "text": transcript.text,
        }
        print(f"{name}: lang={transcript.language} elapsed={elapsed:.1f}s "
              f"words={len(transcript.words)}", flush=True)

    OUT.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
