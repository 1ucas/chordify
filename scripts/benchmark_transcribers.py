"""Benchmark Chordify transcription backends on a local audio file."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chordify.transcriber import (  # noqa: E402
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_QWEN3_MODEL,
    DEFAULT_WHISPER_MODEL,
    transcribe_audio,
)


def audio_duration(audio_path: Path) -> float:
    """Read audio duration in seconds."""
    import soundfile as sf

    info = sf.info(str(audio_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def normalized_words(text: str) -> list[str]:
    """Normalize ASR text for optional word error rate scoring."""
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute a dependency-free word error rate."""
    reference_words = normalized_words(reference)
    hypothesis_words = normalized_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for i, reference_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1] / len(reference_words)


def benchmark_backend(
    audio_path: Path,
    backend: str,
    model_name: str,
    language: str | None,
    duration_seconds: float,
    reference_text: str | None,
    verbose: bool,
) -> dict[str, Any]:
    """Run one backend/model pair and capture benchmark metrics."""
    started = time.perf_counter()
    try:
        result = transcribe_audio(
            audio_path=audio_path,
            language=language,
            model_name=model_name,
            verbose=verbose,
            backend=backend,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going
        elapsed_seconds = time.perf_counter() - started
        return {
            "backend": backend,
            "model": model_name,
            "elapsed_seconds": elapsed_seconds,
            "rtf": elapsed_seconds / duration_seconds if duration_seconds else None,
            "words": None,
            "characters": None,
            "language": None,
            "wer": None,
            "error": str(exc),
        }

    elapsed_seconds = time.perf_counter() - started
    transcript = result.full_text or " ".join(word.text for word in result.words)
    return {
        "backend": backend,
        "model": model_name,
        "elapsed_seconds": elapsed_seconds,
        "rtf": elapsed_seconds / duration_seconds if duration_seconds else None,
        "words": len(result.words),
        "characters": len(transcript),
        "language": result.language,
        "wer": word_error_rate(reference_text, transcript) if reference_text else None,
        "error": None,
    }


def print_rows(rows: list[dict[str, Any]]) -> None:
    """Print benchmark results as a compact table."""
    headers = ("backend", "model", "sec", "rtf", "words", "chars", "lang", "wer", "error")
    print(
        f"{headers[0]:<10} {headers[1]:<36} {headers[2]:>9} {headers[3]:>7} "
        f"{headers[4]:>7} {headers[5]:>7} {headers[6]:<6} {headers[7]:>7} {headers[8]}"
    )
    print("-" * 112)
    for row in rows:
        elapsed = f"{row['elapsed_seconds']:.2f}"
        rtf = f"{row['rtf']:.3f}" if row["rtf"] is not None else "-"
        words = str(row["words"]) if row["words"] is not None else "-"
        characters = str(row["characters"]) if row["characters"] is not None else "-"
        language = row["language"] or "-"
        wer = f"{row['wer']:.3f}" if row["wer"] is not None else "-"
        error = row["error"] or ""
        model = row["model"]
        if len(model) > 36:
            model = f"...{model[-33:]}"
        print(
            f"{row['backend']:<10} {model:<36} {elapsed:>9} {rtf:>7} "
            f"{words:>7} {characters:>7} {language:<6} {wer:>7} {error}"
        )


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="Local audio file to transcribe")
    parser.add_argument(
        "--backend",
        action="append",
        choices=("whisper", "parakeet", "qwen3"),
        help="Backend to run. Pass multiple times; defaults to all three.",
    )
    parser.add_argument("--language", help="Optional language hint")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--parakeet-model", default=DEFAULT_PARAKEET_MODEL)
    parser.add_argument("--qwen3-model", default=DEFAULT_QWEN3_MODEL)
    parser.add_argument("--reference-text", type=Path, help="Optional lyrics text for WER")
    parser.add_argument("--json-output", type=Path, help="Optional path to save JSON results")
    parser.add_argument("--verbose", action="store_true", help="Show backend progress")
    return parser.parse_args()


def main() -> int:
    """Run transcription benchmarks."""
    args = parse_args()
    if not args.audio_path.exists():
        print(f"Audio file does not exist: {args.audio_path}", file=sys.stderr)
        return 2

    backends = args.backend or ["whisper", "parakeet", "qwen3"]
    model_by_backend = {
        "whisper": args.whisper_model,
        "parakeet": args.parakeet_model,
        "qwen3": args.qwen3_model,
    }
    reference_text = args.reference_text.read_text() if args.reference_text else None
    duration_seconds = audio_duration(args.audio_path)

    rows = [
        benchmark_backend(
            audio_path=args.audio_path,
            backend=backend,
            model_name=model_by_backend[backend],
            language=args.language,
            duration_seconds=duration_seconds,
            reference_text=reference_text,
            verbose=args.verbose,
        )
        for backend in backends
    ]

    print_rows(rows)

    if args.json_output:
        args.json_output.write_text(json.dumps(rows, indent=2))
        print(f"\nSaved JSON results to: {args.json_output}")

    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
