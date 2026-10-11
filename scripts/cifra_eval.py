"""Score detected chords against a cifra (chord chart) with exact matching.

A cifra gives the chords in playing order but no timings, so the scores are
timing-free:

    seq    chord-sequence accuracy: consecutive duplicates collapsed on both
           sides, then 1 - edit distance / cifra length. Every wrong, extra or
           missing chord counts.
    vocab  share of detected song time spent on a chord that appears in the
           cifra.

Matching is exact on the chord symbol: C, C#, Cm, C9 and C/E are all
different. Only the spelling of the same pitch is unified (C# == Db), and
nothing is transposed, so a recording in another key than the cifra scores 0.

Paste the cifra as-is into a text file (lyrics and tab lines are ignored; a
line counts only if every token is a chord symbol, optionally after a
``[Section]`` tag). Write repeats out in full.

Usage:
    python scripts/cifra_eval.py song.wav --cifra cifra.txt
    python scripts/cifra_eval.py "https://youtu.be/..." --cifra cifra.txt --configs raw roformer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chord_eval import NOTE_TO_PC, _collapse, _edit_distance  # noqa: E402

CHORD_TOKEN = re.compile(r"^([A-G][#b]?)([a-zA-Z0-9+#()º°-]*?)(?:/([A-G][#b]?))?$")
SECTION_TAG = re.compile(r"^\s*\[[^\]]*\]")
CONFIGS = ("raw", "demucs", "roformer")

Label = tuple[int, str, int | None]  # (root pitch class, suffix, bass pitch class)


def parse_symbol(symbol: str) -> Label | None:
    match = CHORD_TOKEN.match(symbol)
    if not match:
        return None
    bass = NOTE_TO_PC[match.group(3)] if match.group(3) else None
    return NOTE_TO_PC[match.group(1)], match.group(2), bass


def parse_cifra(text: str) -> list[Label]:
    """Chords in playing order from every line made only of chord symbols."""
    chords: list[Label] = []
    for line in text.splitlines():
        tokens = SECTION_TAG.sub("", line).split()
        labels = [parse_symbol(t) for t in tokens]
        if tokens and all(labels):
            chords.extend(labels)
    return chords


def score(detected, cifra: list[Label]) -> dict:
    segments = [(c.start_time, c.end_time, parse_symbol(c.name)) for c in detected if c.name != "N"]
    detected_seq = [label for _, label in _collapse([(s, label) for s, _, label in segments])]
    cifra_seq = [label for _, label in _collapse([(i, label) for i, label in enumerate(cifra)])]
    vocabulary = set(cifra)
    total = sum(e - s for s, e, _ in segments)
    in_vocab = sum(e - s for s, e, label in segments if label in vocabulary)
    return {
        "seq": max(0.0, 1.0 - _edit_distance(detected_seq, cifra_seq) / len(cifra_seq)),
        "vocab": in_vocab / total if total else 0.0,
        "detected_changes": len(detected_seq),
        "cifra_changes": len(cifra_seq),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="audio file or YouTube URL")
    parser.add_argument("--cifra", type=Path, required=True, help="cifra text file")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--show", action="store_true", help="print each detected chord sequence")
    args = parser.parse_args()

    from chordify.detector import detect_chords

    if args.audio.startswith("http"):
        from chordify.downloader import download_audio

        audio_path = download_audio(args.audio).audio_path
    else:
        audio_path = Path(args.audio)

    cifra = parse_cifra(args.cifra.read_text())
    if not cifra:
        raise SystemExit("No chord lines found in the cifra file")
    print(f"cifra: {len(cifra)} chords, vocabulary {len(set(cifra))} symbols")

    print(f"{'config':<10}{'seq':>8}{'vocab':>8}  changes(det/cifra)")
    for config in args.configs:
        kwargs = {} if config == "raw" else {"isolate_instrumental": True, "separator": config}
        detected = detect_chords(audio_path, **kwargs)
        result = score(detected, cifra)
        print(
            f"{config:<10}{result['seq']:>8.1%}{result['vocab']:>8.1%}"
            f"  {result['detected_changes']}/{result['cifra_changes']}",
            flush=True,
        )
        if args.show:
            print("   ", " ".join(c.name for c in detected if c.name != "N"))


if __name__ == "__main__":
    main()
