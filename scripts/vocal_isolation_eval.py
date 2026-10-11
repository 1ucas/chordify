"""Real-vocal benchmark for --isolate-instrumental, using open multitrack clips.

Chord charts for freely licensed recordings don't exist, so this measures
what vocal removal is for directly: how close each separator gets to the
chords the detector reads from the *true* vocal-free mix.

Data: the SiSEC18 / MUSDB18 7-second excerpts (CC BY 4.0, Zenodo record
3270814; ~147 MB, downloaded on first run into .cache/musdb18-7). Each clip
ships as separate mixture / drums / bass / other / vocals streams. The test
split is used (Demucs was not trained on it), and the clips with the most
prominent vocals are kept.

For each clip:
    reference  detector on drums+bass+other (the real stems, no vocals)
    raw        detector on the full mix
    demucs     detector after --separator demucs
    roformer   detector after --separator roformer
and each candidate is scored against the reference: frame agreement (0.1 s
frames after the first second) and chord-sequence agreement (consecutive
duplicates collapsed, 1 - edit distance / reference length).

Usage (needs ffmpeg and the separation extra):
    python scripts/vocal_isolation_eval.py                 # 12 clips, all configs
    python scripts/vocal_isolation_eval.py --clips 4 --configs raw roformer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chord_eval import _collapse, _edit_distance, normalize_detected  # noqa: E402

from chordify.detector import detect_chords  # noqa: E402

DATA_URL = "https://zenodo.org/records/3270814/files/MUSDB18-7-STEMS.zip?download=1"
CACHE_DIR = REPO_ROOT / ".cache" / "musdb18-7"
STEMS = ("mixture", "drums", "bass", "other", "vocals")  # MUSDB stream order
CONFIGS = ("raw", "demucs", "roformer")


def prepare_clips(n_clips: int) -> list[Path]:
    """Download, decode the test split to WAV, and return the most vocal clips."""
    zip_path = CACHE_DIR / "MUSDB18-7-STEMS.zip"
    wav_root = CACHE_DIR / "test"
    if not zip_path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading MUSDB18 7-second excerpts (~147 MB, CC BY 4.0)...", flush=True)
        part = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(DATA_URL, part)
        part.rename(zip_path)
    if not wav_root.exists():
        with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp:
            names = [n for n in archive.namelist() if n.startswith("test/") and n.endswith(".stem.mp4")]
            for name in names:
                stem_file = Path(archive.extract(name, tmp))
                clip_dir = wav_root / "".join(c if c.isalnum() else "_" for c in stem_file.name[:-9])
                clip_dir.mkdir(parents=True, exist_ok=True)
                for index, stem in enumerate(STEMS):
                    subprocess.run(
                        ["ffmpeg", "-v", "error", "-y", "-i", str(stem_file), "-map", f"0:{index}",
                         str(clip_dir / f"{stem}.wav")],
                        check=True,
                    )

    def vocal_prominence(clip: Path) -> float:
        vocals = sf.read(clip / "vocals.wav")[0]
        mix = sf.read(clip / "mixture.wav")[0]
        return float(np.sqrt(np.mean(vocals**2)) / (np.sqrt(np.mean(mix**2)) + 1e-9))

    clips = sorted((p for p in wav_root.iterdir() if p.is_dir()), key=vocal_prominence, reverse=True)
    return clips[:n_clips]


def _segments(chords) -> list[tuple[float, float, tuple[int, bool] | None]]:
    return [(c.start_time, c.end_time, normalize_detected(c.name)) for c in chords]


def _label_at(segments, t: float):
    for start, end, label in segments:
        if start <= t < end:
            return label
    return None


def detect(path: Path, config: str):
    kwargs = {} if config in ("raw", "reference") else {"isolate_instrumental": True, "separator": config}
    return _segments(detect_chords(path, **kwargs))


def reference_segments(clip: Path):
    stems = [sf.read(clip / f"{name}.wav", dtype="float32")[0] for name in ("drums", "bass", "other")]
    sr = sf.info(clip / "drums.wav").samplerate
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        sf.write(path, sum(stems), sr)
        return detect(path, "reference")
    finally:
        path.unlink(missing_ok=True)


def agreement(candidate, reference, duration: float) -> tuple[float, float]:
    frames = [t for t in np.arange(1.0, duration - 0.5, 0.1) if _label_at(reference, t) is not None]
    frame = sum(_label_at(candidate, t) == _label_at(reference, t) for t in frames) / len(frames)
    ref_seq = [label for _, label in _collapse([(s, label) for s, _, label in reference])]
    cand_seq = [label for _, label in _collapse([(s, label) for s, _, label in candidate])]
    seq = max(0.0, 1.0 - _edit_distance(cand_seq, ref_seq) / len(ref_seq))
    return frame, seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=int, default=12, help="number of most-vocal test clips")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--json-output", type=Path, help="also write per-clip results as JSON")
    args = parser.parse_args()

    clips = prepare_clips(args.clips)
    results: dict[str, dict[str, tuple[float, float]]] = {}
    print(f"{'clip':<40}" + "".join(f"{c:>16}" for c in args.configs) + "   (frame / seq)")
    for clip in clips:
        reference = reference_segments(clip)
        duration = sf.info(clip / "mixture.wav").duration
        row = {c: agreement(detect(clip / "mixture.wav", c), reference, duration) for c in args.configs}
        results[clip.name] = row
        print(f"{clip.name[:39]:<40}" + "".join(f"{row[c][0]:>9.1%} /{row[c][1]:>4.0%}" for c in args.configs),
              flush=True)

    means = {c: [float(np.mean([results[k][c][i] for k in results])) for i in (0, 1)] for c in args.configs}
    print(f"{'MEAN':<40}" + "".join(f"{means[c][0]:>9.1%} /{means[c][1]:>4.0%}" for c in args.configs))
    if args.json_output:
        args.json_output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
