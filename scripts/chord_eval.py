"""Ground-truth chord-detection benchmark on synthesized pop-style songs.

Since real recordings with verified charts can't ship with the repo, this
script renders small "songs" (piano comping + bass line + melody + drums)
from known chord progressions and scores the detector frame-by-frame
against the ground truth. Extensions and slash basses are stripped before
comparison, so scoring is on the root + major/minor quality only.

Usage:
    python scripts/chord_eval.py            # run the full suite
    python scripts/chord_eval.py --song axis-g   # run one song
    python scripts/chord_eval.py --dump    # also print detected vs truth timeline
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chordify.detector import detect_chords  # noqa: E402

SR = 22050

NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

CHORD_RE = re.compile(r"^([A-G][#b]?)(m?)")


def parse_chord(name: str) -> tuple[int, bool]:
    """Return (root pitch class, is_minor) for a plain triad name."""
    match = CHORD_RE.match(name)
    if not match:
        raise ValueError(f"Unparseable chord: {name}")
    return NOTE_TO_PC[match.group(1)], match.group(2) == "m"


def normalize_detected(name: str) -> tuple[int, bool] | None:
    """Reduce a detected label (may carry sus/7/9//bass) to (root, is_minor)."""
    if name == "N":
        return None
    base = name.split("/")[0]
    match = re.match(r"^([A-G][#b]?)(m(?!aj))?", base)
    if not match:
        return None
    return NOTE_TO_PC[match.group(1)], match.group(2) is not None


def chord_pitches(root: int, minor: bool, octave: int = 3) -> list[float]:
    """MIDI numbers for a closed triad voicing."""
    base = 12 * (octave + 1) + root
    third = base + (3 if minor else 4)
    fifth = base + 7
    return [base, third, fifth]


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def piano_note(freq: float, dur: float, sr: int = SR, amp: float = 1.0) -> np.ndarray:
    """Decaying harmonic tone, roughly piano/guitar-like."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k, gain in enumerate([1.0, 0.55, 0.33, 0.2, 0.12, 0.08], start=1):
        out += gain * np.sin(2 * np.pi * freq * k * t)
    env = np.exp(-t * 2.2) * np.minimum(t * 200, 1.0)
    return amp * out * env


def bass_note(freq: float, dur: float, sr: int = SR, amp: float = 1.0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2 * t)
    env = np.exp(-t * 3.0) * np.minimum(t * 150, 1.0)
    return amp * out * env


def drum_hits(total_len: int, beat_samples: int, sr: int = SR) -> np.ndarray:
    """Kick on every beat, hat on every half beat, snare on 2 and 4."""
    rng = np.random.default_rng(1234)
    out = np.zeros(total_len)
    n_beats = total_len // beat_samples
    for b in range(n_beats):
        start = b * beat_samples
        # Kick: low decaying sine sweep
        n = min(int(0.09 * sr), total_len - start)
        t = np.arange(n) / sr
        out[start:start + n] += 0.9 * np.sin(2 * np.pi * (55 + 30 * np.exp(-t * 40)) * t) * np.exp(-t * 28)
        if b % 2 == 1:  # snare
            n = min(int(0.08 * sr), total_len - start)
            out[start:start + n] += 0.45 * rng.standard_normal(n) * np.exp(-np.arange(n) / sr * 45)
        for half in (0, 1):  # hats
            hs = start + half * beat_samples // 2
            n = min(int(0.03 * sr), total_len - hs)
            if n > 0:
                out[hs:hs + n] += 0.18 * rng.standard_normal(n) * np.exp(-np.arange(n) / sr * 130)
    return out


@dataclass
class Song:
    name: str
    # Sequence of (chord_name, beats)
    progression: list[tuple[str, int]]
    bpm: float
    repeats: int = 4
    melody_octave: int = 5
    detune_cents: float = 0.0
    seed: int = 42
    truth: list[tuple[float, float, str]] = field(default_factory=list)

    def render(self) -> np.ndarray:
        beat = 60.0 / self.bpm
        beat_samples = int(beat * SR)
        rng = np.random.default_rng(self.seed)
        detune = 2 ** (self.detune_cents / 1200)

        events: list[tuple[str, int]] = []
        for _ in range(self.repeats):
            events.extend(self.progression)

        total_beats = sum(b for _, b in events)
        total_len = total_beats * beat_samples + int(1.5 * SR)
        chords_track = np.zeros(total_len)
        bass_track = np.zeros(total_len)
        melody_track = np.zeros(total_len)

        self.truth = []
        cursor_beats = 0
        for chord_name, beats in events:
            root, minor = parse_chord(chord_name)
            start = cursor_beats * beat_samples
            t0 = cursor_beats * beat
            self.truth.append((t0, t0 + beats * beat, chord_name))

            pitches = chord_pitches(root, minor, octave=3)
            # Comping: restrike the triad every beat
            for b in range(beats):
                s = start + b * beat_samples
                for midi in pitches:
                    tone = piano_note(midi_to_hz(midi) * detune, beat * 1.1, amp=0.5)
                    end = min(s + len(tone), total_len)
                    chords_track[s:end] += tone[: end - s]
            # Bass: root on 1/3, fifth on 2/4
            for b in range(beats):
                s = start + b * beat_samples
                midi = 12 * 3 + root if b % 2 == 0 else 12 * 3 + (root + 7) % 12
                tone = bass_note(midi_to_hz(midi) * detune, beat * 0.95, amp=0.8)
                end = min(s + len(tone), total_len)
                bass_track[s:end] += tone[: end - s]
            # Melody: chord tones + occasional stepwise passing notes
            chord_tone_pcs = [root, (root + (3 if minor else 4)) % 12, (root + 7) % 12]
            for b in range(beats * 2):  # eighth-note-ish melody at half-beat grid
                s = start + b * beat_samples // 2
                if rng.random() < 0.25:
                    continue  # rest
                if rng.random() < 0.25:
                    pc = (rng.choice(chord_tone_pcs) + rng.choice([-1, 1, 2])) % 12
                else:
                    pc = rng.choice(chord_tone_pcs)
                midi = 12 * (self.melody_octave + 1) + int(pc)
                tone = piano_note(midi_to_hz(midi) * detune, beat * 0.6, amp=0.35)
                end = min(s + len(tone), total_len)
                melody_track[s:end] += tone[: end - s]

            cursor_beats += beats

        drums = drum_hits(total_beats * beat_samples, beat_samples)
        mix = chords_track + bass_track + melody_track
        mix[: len(drums)] += 0.7 * drums
        mix += 0.004 * rng.standard_normal(total_len)
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = 0.9 * mix / peak
        return mix.astype(np.float32)


SONGS = [
    # Classic I-V-vi-IV pop in G ("axis of awesome" progression)
    Song("axis-g", [("G", 4), ("D", 4), ("Em", 4), ("C", 4)], bpm=120),
    # Let It Be-style verse in C
    Song("ballad-c", [("C", 4), ("G", 4), ("Am", 4), ("F", 4),
                      ("C", 4), ("G", 4), ("F", 2), ("C", 2)], bpm=74),
    # vi-IV-I-V minor-feel pop in A minor (Save Tonight / Zombie shape)
    Song("minor-am", [("Am", 4), ("F", 4), ("C", 4), ("G", 4)], bpm=126),
    # 50s doo-wop I-vi-IV-V in Eb (flat key + slight detune)
    Song("doowop-eb", [("Eb", 4), ("Cm", 4), ("Ab", 4), ("Bb", 4)], bpm=100,
         detune_cents=12.0),
    # Faster two-beat changes in D (tests temporal resolution)
    Song("fast-d", [("D", 2), ("A", 2), ("Bm", 2), ("G", 2),
                    ("D", 2), ("G", 2), ("A", 2), ("D", 2)], bpm=132, repeats=6),
    # E major with borrowed bVII (tests non-diatonic handling)
    Song("borrowed-e", [("E", 4), ("B", 4), ("C#m", 4), ("A", 4),
                        ("E", 4), ("D", 4), ("A", 4), ("E", 4)], bpm=112),
]


def evaluate_song(song: Song, dump: bool = False, hop: float = 0.1) -> dict:
    audio = song.render()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    sf.write(wav_path, audio, SR)
    try:
        detected = detect_chords(wav_path, verbose=False)
    finally:
        wav_path.unlink(missing_ok=True)

    def truth_at(t: float) -> tuple[int, bool] | None:
        for start, end, name in song.truth:
            if start <= t < end:
                return parse_chord(name)
        return None

    def detected_at(t: float) -> tuple[int, bool] | None:
        for chord in detected:
            if chord.start_time <= t < chord.end_time:
                return normalize_detected(chord.name)
        return None

    total = correct = root_correct = no_chord = 0
    end_time = song.truth[-1][1]
    # Skip the first second (beat tracker warm-up) like a human transcriber would.
    t = 1.0
    timeline = []
    while t < end_time - 0.5:
        truth = truth_at(t)
        if truth is not None:
            got = detected_at(t)
            total += 1
            if got is None:
                no_chord += 1
            else:
                if got == truth:
                    correct += 1
                if got[0] == truth[0]:
                    root_correct += 1
            timeline.append((t, truth, got))
        t += hop

    if dump:
        last = None
        for t, truth, got in timeline:
            row = (truth, got)
            if row != last:
                mark = "  " if truth == got else "XX"
                got_s = "N" if got is None else f"{got[0]}{'m' if got[1] else ''}"
                print(f"  {mark} {t:7.2f}s truth={truth[0]}{'m' if truth[1] else ''}  got={got_s}")
                last = row
    return {
        "song": song.name,
        "accuracy": correct / total if total else 0.0,
        "root_accuracy": root_correct / total if total else 0.0,
        "no_chord_rate": no_chord / total if total else 0.0,
        "n_detected": len(detected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song", help="run only the named song")
    parser.add_argument("--dump", action="store_true", help="print detected-vs-truth timeline")
    args = parser.parse_args()

    songs = [s for s in SONGS if args.song is None or s.name == args.song]
    if not songs:
        raise SystemExit(f"Unknown song {args.song!r}; choices: {[s.name for s in SONGS]}")

    results = []
    for song in songs:
        result = evaluate_song(song, dump=args.dump)
        results.append(result)
        print(
            f"{result['song']:<12} accuracy={result['accuracy']:.3f} "
            f"root={result['root_accuracy']:.3f} "
            f"no-chord={result['no_chord_rate']:.3f} "
            f"changes={result['n_detected']}"
        )

    mean = float(np.mean([r["accuracy"] for r in results]))
    print(f"{'MEAN':<12} accuracy={mean:.3f}")


if __name__ == "__main__":
    main()
