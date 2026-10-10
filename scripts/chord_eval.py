"""Ground-truth chord-detection benchmark on synthesized pop-style songs.

Since real recordings with verified charts can't ship with the repo, this
script renders small "songs" (piano comping + bass line + melody + drums)
from known chord progressions and scores the detector against the ground
truth. Extensions and slash basses are stripped before comparison, so
scoring is on the root + major/minor quality only.

Metrics per song:
    frame   share of 0.1 s frames with the right chord (forgiving: a late
            change or a one-beat flicker barely moves it)
    seq     chord-sequence accuracy: consecutive duplicates are collapsed on
            both sides and scored with edit distance (1 - chord error rate),
            so every spurious, missing or wrong change counts
    F1      chord-change F1: a detected change must land within +/-0.25 s of
            a true change
    offset  mean timing error of the matched changes, in ms

Suites: ``basic`` is the original six songs (block piano comping); ``hard``
adds 7th-chord voicings, inversions, changes pushed off the beat, one chord
per beat, arpeggiated comping and a sustained vibrato "vocal" lead that holds
notes across chord changes.

Usage:
    python scripts/chord_eval.py                 # run every suite
    python scripts/chord_eval.py --suite hard    # only the hard songs
    python scripts/chord_eval.py --song axis-g   # run one song
    python scripts/chord_eval.py --dump          # also print detected vs truth timeline
    python scripts/chord_eval.py --isolate roformer   # score through vocal removal
"""

from __future__ import annotations

import argparse
import json
import math
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

CHORD_RE = re.compile(r"^([A-G][#b]?)(maj7|m7|m|7)?(?:/([A-G][#b]?))?$")


@dataclass(frozen=True)
class ChordSpec:
    root: int
    minor: bool
    intervals: tuple[int, ...]  # semitones above the root
    bass: int  # pitch class played by the bass (differs for inversions)


def chord_spec(name: str) -> ChordSpec:
    """Parse ``C``, ``Am``, ``G7``, ``Dm7``, ``Cmaj7`` and slash forms like ``D/F#``."""
    match = CHORD_RE.match(name)
    if not match:
        raise ValueError(f"Unparseable chord: {name}")
    root = NOTE_TO_PC[match.group(1)]
    quality = match.group(2) or ""
    minor = quality in ("m", "m7")
    intervals = [0, 3 if minor else 4, 7]
    if quality in ("7", "m7"):
        intervals.append(10)
    elif quality == "maj7":
        intervals.append(11)
    bass = NOTE_TO_PC[match.group(3)] if match.group(3) else root
    return ChordSpec(root, minor, tuple(intervals), bass)


def parse_chord(name: str) -> tuple[int, bool]:
    """Return the scored label (root pitch class, is_minor) for a chord name."""
    spec = chord_spec(name)
    return spec.root, spec.minor


def normalize_detected(name: str) -> tuple[int, bool] | None:
    """Reduce a detected label (may carry sus/7/9//bass) to (root, is_minor)."""
    if name == "N":
        return None
    base = name.split("/")[0]
    match = re.match(r"^([A-G][#b]?)(m(?!aj))?", base)
    if not match:
        return None
    return NOTE_TO_PC[match.group(1)], match.group(2) is not None


def chord_pitches(spec: ChordSpec, octave: int = 3) -> list[int]:
    """MIDI numbers for a closed root-position voicing."""
    base = 12 * (octave + 1) + spec.root
    return [base + interval for interval in spec.intervals]


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


def vocal_note(freq: float, dur: float, sr: int = SR, amp: float = 1.0) -> np.ndarray:
    """Sustained, vibrato-laden tone standing in for a sung note."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    vibrato = 1 + 0.018 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t * 3, 1.0)
    phase = 2 * np.pi * freq * np.cumsum(vibrato) / sr
    out = np.zeros(n)
    for k, gain in enumerate([1.0, 0.7, 0.45, 0.3, 0.2], start=1):
        out += gain * np.sin(k * phase)
    env = np.minimum(t * 12, 1.0) * np.minimum((dur - t) * 12, 1.0)
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
    # Sequence of (chord_name, beats); beats may be fractional (e.g. 3.5 for
    # a change pushed half a beat early).
    progression: list[tuple[str, float]]
    bpm: float
    repeats: int = 4
    melody_octave: int = 5
    detune_cents: float = 0.0
    seed: int = 42
    voicing: str = "block"  # "block" (restrike each beat) or "arpeggio"
    lead: str = "piano"  # "piano" (short melody notes) or "vocal" (sustained, vibrato)
    truth: list[tuple[float, float, str]] = field(default_factory=list)

    def render(self) -> np.ndarray:
        beat = 60.0 / self.bpm
        beat_samples = int(beat * SR)
        rng = np.random.default_rng(self.seed)
        detune = 2 ** (self.detune_cents / 1200)

        events: list[tuple[str, float]] = []
        for _ in range(self.repeats):
            events.extend(self.progression)

        total_beats = sum(b for _, b in events)
        grid_beats = math.ceil(total_beats)
        total_len = grid_beats * beat_samples + int(1.5 * SR)
        chords_track = np.zeros(total_len)
        bass_track = np.zeros(total_len)
        melody_track = np.zeros(total_len)

        def place(track: np.ndarray, s: int, tone: np.ndarray) -> None:
            end = min(s + len(tone), total_len)
            if end > s:
                track[s:end] += tone[: end - s]

        self.truth = []
        cursor_beats = 0.0
        for chord_name, beats in events:
            spec = chord_spec(chord_name)
            start = int(round(cursor_beats * beat_samples))
            t0 = cursor_beats * beat
            self.truth.append((t0, t0 + beats * beat, chord_name))
            n_strikes = math.ceil(beats)

            pitches = chord_pitches(spec, octave=3)
            if self.voicing == "arpeggio":
                # One chord tone per half beat, up and back down the voicing.
                pattern = pitches + [pitches[0] + 12] + pitches[-2:0:-1]
                for k in range(math.ceil(beats * 2)):
                    s = start + k * beat_samples // 2
                    midi = pattern[k % len(pattern)]
                    place(chords_track, s, piano_note(midi_to_hz(midi) * detune, beat * 0.9, amp=0.7))
            else:
                # Comping: restrike the chord every beat
                for b in range(n_strikes):
                    s = start + b * beat_samples
                    remaining = beats - b
                    dur = beat * 1.1 if remaining >= 1 else remaining * beat + 0.05
                    for midi in pitches:
                        place(chords_track, s, piano_note(midi_to_hz(midi) * detune, dur, amp=0.5))
            # Bass: root (or the inversion's bass note) on 1/3, fifth on 2/4
            for b in range(n_strikes):
                s = start + b * beat_samples
                if spec.bass != spec.root or b % 2 == 0:
                    pc = spec.bass
                else:
                    pc = (spec.root + 7) % 12
                dur = min(beat * 0.95, (beats - b) * beat)
                place(bass_track, s, bass_note(midi_to_hz(12 * 3 + pc) * detune, dur, amp=0.8))

            chord_tone_pcs = [(spec.root + i) % 12 for i in spec.intervals[:3]]
            if self.lead == "piano":
                # Melody: chord tones + occasional stepwise passing notes
                for b in range(math.ceil(beats * 2)):  # eighth-note-ish, half-beat grid
                    s = start + b * beat_samples // 2
                    if rng.random() < 0.25:
                        continue  # rest
                    if rng.random() < 0.25:
                        pc = (rng.choice(chord_tone_pcs) + rng.choice([-1, 1, 2])) % 12
                    else:
                        pc = rng.choice(chord_tone_pcs)
                    midi = 12 * (self.melody_octave + 1) + int(pc)
                    place(melody_track, s, piano_note(midi_to_hz(midi) * detune, beat * 0.6, amp=0.35))
            else:
                # "Vocal": loud sustained notes of 1-3 beats that often sit on
                # non-chord tones and ring over the next chord change.
                b = 0.0
                while b < beats:
                    length = float(rng.choice([1, 2, 3]))
                    if rng.random() < 0.2:
                        b += length / 2  # breath
                        continue
                    pc = int(rng.choice(chord_tone_pcs))
                    if rng.random() < 0.4:
                        pc = (pc + int(rng.choice([-2, -1, 1, 2]))) % 12
                    s = start + int(b * beat_samples)
                    midi = 12 * (self.melody_octave) + pc
                    place(melody_track, s, vocal_note(midi_to_hz(midi) * detune, length * beat, amp=0.55))
                    b += length

            cursor_beats += beats

        drums = drum_hits(grid_beats * beat_samples, beat_samples)
        mix = chords_track + bass_track + melody_track
        mix[: len(drums)] += 0.7 * drums
        mix += 0.004 * rng.standard_normal(total_len)
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = 0.9 * mix / peak
        return mix.astype(np.float32)


BASIC_SONGS = [
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

HARD_SONGS = [
    # ii-V-I-vi with 7th voicings: Cmaj7 contains Em, Am7 contains C
    Song("sevenths-c", [("Dm7", 4), ("G7", 4), ("Cmaj7", 4), ("Am7", 4)], bpm=96),
    # Inversions: the bass walks on the 3rd instead of the root
    Song("inversions-g", [("G", 4), ("D/F#", 4), ("Em", 4), ("C/E", 2), ("D", 2)], bpm=108),
    # Changes pushed half a beat early (off the beat grid)
    Song("anticipate-a", [("A", 3.5), ("E", 4), ("F#m", 4), ("D", 4.5)], bpm=116),
    # One chord per beat
    Song("per-beat-f", [("F", 1), ("C", 1), ("Dm", 1), ("Bb", 1)], bpm=96, repeats=10),
    # Arpeggiated comping instead of block chords
    Song("arpeggio-e", [("Em", 4), ("C", 4), ("G", 4), ("D", 4)], bpm=104, voicing="arpeggio"),
    # Loud sustained vibrato lead with non-chord tones held across changes
    Song("vocal-bb", [("Bb", 4), ("F", 4), ("Gm", 4), ("Eb", 4)], bpm=90, lead="vocal"),
    # Everything at once: two-beat changes, arpeggios and the vocal lead
    Song("vocal-arp-am", [("Am", 2), ("Dm", 2), ("G", 2), ("C", 2)], bpm=120,
         repeats=6, voicing="arpeggio", lead="vocal", seed=7),
]

SUITES = {"basic": BASIC_SONGS, "hard": HARD_SONGS}
SONGS = BASIC_SONGS + HARD_SONGS

TRANSITION_TOLERANCE = 0.25


def _collapse(segments: list[tuple[float, object]]) -> list[tuple[float, object]]:
    """Merge consecutive segments with the same label; keep each run's start."""
    out: list[tuple[float, object]] = []
    for start, label in segments:
        if not out or out[-1][1] != label:
            out.append((start, label))
    return out


def _edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _match_transitions(truth: list[float], detected: list[float], tol: float) -> list[float]:
    """Greedy one-to-one matching by distance; returns matched offsets (s)."""
    pairs = sorted(
        (abs(d - t), i, j) for i, t in enumerate(truth) for j, d in enumerate(detected) if abs(d - t) <= tol
    )
    used_t, used_d, offsets = set(), set(), []
    for dist, i, j in pairs:
        if i not in used_t and j not in used_d:
            used_t.add(i)
            used_d.add(j)
            offsets.append(dist)
    return offsets


def evaluate_song(song: Song, dump: bool = False, hop: float = 0.1, isolate: str = "none") -> dict:
    audio = song.render()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    sf.write(wav_path, audio, SR)
    kwargs = {} if isolate == "none" else {"isolate_instrumental": True, "separator": isolate}
    try:
        detected = detect_chords(wav_path, verbose=False, **kwargs)
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
    # Sequence and transition metrics on runs of identical labels.
    truth_runs = _collapse([(start, parse_chord(name)) for start, _, name in song.truth])
    detected_runs = _collapse(
        [(c.start_time, normalize_detected(c.name)) for c in detected if c.end_time > c.start_time]
    )
    truth_seq = [label for _, label in truth_runs]
    detected_seq = [label for _, label in detected_runs]
    seq_accuracy = max(0.0, 1.0 - _edit_distance(detected_seq, truth_seq) / len(truth_seq))

    truth_changes = [start for start, _ in truth_runs[1:]]
    detected_changes = [start for start, _ in detected_runs[1:]]
    offsets = _match_transitions(truth_changes, detected_changes, TRANSITION_TOLERANCE)
    precision = len(offsets) / len(detected_changes) if detected_changes else 0.0
    recall = len(offsets) / len(truth_changes) if truth_changes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "song": song.name,
        "accuracy": correct / total if total else 0.0,
        "root_accuracy": root_correct / total if total else 0.0,
        "no_chord_rate": no_chord / total if total else 0.0,
        "n_detected": len(detected),
        "seq_accuracy": seq_accuracy,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
        "change_offset_ms": 1000 * float(np.mean(offsets)) if offsets else float("nan"),
        "true_changes": len(truth_changes),
        "detected_changes": len(detected_changes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song", help="run only the named song")
    parser.add_argument("--dump", action="store_true", help="print detected-vs-truth timeline")
    parser.add_argument("--suite", choices=("all", *SUITES), default="all", help="song set to run")
    parser.add_argument(
        "--isolate",
        choices=("none", "roformer", "demucs"),
        default="none",
        help="run stem separation before detection (needs the separation extra)",
    )
    parser.add_argument("--json-output", type=Path, help="also write per-song results as JSON")
    args = parser.parse_args()

    suites = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    if args.song is not None:
        suites = {
            name: [s for s in songs if s.name == args.song] for name, songs in suites.items()
        }
        suites = {name: songs for name, songs in suites.items() if songs}
        if not suites:
            raise SystemExit(f"Unknown song {args.song!r}; choices: {[s.name for s in SONGS]}")

    print(f"{'song':<14}{'frame':>7}{'seq':>7}{'F1':>7}{'offset':>8}  changes(det/true)")
    all_results = {}
    for suite_name, songs in suites.items():
        results = []
        for song in songs:
            r = evaluate_song(song, dump=args.dump, isolate=args.isolate)
            results.append(r)
            print(
                f"{r['song']:<14}{r['accuracy']:>7.3f}{r['seq_accuracy']:>7.3f}"
                f"{r['change_f1']:>7.3f}{r['change_offset_ms']:>6.0f}ms"
                f"  {r['detected_changes']}/{r['true_changes']}",
                flush=True,
            )
        all_results[suite_name] = results
        means = {k: float(np.nanmean([r[k] for r in results])) for k in
                 ("accuracy", "seq_accuracy", "change_f1", "change_offset_ms")}
        print(
            f"{'MEAN ' + suite_name:<14}{means['accuracy']:>7.3f}{means['seq_accuracy']:>7.3f}"
            f"{means['change_f1']:>7.3f}{means['change_offset_ms']:>6.0f}ms",
            flush=True,
        )

    if args.json_output:
        args.json_output.write_text(json.dumps(all_results, indent=1))


if __name__ == "__main__":
    main()
