"""Chord notation helpers for display transposition and enharmonic spelling."""

from __future__ import annotations

import re

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_INDEX = {
    **{name: index for index, name in enumerate(NOTE_NAMES_SHARP)},
    **{name: index for index, name in enumerate(NOTE_NAMES_FLAT)},
    "B#": 0,
    "E#": 5,
    "Cb": 11,
    "Fb": 4,
}

FLAT_MAJOR_TONICS = {1, 3, 5, 6, 8, 10, 11}
CHORD_RE = re.compile(r"^([A-G](?:#|b)?)([^/]*)?(?:/([A-G](?:#|b)?))?$")


def note_name(pitch_class: int, prefer_flats: bool = False) -> str:
    """Return a note name for a pitch class."""
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    return names[pitch_class % 12]


def parse_note(note: str) -> int:
    """Parse a note name into a pitch class."""
    normalized = note.strip()
    if normalized not in NOTE_TO_INDEX:
        raise ValueError(f"Unsupported note name `{note}`")
    return NOTE_TO_INDEX[normalized]


def chord_root(chord_name: str) -> int | None:
    """Return the root pitch class for a chord symbol, if it can be parsed."""
    match = CHORD_RE.match(chord_name.strip())
    if not match:
        return None
    return parse_note(match.group(1))


def prefer_flats_for_key(tonic: int, mode: str = "major") -> bool:
    """Return whether a key is normally easier to read with flat spellings."""
    major_tonic = (tonic + 3) % 12 if mode == "minor" else tonic
    return major_tonic in FLAT_MAJOR_TONICS


def prefer_flats_for_note_name(note: str) -> bool:
    """Return a reasonable accidental convention for a display key name."""
    stripped = note.strip()
    if "b" in stripped:
        return True
    if "#" in stripped:
        return False
    mode = "minor" if stripped.endswith("m") else "major"
    return prefer_flats_for_key(parse_note(stripped.rstrip("m")), mode)


def transpose_chord_name(
    chord_name: str,
    semitones: int = 0,
    prefer_flats: bool = False,
) -> str:
    """Transpose and respell a chord symbol while preserving suffixes and slash bass."""
    if chord_name == "N":
        return chord_name

    match = CHORD_RE.match(chord_name.strip())
    if not match:
        return chord_name

    root, suffix, bass = match.groups()
    transposed_root = note_name(parse_note(root) + semitones, prefer_flats)
    transposed_bass = (
        f"/{note_name(parse_note(bass) + semitones, prefer_flats)}" if bass else ""
    )
    return f"{transposed_root}{suffix or ''}{transposed_bass}"


def respell_chord_name(chord_name: str, prefer_flats: bool = False) -> str:
    """Respell a chord symbol without changing its sounding pitch."""
    return transpose_chord_name(chord_name, 0, prefer_flats)
