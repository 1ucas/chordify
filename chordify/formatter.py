"""Format and align chords with lyrics."""

from dataclasses import dataclass

from .detector import Chord
from .transcriber import LyricLine, Word


@dataclass
class ChordPosition:
    """A chord positioned at a specific character offset in a line."""

    chord_name: str
    char_position: int


def find_chord_at_time(chords: list[Chord], time: float) -> str | None:
    """Find which chord is active at a given time."""
    for chord in chords:
        if chord.start_time <= time < chord.end_time:
            return chord.name
    return None


def _word_offsets(line: LyricLine) -> list[int]:
    """Return the character offset for each word in a lyric line."""
    offsets: list[int] = []
    offset = 0
    for i, word in enumerate(line.words):
        offsets.append(offset)
        offset += len(word.text)
        if i < len(line.words) - 1:
            offset += 1
    return offsets


def _time_to_char_position(line: LyricLine, time: float) -> int:
    """Map a timestamp inside a lyric line to an approximate character offset."""
    if not line.words:
        return 0

    offsets = _word_offsets(line)
    if time <= line.words[0].start_time:
        return 0

    for i, word in enumerate(line.words):
        start_offset = offsets[i]
        end_offset = start_offset + len(word.text)

        if word.start_time <= time <= word.end_time:
            duration = max(word.end_time - word.start_time, 0.001)
            ratio = (time - word.start_time) / duration
            return start_offset + round(ratio * len(word.text))

        if i < len(line.words) - 1 and time < line.words[i + 1].start_time:
            return end_offset

    return len(line.text)


def get_chord_positions_for_line(
    line: LyricLine,
    chords: list[Chord],
) -> list[ChordPosition]:
    """
    Determine where chords should be positioned above a lyric line.

    Args:
        line: A line of lyrics with word timing
        chords: List of detected chords

    Returns:
        List of ChordPosition objects indicating where to place each chord
    """
    positions: list[ChordPosition] = []
    current_chord: str | None = None

    active_chord = find_chord_at_time(chords, line.start_time)
    if active_chord:
        positions.append(ChordPosition(chord_name=active_chord, char_position=0))
        current_chord = active_chord

    for chord in chords:
        if not (line.start_time < chord.start_time < line.end_time):
            continue
        if chord.name != current_chord:
            positions.append(
                ChordPosition(
                    chord_name=chord.name,
                    char_position=_time_to_char_position(line, chord.start_time),
                )
            )
            current_chord = chord.name

    deduped: list[ChordPosition] = []
    for position in sorted(positions, key=lambda item: item.char_position):
        if deduped and position.char_position == deduped[-1].char_position:
            deduped[-1] = position
            continue
        if deduped and position.chord_name == deduped[-1].chord_name:
            continue
        deduped.append(position)
    return deduped


def format_line_with_chords(
    line: LyricLine,
    chords: list[Chord],
) -> tuple[str, str]:
    """
    Format a single line with chords positioned above.

    Args:
        line: A line of lyrics
        chords: List of detected chords

    Returns:
        Tuple of (chord_line, lyric_line)
    """
    lyric_text = line.text
    positions = get_chord_positions_for_line(line, chords)

    if not positions:
        return "", lyric_text

    # Build the chord line
    chord_line = ""
    last_end = 0

    for pos in positions:
        # Add spaces to reach the chord position
        spaces_needed = pos.char_position - last_end
        if spaces_needed > 0:
            chord_line += " " * spaces_needed
        elif chord_line:
            chord_line += " "

        # Add the chord name
        chord_line += pos.chord_name
        last_end = max(pos.char_position + len(pos.chord_name), len(chord_line))

    return chord_line, lyric_text


def format_aligned_output(
    lines: list[LyricLine],
    chords: list[Chord],
    title: str | None = None,
) -> str:
    """
    Format complete output with chords aligned above lyrics.

    Args:
        lines: List of lyric lines with timing
        chords: List of detected chords
        title: Optional song title

    Returns:
        Formatted string with chords above lyrics
    """
    output_parts: list[str] = []

    # Add title if provided
    if title:
        output_parts.append(title)
        output_parts.append("=" * len(title))
        output_parts.append("")

    # Track sections (intro before first lyrics)
    if lines and chords:
        # There's an intro section
        intro_chords = [c for c in chords if c.start_time < lines[0].start_time]
        first_chord_start = min(c.start_time for c in chords)
        intro_duration = lines[0].start_time - first_chord_start
        if intro_chords and intro_duration > 4.0:
            output_parts.append("[Intro]")
            chord_names = "    ".join(c.name for c in intro_chords[:8])  # Limit intro chords
            output_parts.append(chord_names)
            output_parts.append("")

    # Format each line
    prev_end_time = 0.0
    for line in lines:
        # Add blank line for significant pauses (new section)
        if line.start_time - prev_end_time > 3.0:
            output_parts.append("")

        chord_line, lyric_line = format_line_with_chords(line, chords)

        if chord_line.strip():
            output_parts.append(chord_line)
        output_parts.append(lyric_line)

        prev_end_time = line.end_time

    return "\n".join(output_parts)


def format_simple_output(
    lines: list[LyricLine],
    chords: list[Chord],
    title: str | None = None,
) -> str:
    """
    Format output with chords and lyrics in a simpler format.

    This is an alternative format that's more compact.

    Args:
        lines: List of lyric lines with timing
        chords: List of detected chords
        title: Optional song title

    Returns:
        Formatted string
    """
    output_parts: list[str] = []

    if title:
        output_parts.append(title)
        output_parts.append("=" * len(title))
        output_parts.append("")

    for line in lines:
        # Get chords for this line
        line_chords = [
            c for c in chords
            if c.start_time >= line.start_time and c.start_time < line.end_time
        ]

        if line_chords:
            chord_str = "  ".join(c.name for c in line_chords)
            output_parts.append(f"[{chord_str}]")

        output_parts.append(line.text)
        output_parts.append("")

    return "\n".join(output_parts)
