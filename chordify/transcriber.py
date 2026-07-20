"""Lyrics transcription backends for lyrics alignment."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
SUPPORTED_TRANSCRIBERS = {"whisper", "parakeet"}
NON_LYRIC_WORDS = {"music", "[music]", "(music)"}
MAX_LYRIC_NO_SPEECH_PROB = 0.65
LINE_END_PUNCTUATION = {",", ".", "?", "!", ";", ":"}


@dataclass
class Word:
    """A transcribed word with timing information."""

    start_time: float
    end_time: float
    text: str
    segment_id: int | None = None


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    words: list[Word]
    full_text: str
    language: str


def is_lyric_word(text: str) -> bool:
    """Filter caption-style non-lyric markers from ASR output."""
    normalized = text.strip().lower()
    return bool(normalized) and normalized not in NON_LYRIC_WORDS


def cleanup_lyric_text(text: str) -> str:
    """Clean ASR spacing artifacts that are common in lyric output."""
    text = re.sub(r"\s+-(?=\w)", "-", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


def _transcribe_with_whisper(
    audio_path: Path,
    language: str | None = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    verbose: bool = False,
) -> TranscriptionResult:
    """Transcribe audio using OpenAI Whisper."""
    # Imported lazily so chord detection works without the whisper/torch stack.
    import whisper

    if verbose:
        print(f"Loading Whisper model: {model_name}")

    model = whisper.load_model(model_name)

    if verbose:
        print(f"Transcribing: {audio_path.name}")

    # Transcribe with word-level timestamps
    result = model.transcribe(
        str(audio_path),
        language=language,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.2,
        hallucination_silence_threshold=2.0,
        word_timestamps=True,
        verbose=verbose,
    )

    # Extract words with timestamps
    words: list[Word] = []

    seen_lyric_segment = False
    for segment_id, segment in enumerate(result.get("segments", [])):
        if (
            not seen_lyric_segment
            and segment.get("no_speech_prob", 0.0) > MAX_LYRIC_NO_SPEECH_PROB
        ):
            continue
        segment_words = segment.get("words", [])
        accepted_segment_word = False
        for word_info in segment_words:
            text = word_info["word"].strip()
            if not is_lyric_word(text):
                continue
            accepted_segment_word = True
            words.append(
                Word(
                    start_time=word_info["start"],
                    end_time=word_info["end"],
                    text=text,
                    segment_id=segment_id,
                )
            )
        seen_lyric_segment = seen_lyric_segment or accepted_segment_word

    detected_language = result.get("language", language or "en")

    return TranscriptionResult(
        words=words,
        full_text=result.get("text", "").strip(),
        language=detected_language,
    )


def _get_parakeet_text(output: Any) -> str:
    """Extract transcript text from a NeMo transcription object."""
    text = getattr(output, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(output, dict):
        for key in ("text", "pred_text"):
            if key in output and output[key] is not None:
                return str(output[key]).strip()
    return str(output).strip()


def _get_parakeet_timestamps(output: Any) -> dict[str, Any]:
    """Extract timestamp metadata from common NeMo transcription shapes."""
    timestamps = getattr(output, "timestamp", None)
    if timestamps is None:
        timestamps = getattr(output, "timestamps", None)
    if timestamps is None and isinstance(output, dict):
        timestamps = output.get("timestamp") or output.get("timestamps")
    return timestamps if isinstance(timestamps, dict) else {}


def _timestamp_value(timestamp: Any, *keys: str) -> float | None:
    """Read a numeric timestamp value from dict-like or object-like metadata."""
    for key in keys:
        value = timestamp.get(key) if isinstance(timestamp, dict) else getattr(timestamp, key, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _timestamp_text(timestamp: Any) -> str:
    """Read word text from dict-like or object-like timestamp metadata."""
    for key in ("word", "text", "token", "segment"):
        value = timestamp.get(key) if isinstance(timestamp, dict) else getattr(timestamp, key, None)
        if value:
            return str(value).strip()
    return ""


def _extract_parakeet_words(output: Any) -> list[Word]:
    """Convert NeMo word timestamp metadata to Chordify Word instances."""
    timestamps = _get_parakeet_timestamps(output)
    raw_words = timestamps.get("word") or timestamps.get("words") or []

    words: list[Word] = []
    for raw_word in raw_words:
        text = _timestamp_text(raw_word)
        if not is_lyric_word(text):
            continue

        start_time = _timestamp_value(raw_word, "start", "start_time")
        end_time = _timestamp_value(raw_word, "end", "end_time")
        if start_time is None or end_time is None:
            continue

        words.append(
            Word(
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    return words


def _transcribe_with_parakeet(
    audio_path: Path,
    language: str | None = None,
    model_name: str = DEFAULT_PARAKEET_MODEL,
    verbose: bool = False,
) -> TranscriptionResult:
    """Transcribe audio using NVIDIA NeMo Parakeet."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA Parakeet transcription requires NeMo. "
            "Install it with `pip install -e '.[parakeet]'`."
        ) from exc

    if language and verbose:
        print(
            "Parakeet handles language detection internally; "
            f"received language hint: {language}"
        )

    if verbose:
        print(f"Loading Parakeet model: {model_name}")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    if verbose:
        print(f"Transcribing: {audio_path.name}")

    outputs = model.transcribe([str(audio_path)])
    if not outputs:
        return TranscriptionResult(words=[], full_text="", language=language or "auto")

    output = outputs[0]
    words = _extract_parakeet_words(output)
    if not words:
        raise RuntimeError(
            f"Parakeet model `{model_name}` did not return word-level timestamps. "
            "Chordify needs word timings to align lyrics with chords; use a "
            "timestamp-capable model such as nvidia/parakeet-tdt-0.6b-v3."
        )

    detected_language = (
        getattr(output, "lang", None)
        or getattr(output, "language", None)
        or (output.get("lang") if isinstance(output, dict) else None)
        or (output.get("language") if isinstance(output, dict) else None)
        or language
        or "auto"
    )

    return TranscriptionResult(
        words=words,
        full_text=_get_parakeet_text(output),
        language=str(detected_language),
    )


def transcribe_audio(
    audio_path: Path,
    language: str | None = None,
    model_name: str | None = None,
    verbose: bool = False,
    backend: str = "whisper",
) -> TranscriptionResult:
    """
    Transcribe audio with word-level timestamps.

    Args:
        audio_path: Path to the audio file
        language: Language code (e.g., "en", "pt"). If None, auto-detect.
        model_name: Backend-specific model name. Defaults to Whisper large-v3
            or Parakeet TDT 0.6B v3.
        verbose: Whether to show progress information
        backend: Transcription backend: "whisper" or "parakeet".

    Returns:
        TranscriptionResult with word-level timestamps
    """
    normalized_backend = backend.strip().lower()
    if normalized_backend not in SUPPORTED_TRANSCRIBERS:
        supported = ", ".join(sorted(SUPPORTED_TRANSCRIBERS))
        raise ValueError(f"Unsupported transcriber `{backend}`. Choose one of: {supported}")

    if normalized_backend == "whisper":
        return _transcribe_with_whisper(
            audio_path=audio_path,
            language=language,
            model_name=model_name or DEFAULT_WHISPER_MODEL,
            verbose=verbose,
        )

    return _transcribe_with_parakeet(
        audio_path=audio_path,
        language=language,
        model_name=model_name or DEFAULT_PARAKEET_MODEL,
        verbose=verbose,
    )


@dataclass
class LyricLine:
    """A line of lyrics with its words and timing."""

    words: list[Word]
    start_time: float
    end_time: float

    @property
    def text(self) -> str:
        """Get the full text of the line."""
        return cleanup_lyric_text(" ".join(w.text for w in self.words))


def group_words_into_lines(
    words: list[Word],
    pause_threshold: float = 0.45,
    max_line_duration: float = 4.5,
    max_line_chars: int = 58,
    max_line_words: int = 12,
) -> list[LyricLine]:
    """
    Group words into lines based on pauses and timing.

    Args:
        words: List of transcribed words
        pause_threshold: Minimum pause (seconds) to start a new line
        max_line_duration: Maximum duration of a single line
        max_line_chars: Maximum formatted lyric characters in one line
        max_line_words: Maximum words in one line

    Returns:
        List of LyricLine objects
    """
    if not words:
        return []

    lines: list[LyricLine] = []
    current_line_words: list[Word] = [words[0]]

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        # Calculate gap between words
        gap = curr_word.start_time - prev_word.end_time

        # Calculate current line duration
        line_start = current_line_words[0].start_time
        line_duration = curr_word.end_time - line_start
        projected_words = [*current_line_words, curr_word]
        projected_chars = len(cleanup_lyric_text(" ".join(w.text for w in projected_words)))
        segment_changed = (
            prev_word.segment_id is not None
            and curr_word.segment_id is not None
            and prev_word.segment_id != curr_word.segment_id
        )
        punctuation_break = prev_word.text.rstrip()[-1:] in LINE_END_PUNCTUATION

        # Start new line if:
        # 1. There's a significant pause, or
        # 2. Whisper/Parakeet already found a phrase boundary, or
        # 3. The line would be too long
        should_break = (
            gap >= pause_threshold
            or segment_changed
            or (punctuation_break and gap >= 0.15)
            or line_duration > max_line_duration
            or projected_chars > max_line_chars
            or len(projected_words) > max_line_words
        )

        if should_break and current_line_words:
            lines.append(
                LyricLine(
                    words=current_line_words,
                    start_time=current_line_words[0].start_time,
                    end_time=current_line_words[-1].end_time,
                )
            )
            current_line_words = [curr_word]
        else:
            current_line_words.append(curr_word)

    # Don't forget the last line
    if current_line_words:
        lines.append(
            LyricLine(
                words=current_line_words,
                start_time=current_line_words[0].start_time,
                end_time=current_line_words[-1].end_time,
            )
        )

    return lines
