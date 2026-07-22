"""CLI entry point for Chordify."""

import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .detector import Chord, detect_chords, format_chords_only
from .downloader import download_audio, extract_video_id, is_valid_youtube_url
from .formatter import format_aligned_output
from .notation import chord_root, parse_note, prefer_flats_for_note_name, transpose_chord_name
from .transcriber import (
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_WHISPER_MODEL,
    group_words_into_lines,
    transcribe_audio,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

app = typer.Typer(
    name="chordify",
    help="Extract chords and lyrics from YouTube videos.",
    add_completion=False,
)
console = Console()


TITLE_BOILERPLATE_PATTERNS = [
    re.compile(
        r"\s*[\[(]\s*(?:official\s+)?(?:music\s+)?"
        r"(?:lyric[s]?|lyric\s+video|video|audio|hd|4k)\s*[\])]\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\s+\|\s+.*$"),
]


def clean_video_title(title: str) -> str:
    """Strip common YouTube boilerplate from chart titles."""
    cleaned = title.strip()
    for pattern in TITLE_BOILERPLATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned or title.strip() or "Unknown"


def build_output_filename(title: str, url: str, chords_only: bool) -> str:
    """Build a stable filename for the repo-local output copy."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-").lower()
    video_id = extract_video_id(url)
    parts = [part for part in (slug, video_id) if part]
    suffix = "_chords" if chords_only else ""
    return f"{'_'.join(parts) or 'chordify_output'}{suffix}.txt"


def save_repo_output(
    output: str,
    title: str,
    url: str,
    chords_only: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save a formatted output copy in the requested output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / build_output_filename(title, url, chords_only)
    output_path.write_text(output)
    return output_path


def estimate_key_tonic_from_chords(chords: list[Chord]) -> int | None:
    """Estimate a display-key tonic from decoded chord durations."""
    if not chords:
        return None

    histogram = [0.0] * 12
    for chord in chords:
        root = chord_root(chord.name)
        if root is not None:
            histogram[root] += max(chord.duration, 0.0)

    if not any(histogram):
        return None

    best_score = float("-inf")
    best_tonic: int | None = None
    for tonic in range(12):
        major_score = (
            histogram[tonic]
            + histogram[(tonic + 5) % 12]
            + histogram[(tonic + 7) % 12]
            + (0.9 * histogram[(tonic + 9) % 12])
        )
        minor_score = (
            histogram[tonic]
            + histogram[(tonic + 5) % 12]
            + (0.8 * histogram[(tonic + 7) % 12])
            + histogram[(tonic + 8) % 12]
            + (0.8 * histogram[(tonic + 10) % 12])
        )
        score = max(major_score, minor_score)
        if score > best_score:
            best_score = score
            best_tonic = tonic

    return best_tonic


def transform_chords_for_display(
    chords: list[Chord],
    transpose: int = 0,
    prefer_flats: bool = False,
) -> list[Chord]:
    """Return a chord sequence transposed/respelled for display."""
    return [
        Chord(
            start_time=chord.start_time,
            end_time=chord.end_time,
            name=transpose_chord_name(chord.name, transpose, prefer_flats),
        )
        for chord in chords
    ]


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"chordify version {__version__}")
        raise typer.Exit()


@app.command()
def main(
    url: Annotated[
        str,
        typer.Argument(help="YouTube video URL"),
    ],
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output-file",
            "-o",
            help="Save output to a file",
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Language hint for transcription (e.g., 'en', 'pt')",
        ),
    ] = None,
    chords_only: Annotated[
        bool,
        typer.Option(
            "--chords-only",
            help="Show only chords with timestamps (no lyrics)",
        ),
    ] = False,
    keep_audio: Annotated[
        bool,
        typer.Option(
            "--keep-audio",
            "-k",
            help="Keep the downloaded audio file",
        ),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Skip the local audio cache (force re-download)",
        ),
    ] = False,
    isolate_instrumental: Annotated[
        bool,
        typer.Option(
            "--isolate-instrumental",
            help=(
                "Strip vocals and drums with Demucs before chord detection "
                "(keeps bass+other). Requires the 'separation' extra and is "
                "much slower, but helps on vocal-heavy recordings"
            ),
        ),
    ] = False,
    transcriber: Annotated[
        str,
        typer.Option(
            "--transcriber",
            help="Lyrics transcription backend: whisper or parakeet",
        ),
    ] = "whisper",
    transcription_model: Annotated[
        Optional[str],
        typer.Option(
            "--transcription-model",
            help=(
                "Override the backend model "
                f"(defaults: {DEFAULT_WHISPER_MODEL} or {DEFAULT_PARAKEET_MODEL})"
            ),
        ),
    ] = None,
    display_key: Annotated[
        Optional[str],
        typer.Option(
            "--display-key",
            help="Transpose the displayed chart to this key (for example: D, Bb, Am)",
        ),
    ] = None,
    transpose: Annotated[
        int,
        typer.Option(
            "--transpose",
            help="Transpose displayed chords by this many semitones",
        ),
    ] = 0,
    capo: Annotated[
        Optional[int],
        typer.Option(
            "--capo",
            help="Annotate the output with a capo number",
        ),
    ] = None,
    repo_copy: Annotated[
        bool,
        typer.Option(
            "--repo-copy/--no-repo-copy",
            help="Also save a generated copy under the repo output directory",
        ),
    ] = False,
    repo_output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Directory for the optional generated repo copy",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed progress",
        ),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """
    Extract chords and lyrics from a YouTube video.

    Example:
        chordify "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    """
    # Validate URL
    if not is_valid_youtube_url(url):
        console.print("[red]Error:[/red] Invalid YouTube URL")
        raise typer.Exit(1)

    temp_dir = None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=verbose,  # Disable progress bar in verbose mode
        ) as progress:
            # Step 1: Download audio
            task = progress.add_task("Downloading audio...", total=None)

            output_dir = Path.cwd() if keep_audio else None

            download_result = download_audio(
                url=url,
                output_dir=output_dir,
                verbose=verbose,
                use_cache=not no_cache,
            )

            # Track temp dir for cleanup (only set if downloader created one for us)
            if (
                not keep_audio
                and not download_result.from_cache
                and download_result.audio_path.parent.name.startswith("chordify_")
            ):
                temp_dir = download_result.audio_path.parent
            progress.update(task, completed=True)

            if verbose:
                source = "Cache" if download_result.from_cache else "Downloaded"
                console.print(f"[green]{source}:[/green] {download_result.title}")
                console.print(f"[dim]Audio file: {download_result.audio_path}[/dim]")

            display_title = clean_video_title(download_result.title)

            # Step 2: Detect chords
            progress.update(task, description="Detecting chords...")
            chords = detect_chords(
                audio_path=download_result.audio_path,
                verbose=verbose,
                isolate_instrumental=isolate_instrumental,
            )
            progress.update(task, completed=True)

            display_transpose = transpose
            prefer_flats = False
            if display_key:
                target_key = display_key.strip()
                target_note = target_key[:-1] if target_key.endswith("m") else target_key
                target_tonic = parse_note(target_note)
                source_tonic = estimate_key_tonic_from_chords(chords)
                if source_tonic is not None:
                    key_delta = target_tonic - source_tonic
                    if key_delta > 6:
                        key_delta -= 12
                    elif key_delta < -6:
                        key_delta += 12
                    display_transpose += key_delta
                prefer_flats = prefer_flats_for_note_name(target_key)

            if display_transpose or display_key:
                chords = transform_chords_for_display(
                    chords=chords,
                    transpose=display_transpose,
                    prefer_flats=prefer_flats,
                )

            if verbose:
                console.print(f"[green]Detected {len(chords)} chord changes[/green]")

            # Step 3: Transcribe lyrics (unless chords-only mode)
            if chords_only:
                output = format_chords_only(chords)
            else:
                progress.update(task, description="Transcribing lyrics...")
                transcription = transcribe_audio(
                    audio_path=download_result.audio_path,
                    language=language,
                    model_name=transcription_model,
                    verbose=verbose,
                    backend=transcriber,
                )
                progress.update(task, completed=True)

                if verbose:
                    console.print(
                        f"[green]Transcribed {len(transcription.words)} words[/green]"
                    )
                    console.print(f"[dim]Detected language: {transcription.language}[/dim]")

                # Step 4: Group words into lines
                lines = group_words_into_lines(transcription.words)

                # Step 5: Format output
                chart_title = display_title
                if capo is not None:
                    chart_title = f"{chart_title} (capo {capo})"
                output = format_aligned_output(
                    lines=lines,
                    chords=chords,
                    title=chart_title,
                )

        # Output results
        repo_output_file: Path | None = None
        if repo_copy or repo_output_dir is not None:
            repo_output_file = save_repo_output(
                output=output,
                title=display_title,
                url=url,
                chords_only=chords_only,
                output_dir=repo_output_dir or DEFAULT_OUTPUT_DIR,
            )

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output)
            console.print(f"[green]Saved to:[/green] {output_file}")
        else:
            console.print()
            console.print(output)

        if repo_output_file and (
            not output_file or output_file.resolve() != repo_output_file.resolve()
        ):
            console.print(f"[green]Saved repo copy to:[/green] {repo_output_file}")

        if keep_audio:
            console.print(f"\n[dim]Audio saved to: {download_result.audio_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    finally:
        # Clean up temp directory
        if temp_dir and temp_dir.exists() and not keep_audio:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
