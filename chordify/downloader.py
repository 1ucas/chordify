"""YouTube audio extraction using yt-dlp."""

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


@dataclass
class DownloadResult:
    """Result of a YouTube audio download."""

    audio_path: Path
    title: str
    duration: float
    from_cache: bool = False


YOUTUBE_ID_PATTERNS = [
    r"youtube\.com/watch\?v=([\w-]{11})",
    r"youtu\.be/([\w-]{11})",
    r"youtube\.com/shorts/([\w-]{11})",
]


def is_valid_youtube_url(url: str) -> bool:
    """Check if the given URL is a valid YouTube URL."""
    return any(re.search(p, url) for p in YOUTUBE_ID_PATTERNS)


def extract_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a URL."""
    for pattern in YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_cache_dir() -> Path:
    """Return the in-repo audio cache directory, creating it if needed."""
    cache_dir = _REPO_ROOT / ".cache" / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_audio(
    url: str,
    output_dir: Path | None = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> DownloadResult:
    """
    Download audio from a YouTube video, with an optional local cache keyed by video ID.

    When the cache is enabled (default) and contains an entry for this video,
    the cached file is returned immediately and no network request is made.

    Args:
        url: YouTube video URL
        output_dir: Directory to save a copy of the audio file. If None, uses a
            temp directory (or the cache dir when caching is enabled).
        verbose: Whether to show download progress
        use_cache: If True, read from / write to ``<repo>/.cache/audio``

    Returns:
        DownloadResult with path to the audio file and metadata.

    Raises:
        ValueError: If the URL is not a valid YouTube URL
        yt_dlp.utils.DownloadError: If the download fails
    """
    if not is_valid_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    video_id = extract_video_id(url)
    cache_dir = get_cache_dir() if use_cache else None
    cached_audio = cache_dir / f"{video_id}.wav" if cache_dir and video_id else None
    cached_meta = cache_dir / f"{video_id}.json" if cache_dir and video_id else None

    if cached_audio and cached_audio.exists() and cached_meta and cached_meta.exists():
        try:
            meta = json.loads(cached_meta.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
        if verbose:
            print(f"Cache hit: {cached_audio}")
        audio_path = cached_audio
        if output_dir is not None and output_dir.resolve() != cached_audio.parent.resolve():
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = output_dir / cached_audio.name
            shutil.copy2(cached_audio, audio_path)
        return DownloadResult(
            audio_path=audio_path,
            title=meta.get("title", "Unknown"),
            duration=float(meta.get("duration", 0.0)),
            from_cache=True,
        )

    download_dir = output_dir or Path(tempfile.mkdtemp(prefix="chordify_"))
    output_template = str(download_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown")
        duration = float(info.get("duration") or 0.0)
        downloaded_id = info.get("id") or video_id

        wav_files = list(download_dir.glob("*.wav"))
        if not wav_files:
            raise RuntimeError("Failed to find downloaded audio file")

        audio_path = wav_files[0]

    if cache_dir and downloaded_id:
        target_audio = cache_dir / f"{downloaded_id}.wav"
        target_meta = cache_dir / f"{downloaded_id}.json"
        try:
            if audio_path.resolve() != target_audio.resolve():
                shutil.copy2(audio_path, target_audio)
            target_meta.write_text(json.dumps({"title": title, "duration": duration}))
        except OSError as exc:
            if verbose:
                print(f"Warning: failed to populate cache: {exc}")

    return DownloadResult(
        audio_path=audio_path,
        title=title,
        duration=duration,
        from_cache=False,
    )
