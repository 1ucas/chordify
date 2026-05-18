# Chordify

![Chordify banner](chordify.png)

A CLI tool that extracts audio from YouTube videos, transcribes lyrics, and generates chord progressions aligned with the lyrics.

## Why

Mainstream chord-sheet sites cover hit songs well but tend to miss niche material — kids' songs, regional music, older deep cuts, indie releases. The project started from trying to play songs a toddler wanted to sing along with that simply weren't on any cifra site; the only source was YouTube. Chordify closes that gap by deriving the chart directly from the audio.

## Features

- **YouTube Audio Extraction** - Download audio from any YouTube video (with a local cache so each video is fetched only once)
- **Automatic Chord Detection** - CQT/CENS chroma matched against major/minor triad templates, with adaptive bar aggregation, bass-aware Viterbi decoding, key + song-palette priors, and conservative slash/extension display
- **Lyrics Transcription** - Speech-to-text using OpenAI Whisper or NVIDIA Parakeet (word-level timestamps)
- **Aligned Output** - Chord symbols positioned above corresponding lyrics, with sharp/flat respelling and optional `--display-key` / `--capo` transposition
- **Offline Processing** - All processing happens locally, no API keys required

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  YouTube URL    │───▶│  Audio Extractor │───▶│  Chord Detector      │
│  (user input)   │    │  yt-dlp + cache  │    │  librosa CQT chroma  │
└─────────────────┘    └──────────────────┘    │  + template matching │
                                               └──────────┬───────────┘
                                                          │
                                                          ▼
                       ┌──────────────────┐    ┌──────────────────────┐
                       │  Lyrics + Chords │◀───│  Lyrics Extractor    │
                       │  (aligned output)│    │  Whisper / Parakeet  │
                       │                  │    │  (word-level timing) │
                       └──────────────────┘    └──────────────────────┘
```

Audio downloads are cached at `<repo>/.cache/audio/{video_id}.wav` so the same
video is downloaded only once. Pass `--no-cache` to bypass the cache.
Formatted results are printed to stdout or written to `--output-file`; pass
`--repo-copy` to also write a generated copy under `<repo>/outputs/`.

## Accuracy

On an internal set of three reference songs, the current chord detector
averages **~88%** against verified chord charts (composite score 0.877).
Per-song breakdown:

| Song | Score |
|---|---|
| Save Tonight | 0.913 |
| Welcome to My Life | 0.902 |
| Senhor Te Quero | 0.815 |

The composite score combines palette coverage, palette overlap,
consecutive-chord transition overlap, and chord-change density, with a
12-key transposition search so recordings in a different key than the
reference chart still score fairly. The evaluation set is small and skewed
toward verse/chorus pop with mostly-triadic harmony — treat the number as
directional, not a published benchmark.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chordify.git
cd chordify

# Create a virtual environment (Python 3.10–3.12 per pyproject; 3.13 also works
# in practice with librosa 0.11+, but is outside the pinned range)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

To enable NVIDIA Parakeet transcription, install the optional NeMo dependencies:

```bash
pip install -e '.[parakeet]'
```

### System Dependencies

On macOS, you may need to install FFmpeg:
```bash
brew install ffmpeg
```

On Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Usage](docs/usage.md) | Common commands and option examples |
| [CLI Reference](docs/cli-reference.md) | Full flag reference table |
| [Output Formats](docs/output-formats.md) | Lyrics+chords and chords-only output |
| [Transcription Benchmark](docs/transcription-benchmark.md) | Comparing Whisper vs Parakeet |
| [How It Works](docs/how-it-works.md) | Pipeline internals |
| [Technology Stack](docs/technology-stack.md) | Libraries, models, and licenses |

## Notes

- First run will download the selected transcription model.
- Python 3.10 is the primary supported runtime (`pyproject.toml` pins
  `>=3.10,<3.13`). 3.13 has worked in practice with `librosa 0.11.0`, but
  isn't pinned — fall back to 3.10/3.11/3.12 if you hit numba/librosa
  import errors.
- Parakeet support requires the optional `parakeet` extra and a model that
  returns word-level timestamps; the default is `nvidia/parakeet-tdt-0.6b-v3`.
- Processing time depends on video length and your hardware.
- GPU acceleration is supported via Whisper's CUDA path if you have CUDA
  installed.
- Extension and slash-chord labels are intentionally conservative; unsupported
  or ambiguous audio falls back to the plain major/minor triad.
- The `outputs/` directory is git-ignored; pass `--repo-copy` if you want
  generated charts persisted there.

## Contributing

Contributions of any size are very welcome — bug reports, feature ideas, doc
fixes, typos, or full pull requests. If you hit a song where the detector
struggles, an issue with the YouTube link is especially helpful since it
gives a concrete failure case to work from.