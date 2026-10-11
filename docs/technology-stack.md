# Technology Stack

| Component | Library / Model | Version | License |
|-----------|-----------------|---------|---------|
| CLI Framework | [Typer](https://typer.tiangolo.com/) | `>=0.9` | MIT |
| Console UI | [Rich](https://github.com/Textualize/rich) | `>=13.0` | MIT |
| YouTube Download | [yt-dlp](https://github.com/yt-dlp/yt-dlp) (`default` extra, incl. `yt-dlp-ejs`) + [Deno](https://deno.com/) JS runtime | `>=2025.11.12` | Unlicense / MIT |
| Audio I/O + Chroma Features | [librosa](https://librosa.org/) | `>=0.10` (tested with 0.11.0) | ISC |
| Numerics | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) | `>=1.23` / `>=1.9` | BSD |
| Chord Detection | Custom — CQT+CENS chroma, HPSS pre-filter, beat-tracked bar aggregation, bass-aware Viterbi with key + song-palette priors (`chordify/detector.py`) | — | MIT |
| Speech Recognition (default) | [OpenAI Whisper](https://github.com/openai/whisper) — model `large-v3` | `>=20250625` | MIT |
| Speech Recognition (alt) | [NVIDIA NeMo Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — model `parakeet-tdt-0.6b-v3` | `nemo_toolkit[asr] >=2.4` | Apache 2.0 / CC BY 4.0 model weights |
| Stem Separation (optional, `--separator demucs`) | [Demucs](https://github.com/adefossez/demucs) — model `htdemucs_ft` | `>=4.1.0` | MIT |
| Vocal Removal (optional, default separator) | Mel-Band RoFormer — architecture vendored from [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) v0.3.10, weights by [Kimberley Jensen](https://huggingface.co/KimberleyJSN/melbandroformer) (converted to safetensors locally) | `safetensors`, `einops` | MIT / MIT |
| Audio post-processing | [FFmpeg](https://ffmpeg.org/) (via yt-dlp) | system package | LGPL/GPL |

All components are **open-source and free**. Processing happens entirely on your local machine.
