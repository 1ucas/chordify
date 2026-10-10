# How It Works

1. **Download**: yt-dlp extracts the audio stream from the YouTube video and
   transcodes to WAV via FFmpeg. Files are cached at `.cache/audio/{video_id}.wav`
   so subsequent runs on the same video are instant.
2. **Detect Chords** (`chordify/detector.py`):
   - *(optional, `--isolate-instrumental`)* A Mel-Band RoFormer vocal model
     removes the vocal melody that pollutes the chroma; drums are left to the
     HPSS step below. Its weights are loaded from a safetensors file converted
     once from the hash-pinned checkpoint. With `--separator demucs`, Demucs
     `htdemucs_ft` instead splits the mix into drums/bass/other/vocals and only
     bass + other are recombined (bass carries the chord root the bass-focused
     chroma relies on). Requires the optional `separation` extra and is much
     slower. See `chordify/separation.py`.
   - HPSS suppresses percussion before chroma extraction.
   - Two chroma representations are blended — CENS (0.7, with a short
     11-frame smoothing window) for stability, CQT (0.3) for harmonic
     detail — both over C2–C7, plus a bass-focused CQT chroma in the
     C2–C5 range for root disambiguation. The C1 octave is excluded: its
     long CQT analysis windows smear bass energy across chord boundaries.
   - `librosa.beat.beat_track` produces beat boundaries; single beats are
     used as segments, absorbing extra beats only when a segment would be
     shorter than `MIN_SEGMENT_SECONDS` (0.25s).
   - Per-segment chroma is median-aggregated and smoothed with a triangular
     window of length 3.
   - A Krumhansl-Schmuckler key estimate seeds a soft key prior; a Viterbi
     decoder scores each major/minor triad template (equal-weight chord
     tones — root-weighted templates systematically follow the bass note,
     which spends half its time on the fifth in pop) against per-segment
     chroma with bass-root bonuses, non-chord-tone-bass penalties, and a
     chord-change penalty.
   - A second Viterbi pass adds a song-palette prior (chord types with
     negligible time share in pass 1 get a small penalty in pass 2),
     followed by a single-segment flicker suppressor.
   - Display decoration conservatively adds `sus2`/`sus4`/`9`/`7` extensions
     and slash bass when the residual chroma supports it.
3. **Transcribe**: Whisper `large-v3` (default), Parakeet `parakeet-tdt-0.6b-v3`,
   or Qwen3-ASR `1.7B` produces word-level timestamps. Qwen3-ASR is trained on
   singing voice and full songs with backing music; its word timings come from
   `Qwen3-ForcedAligner-0.6B`, whose stripped punctuation is mapped back from
   the transcript so phrase breaks survive. Whisper segment boundaries are
   preferred as line breaks; PT enclitic-pronoun hyphens (`tocar-te`) are
   re-joined.
4. **Align**: the formatter groups words into lines (with sub-word interpolation
   when a chord change lands inside a word) and places chord symbols above the
   lyrics. `--display-key` / `--transpose` / `--capo` respells the chart
   without changing detection.
