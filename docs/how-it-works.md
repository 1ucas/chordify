# How It Works

1. **Download**: yt-dlp extracts the audio stream from the YouTube video and
   transcodes to WAV via FFmpeg. Files are cached at `.cache/audio/{video_id}.wav`
   so subsequent runs on the same video are instant.
2. **Detect Chords** (`chordify/detector.py`):
   - HPSS suppresses percussion before chroma extraction.
   - Two chroma representations are blended — CENS (0.6) for stability across
     beats, CQT (0.4) for harmonic detail — plus a bass-focused CQT chroma in
     the C2–C5 range for root disambiguation.
   - `librosa.beat.beat_track` produces beat boundaries, then groups of
     `BEATS_PER_CHORD_SEGMENT` beats are absorbed until each segment is at
     least `MIN_SEGMENT_SECONDS` (0.7s) long — compensates for librosa
     over-estimating tempo on slow songs.
   - Per-segment chroma is median-aggregated and smoothed with a triangular
     window of length 3.
   - A Krumhansl-Schmuckler key estimate seeds a soft key prior; a Viterbi
     decoder scores each major/minor triad template against per-segment
     chroma with bass-root bonuses, non-chord-energy penalties, and a
     chord-change penalty.
   - A second Viterbi pass adds a song-palette prior (chord types with
     negligible time share in pass 1 get a small penalty in pass 2),
     followed by a single-segment flicker suppressor.
   - Display decoration conservatively adds `sus2`/`sus4`/`9`/`7` extensions
     and slash bass when the residual chroma supports it.
3. **Transcribe**: Whisper `large-v3` (default) or Parakeet `parakeet-tdt-0.6b-v3`
   produces word-level timestamps. Whisper segment boundaries are preferred as
   line breaks; PT enclitic-pronoun hyphens (`tocar-te`) are re-joined.
4. **Align**: the formatter groups words into lines (with sub-word interpolation
   when a chord change lands inside a word) and places chord symbols above the
   lyrics. `--display-key` / `--transpose` / `--capo` respells the chart
   without changing detection.
