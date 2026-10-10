# Usage

## Basic Usage

```bash
# Extract chords and lyrics from a YouTube video
chordify "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Options

```bash
# Specify language for better transcription accuracy
chordify "https://www.youtube.com/watch?v=..." -l pt  # Portuguese
chordify "https://www.youtube.com/watch?v=..." -l en  # English (default)

# Save output to a file
chordify "https://www.youtube.com/watch?v=..." -o song_chords.txt

# Also save a generated copy under the repo outputs folder
chordify "https://www.youtube.com/watch?v=..." --repo-copy

# Choose a different folder for generated copies
chordify "https://www.youtube.com/watch?v=..." --output-dir /tmp/chordify-runs

# Transpose the displayed chart
chordify "https://www.youtube.com/watch?v=..." --display-key D --capo 1
chordify "https://www.youtube.com/watch?v=..." --transpose 1

# Show only chords with timestamps (no lyrics)
chordify "https://www.youtube.com/watch?v=..." --chords-only

# Use NVIDIA Parakeet for lyric transcription
chordify "https://www.youtube.com/watch?v=..." --transcriber parakeet

# Use Qwen3-ASR (trained on singing / songs with backing music)
chordify "https://www.youtube.com/watch?v=..." --transcriber qwen3 -l pt

# Override the transcription model
chordify "https://www.youtube.com/watch?v=..." \
  --transcriber parakeet \
  --transcription-model nvidia/parakeet-tdt-0.6b-v3

# Remove vocals/drums before chord detection (needs the 'separation' extra)
chordify "https://www.youtube.com/watch?v=..." --isolate-instrumental
chordify "https://www.youtube.com/watch?v=..." --separator roformer

# Keep the downloaded audio file in the current directory
chordify "https://www.youtube.com/watch?v=..." -k

# Bypass the audio cache and force a fresh download
chordify "https://www.youtube.com/watch?v=..." --no-cache

# Verbose output (show progress details)
chordify "https://www.youtube.com/watch?v=..." -v
```
