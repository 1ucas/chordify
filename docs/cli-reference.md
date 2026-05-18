# CLI Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--output-file` | `-o` | Save output to a file |
| `--repo-copy` / `--no-repo-copy` | | Also write a generated copy under the repo output directory |
| `--output-dir` | | Directory for the optional generated repo copy |
| `--language` | `-l` | Language hint for transcription (en/pt) |
| `--chords-only` | | Skip lyrics, show timestamped chords only |
| `--keep-audio` | `-k` | Copy the audio file into the current directory |
| `--no-cache` | | Bypass the local audio cache (force re-download) |
| `--transcriber` | | Lyrics transcription backend (`whisper` or `parakeet`) |
| `--transcription-model` | | Override the backend model name |
| `--display-key` | | Transpose the displayed chart to a target key |
| `--transpose` | | Transpose displayed chords by semitones |
| `--capo` | | Add a capo annotation to the chart title |
| `--verbose` | `-v` | Show detailed progress |
| `--help` | | Show help message |
