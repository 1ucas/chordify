# Transcription Benchmark

Compare Whisper and Parakeet on a local audio file:

```bash
./venv/bin/python scripts/benchmark_transcribers.py \
  .cache/audio/Nntd2fgMUYw.wav \
  --backend whisper \
  --backend parakeet \
  --language en
```

The benchmark reports elapsed seconds, real-time factor, word count, character
count, detected language, and errors. If you have reference lyrics, add
`--reference-text lyrics.txt` to include word error rate.
