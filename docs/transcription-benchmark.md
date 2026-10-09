# Transcription Benchmark

Compare Whisper, Parakeet, and Qwen3-ASR on a local audio file:

```bash
./venv/bin/python scripts/benchmark_transcribers.py \
  .cache/audio/Nntd2fgMUYw.wav \
  --backend whisper \
  --backend parakeet \
  --backend qwen3 \
  --language en
```

The benchmark reports elapsed seconds, real-time factor, word count, character
count, detected language, and errors. If you have reference lyrics, add
`--reference-text lyrics.txt` to include word error rate.
