# Output Formats

## Default (Lyrics + Chords)

```
Never Gonna Give You Up - Rick Astley
=====================================

        D                      F#m
We're no strangers to love
               G                    D
You know the rules and so do I
        D                    F#m
A full commitment's what I'm thinking of
                G                   D
You wouldn't get this from any other guy
```

## Chords-only Mode

```
0:00    [Intro]
0:18    D
0:22    F#m
0:26    Bm
0:30    F#m
0:34    G
0:38    D
...
```

Audio downloads are cached at `<repo>/.cache/audio/{video_id}.wav` so the same
video is downloaded only once. Pass `--no-cache` to bypass the cache.
Formatted results are printed to stdout or written to `--output-file`; pass
`--repo-copy` to also write a generated copy under `<repo>/outputs/`.
