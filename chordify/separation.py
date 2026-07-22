"""Optional vocal/drum removal via Demucs stem separation.

Chroma-based chord detection is confused by two things a mix contains that
aren't the harmony: the sung melody (vocal notes land on non-chord tones) and
drum transients (broadband energy that smears the chroma). This module runs
Demucs' fine-tuned Hybrid Transformer model (``htdemucs_ft``) to split the
audio into ``drums / bass / other / vocals`` and keeps only the chord-bearing
accompaniment.

Which stems are kept is deliberate:

* ``other`` — guitars/keys/synths, i.e. the chordal instruments.
* ``bass``  — the bass note is the chord *root* most of the time, and the
  detector explicitly scores against a bass-focused chroma (see
  ``BASS_ROOT_BOOST`` and slash-chord decoration in ``detector.py``). Dropping
  it would strip the strongest root cue, so it is kept by default.

``vocals`` and ``drums`` are discarded.

Demucs is an **optional** dependency (``pip install 'chordify[separation]'``).
It is imported lazily so the core package works without it, and the model
(~320 MB, safetensors) is downloaded from the Demucs author's Hugging Face repo
on first use and cached by ``huggingface_hub``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DETECTOR_SR = 22050
DEMUCS_MODEL = "htdemucs_ft"
# Stems summed to form the instrumental fed to the chroma stage. Vocals (melody)
# and drums (percussion) are dropped; bass carries the root, "other" the chords.
INSTRUMENTAL_STEMS = ("bass", "other")

_IMPORT_HINT = (
    "Demucs is required for stem separation. Install it with:\n"
    "    pip install 'chordify[separation]'   (or: pip install demucs)"
)


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the htdemucs_ft model (downloads weights on first call)."""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:  # pragma: no cover - exercised only without demucs
        raise ImportError(_IMPORT_HINT) from exc
    model = get_model(DEMUCS_MODEL)
    model.eval()
    return model


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate_stems(audio_path: Path, verbose: bool = False) -> dict[str, np.ndarray]:
    """Separate ``audio_path`` into stems, each a mono waveform at ``DETECTOR_SR``.

    Returns a dict keyed by stem name (``drums``/``bass``/``other``/``vocals``).
    """
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = _get_model()

    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    wav = torch.from_numpy(data.T)  # [channels, samples]
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    # Demucs expects the input standardized; undo it on the output.
    ref = wav.mean(0)
    wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)

    device = _pick_device()
    if verbose:
        print(f"Separating stems with {DEMUCS_MODEL} on {device} (this can take minutes)...")

    with torch.no_grad():
        out = apply_model(
            model, wav_n[None], device=device, split=True, overlap=0.25, progress=False
        )[0]
    out = out * ref.std() + ref.mean()

    stems: dict[str, np.ndarray] = {}
    for i, name in enumerate(model.sources):
        mono = out[i].mean(0).cpu().numpy()  # downmix to mono
        stems[name] = librosa.resample(
            mono, orig_sr=model.samplerate, target_sr=DETECTOR_SR
        ).astype(np.float32)
    return stems


def load_instrumental(
    audio_path: Path,
    keep: tuple[str, ...] = INSTRUMENTAL_STEMS,
    verbose: bool = False,
) -> np.ndarray:
    """Return a mono ``DETECTOR_SR`` mix of the kept stems (default: bass+other)."""
    stems = separate_stems(audio_path, verbose=verbose)
    length = max(len(v) for v in stems.values())
    mix = np.zeros(length, dtype=np.float32)
    for name in keep:
        stem = stems[name]
        mix[: len(stem)] += stem
    return mix
