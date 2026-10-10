"""Optional vocal/drum removal via stem separation (Demucs or Mel-Band RoFormer).

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

The ``roformer`` separator instead runs Kimberley Jensen's Mel-Band RoFormer
vocal model (MIT), which removes vocals more cleanly than Demucs but does not
split out drums; the detector's HPSS stage already suppresses percussion. The
architecture is vendored in ``mel_band_roformer.py``. Its weights are published
only as a PyTorch pickle, which can execute code when loaded, so on first use
the pickle is downloaded from a pinned revision, checked against a pinned
SHA-256, loaded once in PyTorch's restricted ``weights_only`` mode, and
converted to ``safetensors`` (pure tensor data). The pickle is then deleted and
only the safetensors file is ever loaded again.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

DETECTOR_SR = 22050
SEPARATORS = ("demucs", "roformer")
DEFAULT_SEPARATOR = "roformer"
DEMUCS_MODEL = "htdemucs_ft"
# Stems summed to form the instrumental fed to the chroma stage. Vocals (melody)
# and drums (percussion) are dropped; bass carries the root, "other" the chords.
INSTRUMENTAL_STEMS = ("bass", "other")

_IMPORT_HINT = (
    "Demucs is required for stem separation. Install it with:\n"
    "    pip install 'chordify[separation]'   (or: pip install demucs)"
)

_ROFORMER_IMPORT_HINT = (
    "The RoFormer separator needs torch, einops and safetensors. Install them with:\n"
    "    pip install 'chordify[separation]'"
)

# Kimberley Jensen's Mel-Band RoFormer vocal model (MIT), pinned to an exact
# Hugging Face revision and file hash.
ROFORMER_SOURCE_URL = (
    "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/"
    "ac9b0614ab3cd7f77219e18ba494dfd93956c348/MelBandRoformer.ckpt"
)
ROFORMER_SOURCE_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
ROFORMER_WEIGHTS_FILE = "mel_band_roformer_vocals_kim.safetensors"
ROFORMER_SR = 44100
ROFORMER_CONFIG = dict(
    dim=384,
    depth=6,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_bands=60,
    dim_head=64,
    heads=8,
    sample_rate=ROFORMER_SR,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
)
# 8 s chunks (the training length) with 2x overlap, as in the reference inference.
ROFORMER_CHUNK_SIZE = 352800
ROFORMER_NUM_OVERLAP = 2
# Older torch.load(weights_only=True) could be bypassed (CVE-2025-32434).
MIN_TORCH_FOR_PICKLE = (2, 6)

_REPO_ROOT = Path(__file__).resolve().parent.parent


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


def get_model_cache_dir() -> Path:
    """Return the in-repo model cache directory, creating it if needed."""
    cache_dir = _REPO_ROOT / ".cache" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_safe_pickle_loader() -> None:
    import torch

    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if version < MIN_TORCH_FOR_PICKLE:
        raise RuntimeError(
            f"Converting the RoFormer checkpoint needs PyTorch >= "
            f"{'.'.join(map(str, MIN_TORCH_FOR_PICKLE))} (found {torch.__version__}); "
            "older weights_only loaders are bypassable (CVE-2025-32434)."
        )


def convert_checkpoint_to_safetensors(
    ckpt_path: Path, out_path: Path, expected_sha256: str
) -> None:
    """Convert a pickled state dict to safetensors, refusing anything but tensors.

    The file must match ``expected_sha256`` before it is unpickled, and it is
    unpickled only with ``weights_only=True``, which rejects any pickle that
    references code. The result is written atomically.
    """
    import torch
    from safetensors.torch import save_file

    actual = _sha256(ckpt_path)
    if actual != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {ckpt_path.name}: expected {expected_sha256}, "
            f"got {actual}. The file was not loaded; a corrupted or altered "
            "download is discarded, so re-running retries it."
        )
    _require_safe_pickle_loader()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
    ):
        raise ValueError(f"{ckpt_path.name} is not a plain tensor state dict")

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    # The rotary tables are one tensor shared by every layer; safetensors
    # stores each key separately, so give every entry its own copy.
    save_file({k: v.detach().clone().contiguous() for k, v in state.items()}, str(tmp_path))
    os.replace(tmp_path, out_path)


def _download(url: str, dest: Path) -> None:
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        for block in iter(lambda: response.read(1 << 20), b""):
            f.write(block)
    os.replace(tmp_path, dest)


def ensure_roformer_weights(verbose: bool = False) -> Path:
    """Return the cached RoFormer safetensors, downloading and converting once."""
    cache_dir = get_model_cache_dir()
    weights_path = cache_dir / ROFORMER_WEIGHTS_FILE
    if weights_path.exists():
        return weights_path

    ckpt_path = cache_dir / "MelBandRoformer.ckpt"
    if verbose:
        print("Downloading Mel-Band RoFormer vocal model (~913 MB, one time)...")
    _download(ROFORMER_SOURCE_URL, ckpt_path)
    try:
        if verbose:
            print("Verifying checksum and converting to safetensors...")
        convert_checkpoint_to_safetensors(ckpt_path, weights_path, ROFORMER_SOURCE_SHA256)
    finally:
        ckpt_path.unlink(missing_ok=True)
    return weights_path


@lru_cache(maxsize=1)
def _get_roformer_model():
    """Load the vendored Mel-Band RoFormer from safetensors (never a pickle)."""
    try:
        from safetensors.torch import load_file

        from .mel_band_roformer import MelBandRoformer
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(_ROFORMER_IMPORT_HINT) from exc

    model = MelBandRoformer(**ROFORMER_CONFIG)
    model.load_state_dict(load_file(str(ensure_roformer_weights(verbose=True))), strict=True)
    model.eval()
    return model


def _demix(model, mix, device: str, chunk_size: int, num_overlap: int):
    """Overlap-add chunked inference over a ``[channels, samples]`` tensor.

    Mirrors the reference inference: reflect-pad the borders, run fixed-size
    chunks, and cross-fade them with a linear fade window.
    """
    import contextlib

    import torch
    import torch.nn.functional as F

    step = chunk_size // num_overlap
    fade = chunk_size // 10
    border = chunk_size - step
    padded = mix.shape[1] > 2 * border and border > 0
    if padded:
        mix = F.pad(mix, (border, border), mode="reflect")

    window = torch.ones(chunk_size)
    window[:fade] *= torch.linspace(0, 1, fade)
    window[-fade:] *= torch.linspace(1, 0, fade)

    total = mix.shape[1]
    result = torch.zeros_like(mix)
    counter = torch.zeros_like(mix)
    amp = torch.autocast("cuda") if device == "cuda" else contextlib.nullcontext()
    i = 0
    with torch.no_grad(), amp:
        while i < total:
            part = mix[:, i : i + chunk_size]
            length = part.shape[-1]
            if length < chunk_size:
                if length > chunk_size // 2 + 1:
                    part = F.pad(part, (0, chunk_size - length), mode="reflect")
                else:
                    part = F.pad(part, (0, chunk_size - length))
            out = model(part.unsqueeze(0).to(device))[0].float().cpu()

            w = window.clone()
            if i == 0:
                w[:fade] = 1
            elif i + chunk_size >= total:
                w[-fade:] = 1
            result[..., i : i + length] += out[..., :length] * w[:length]
            counter[..., i : i + length] += w[:length]
            i += step

    estimate = torch.nan_to_num(result / counter, nan=0.0)
    if padded:
        estimate = estimate[..., border:-border]
    return estimate


def separate_vocals_roformer(
    audio_path: Path, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mix, vocals)`` as ``[2, samples]`` arrays at ``ROFORMER_SR``."""
    import librosa
    import soundfile as sf
    import torch

    model = _get_roformer_model()

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mix = data.T  # [channels, samples]
    if mix.shape[0] == 1:
        mix = np.repeat(mix, 2, axis=0)
    mix = mix[:2]
    if sr != ROFORMER_SR:
        mix = librosa.resample(mix, orig_sr=sr, target_sr=ROFORMER_SR).astype(np.float32)

    device = _pick_device()
    if verbose:
        print(f"Removing vocals with Mel-Band RoFormer on {device} (this can take minutes)...")
    mix_t = torch.from_numpy(np.ascontiguousarray(mix))
    try:
        model.to(device)
        vocals = _demix(model, mix_t, device, ROFORMER_CHUNK_SIZE, ROFORMER_NUM_OVERLAP)
    except (NotImplementedError, RuntimeError):
        if device != "mps":
            raise
        # Some STFT ops are missing on older MPS builds; CPU is always correct.
        model.to("cpu")
        vocals = _demix(model, mix_t, "cpu", ROFORMER_CHUNK_SIZE, ROFORMER_NUM_OVERLAP)
    return mix, vocals.numpy()


def _load_instrumental_roformer(audio_path: Path, verbose: bool = False) -> np.ndarray:
    import librosa

    mix, vocals = separate_vocals_roformer(audio_path, verbose=verbose)
    instrumental = (mix - vocals).mean(axis=0)
    return librosa.resample(instrumental, orig_sr=ROFORMER_SR, target_sr=DETECTOR_SR).astype(
        np.float32
    )


def load_instrumental(
    audio_path: Path,
    keep: tuple[str, ...] = INSTRUMENTAL_STEMS,
    verbose: bool = False,
    separator: str = DEFAULT_SEPARATOR,
) -> np.ndarray:
    """Return a mono ``DETECTOR_SR`` instrumental for chord detection.

    ``demucs`` mixes the kept stems (default: bass+other). ``roformer`` returns
    the mix minus the estimated vocals (``keep`` does not apply).
    """
    if separator not in SEPARATORS:
        raise ValueError(f"Unknown separator `{separator}`. Choose one of: {', '.join(SEPARATORS)}")
    if separator == "roformer":
        return _load_instrumental_roformer(audio_path, verbose=verbose)

    stems = separate_stems(audio_path, verbose=verbose)
    length = max(len(v) for v in stems.values())
    mix = np.zeros(length, dtype=np.float32)
    for name in keep:
        stem = stems[name]
        mix[: len(stem)] += stem
    return mix
