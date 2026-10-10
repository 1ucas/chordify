"""Inference-only Mel-Band RoFormer, vendored for optional vocal removal.

Adapted from lucidrains/BS-RoFormer v0.3.10 (``bs_roformer/mel_band_roformer.py``),
the architecture the Kimberley Jensen vocal checkpoint was trained against.
Training/loss code, the ``beartype``/``rotary-embedding-torch`` dependencies,
and the ``Attend`` wrapper were removed; module and parameter names are kept
identical so the checkpoint's state dict loads with ``strict=True``. Only
``torch``, ``einops`` and ``librosa`` are needed.

Original code: https://github.com/lucidrains/BS-RoFormer

MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from librosa import filters
from torch import nn
from torch.nn import Module, ModuleList


class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


def _rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


class RotaryEmbedding(Module):
    """Interleaved rotary position embedding over the sequence dim (-2)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        # A (frozen) parameter, not a buffer, so the checkpoint key matches.
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate_queries_or_keys(self, t):
        seq = torch.arange(t.shape[-2], device=t.device, dtype=self.freqs.dtype)
        freqs = torch.einsum("i,j->ij", seq, self.freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        out = t.float() * freqs.cos() + _rotate_half(t.float()) * freqs.sin()
        return out.type(t.dtype)


class FeedForward(Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(dim_inner, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_embed: RotaryEmbedding):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(0.0))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
        out = F.scaled_dot_product_attention(q, k, v)
        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Transformer(Module):
    def __init__(self, *, dim, depth, dim_head, heads, rotary_embed):
        super().__init__()
        self.layers = ModuleList(
            [
                ModuleList(
                    [
                        Attention(dim, heads, dim_head, rotary_embed),
                        FeedForward(dim),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList(
            [nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs]
        )

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)
        return torch.stack([fn(part) for part, fn in zip(x, self.to_features)], dim=-2)


def _mlp(dim_in: int, dim_out: int, dim_hidden: int, depth: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    net: list[Module] = []
    for ind, (layer_in, layer_out) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(layer_in, layer_out))
        if ind < len(dims) - 2:
            net.append(nn.Tanh())
    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int, mlp_expansion_factor: int = 4):
        super().__init__()
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList(
            [
                nn.Sequential(
                    _mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                    nn.GLU(dim=-1),
                )
                for dim_in in dim_inputs
            ]
        )

    def forward(self, x):
        bands = x.unbind(dim=-2)
        return torch.cat([mlp(band) for band, mlp in zip(bands, self.to_freqs)], dim=-1)


class MelBandRoformer(Module):
    """Mel-Band RoFormer separator: (batch, channels, samples) -> stem audio."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        stereo: bool = False,
        num_stems: int = 1,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        sample_rate: int = 44100,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        mask_estimator_depth: int = 1,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        transformer_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head)
        self.layers = ModuleList(
            [
                ModuleList(
                    [
                        Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                        Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.stft_win_length = stft_win_length
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        freqs = stft_n_fft // 2 + 1

        mel_filter_bank = torch.from_numpy(
            filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)
        )
        # Upstream forces these so every frequency is covered by some band.
        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0
        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), "all frequencies need to be covered by a band"

        freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)[freqs_per_band]
        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = rearrange(freq_indices * 2 + torch.arange(2), "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)
        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")
        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        dims_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())
        self.band_split = BandSplit(dim=dim, dim_inputs=dims_with_complex)
        self.mask_estimators = ModuleList(
            [
                MaskEstimator(dim=dim, dim_inputs=dims_with_complex, depth=mask_estimator_depth)
                for _ in range(num_stems)
            ]
        )

    def forward(self, raw_audio):
        """b=batch f=freq t=time s=channel n=stems c=complex(2) d=feature."""
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        batch, channels, _ = raw_audio.shape
        assert channels == self.audio_channels, "channel count does not match the model's stereo setting"

        raw_audio, packed_shape = pack([raw_audio], "* t")
        stft_window = torch.hann_window(self.stft_win_length, device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack(stft_repr, packed_shape, "* f t c")[0]
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        batch_arange = torch.arange(batch, device=device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.band_split(x)

        # Axial attention: across time within each band, then across bands.
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = torch.view_as_complex(rearrange(stft_repr, "b f t c -> b 1 f t c"))
        masks = torch.view_as_complex(masks.contiguous()).type(stft_repr.dtype)

        # Bands overlap in frequency; average the masks where they do.
        scatter_indices = repeat(
            self.freq_indices, "f -> b n f t", b=batch, n=num_stems, t=stft_repr.shape[-1]
        )
        stems_repr = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(stems_repr).scatter_add_(2, scatter_indices, masks)
        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)
        stft_repr = stft_repr * (masks_summed / denom.clamp(min=1e-8))

        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        recon = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)
        recon = rearrange(recon, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=num_stems)
        if num_stems == 1:
            recon = rearrange(recon, "b 1 s t -> b s t")
        return recon
