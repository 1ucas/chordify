import hashlib
import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
except ImportError:  # pragma: no cover - separation extra not installed
    torch = None

from chordify.separation import _demix, convert_checkpoint_to_safetensors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _RunsCodeWhenUnpickled:
    def __reduce__(self):
        return (exec, ("import os; os.environ['CHORDIFY_TEST_PWNED'] = '1'",))


def _tiny_roformer():
    from chordify.mel_band_roformer import MelBandRoformer

    return MelBandRoformer(
        dim=16,
        depth=2,  # >1 so layers share rotary tensors, as in the real model
        stereo=True,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=8,
        dim_head=8,
        heads=2,
        sample_rate=8000,
        stft_n_fft=256,
        stft_hop_length=64,
        stft_win_length=256,
        mask_estimator_depth=1,
    ).eval()


@unittest.skipIf(torch is None, "requires the separation extra (torch, safetensors)")
class CheckpointConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_converted_weights_reproduce_the_model(self) -> None:
        torch.manual_seed(0)
        model = _tiny_roformer()
        ckpt = self.tmp / "model.ckpt"
        out = self.tmp / "model.safetensors"
        torch.save(model.state_dict(), ckpt)

        convert_checkpoint_to_safetensors(ckpt, out, _sha256(ckpt))

        reloaded = _tiny_roformer()
        reloaded.load_state_dict(load_file(str(out)), strict=True)
        audio = torch.randn(1, 2, 4000)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(audio), reloaded(audio)))

    def test_checksum_mismatch_is_refused_before_unpickling(self) -> None:
        ckpt = self.tmp / "model.ckpt"
        out = self.tmp / "model.safetensors"
        torch.save({"w": torch.ones(2)}, ckpt)

        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            convert_checkpoint_to_safetensors(ckpt, out, "0" * 64)
        self.assertFalse(out.exists())

    def test_pickle_that_runs_code_is_rejected(self) -> None:
        os.environ.pop("CHORDIFY_TEST_PWNED", None)
        ckpt = self.tmp / "evil.ckpt"
        out = self.tmp / "evil.safetensors"
        torch.save({"w": torch.ones(2), "payload": _RunsCodeWhenUnpickled()}, ckpt)

        with self.assertRaises(Exception):
            convert_checkpoint_to_safetensors(ckpt, out, _sha256(ckpt))
        self.assertNotIn("CHORDIFY_TEST_PWNED", os.environ)
        self.assertFalse(out.exists())


@unittest.skipIf(torch is None, "requires the separation extra (torch, safetensors)")
class DemixTests(unittest.TestCase):
    def test_overlap_add_reconstructs_identity_model(self) -> None:
        torch.manual_seed(0)
        for length in (30_000, 8_000, 3_000):
            mix = torch.randn(2, length)
            out = _demix(lambda x: x, mix, "cpu", chunk_size=8_000, num_overlap=2)
            self.assertEqual(out.shape, mix.shape)
            self.assertTrue(torch.allclose(out, mix, atol=1e-5), f"length={length}")


if __name__ == "__main__":
    unittest.main()
