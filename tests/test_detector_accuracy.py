"""End-to-end chord detection regression test on synthesized audio.

Renders a short C-G-Am-F progression (piano comping, root/fifth bass,
melody, drums) and asserts the detector recovers the exact chord sequence.
This guards the detector tuning against regressions; the full benchmark
lives in scripts/chord_eval.py.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chord_eval import SR, Song  # noqa: E402

from chordify.detector import detect_chords  # noqa: E402


class DetectorAccuracyTests(unittest.TestCase):
    def test_pop_progression_is_recovered_exactly(self) -> None:
        song = Song("mini", [("C", 4), ("G", 4), ("Am", 4), ("F", 4)], bpm=110, repeats=2)
        audio = song.render()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            sf.write(wav_path, audio, SR)
            chords = detect_chords(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        self.assertEqual(
            [chord.name for chord in chords],
            ["C", "G", "Am", "F", "C", "G", "Am", "F"],
        )

        # Boundaries should land within a third of a bar of the truth.
        expected_starts = [i * (60.0 / 110) * 4 for i in range(8)]
        for chord, expected in zip(chords, expected_starts):
            self.assertLess(abs(chord.start_time - expected), 0.75)


if __name__ == "__main__":
    unittest.main()
