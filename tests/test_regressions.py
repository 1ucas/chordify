import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from chordify.cli import clean_video_title, transform_chords_for_display
from chordify.detector import Chord
from chordify.formatter import get_chord_positions_for_line
from chordify.transcriber import (
    LyricLine,
    Word,
    group_words_into_lines,
    transcribe_audio,
)


class CliRegressionTests(unittest.TestCase):
    def test_clean_video_title_strips_lyrics_boilerplate(self) -> None:
        self.assertEqual(
            clean_video_title("Simple Plan - Welcome To My Life (Lyrics)"),
            "Simple Plan - Welcome To My Life",
        )

    def test_transform_chords_for_display_transposes_extensions_and_slash(self) -> None:
        chords = [
            Chord(start_time=0.0, end_time=2.0, name="C#"),
            Chord(start_time=2.0, end_time=4.0, name="A#m/F"),
        ]

        transformed = transform_chords_for_display(chords, transpose=1)

        self.assertEqual([chord.name for chord in transformed], ["D", "Bm/F#"])


class TranscriberRegressionTests(unittest.TestCase):
    def test_group_words_uses_segments_and_cleans_hyphen_spacing(self) -> None:
        words = [
            Word(0.0, 0.2, "Conhecer", segment_id=0),
            Word(0.2, 0.3, "-te", segment_id=0),
            Word(0.4, 0.6, "eu", segment_id=0),
            Word(0.7, 0.9, "quero", segment_id=1),
            Word(1.0, 1.2, "mais", segment_id=1),
        ]

        lines = group_words_into_lines(words)

        self.assertEqual([line.text for line in lines], ["Conhecer-te eu", "quero mais"])

    def test_parakeet_requests_word_timestamps(self) -> None:
        output = SimpleNamespace(
            text="hello world",
            timestamp={
                "word": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ]
            },
        )
        model = mock.Mock()
        model.transcribe.return_value = [output]
        asr = types.ModuleType("nemo.collections.asr")
        asr.models = SimpleNamespace(
            ASRModel=SimpleNamespace(from_pretrained=mock.Mock(return_value=model))
        )
        collections = types.ModuleType("nemo.collections")
        collections.asr = asr
        nemo = types.ModuleType("nemo")
        nemo.collections = collections
        fake_modules = {
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": asr,
        }

        with mock.patch.dict(sys.modules, fake_modules):
            result = transcribe_audio(Path("song.wav"), backend="parakeet")

        model.transcribe.assert_called_once_with(["song.wav"], timestamps=True)
        self.assertEqual([w.text for w in result.words], ["hello", "world"])


class FormatterRegressionTests(unittest.TestCase):
    def test_chord_change_inside_word_gets_interpolated_position(self) -> None:
        line = LyricLine(
            words=[
                Word(10.0, 11.0, "fighting"),
                Word(11.1, 11.4, "dawn"),
            ],
            start_time=10.0,
            end_time=11.4,
        )
        chords = [
            Chord(start_time=9.0, end_time=10.5, name="Am"),
            Chord(start_time=10.5, end_time=12.0, name="F"),
        ]

        positions = get_chord_positions_for_line(line, chords)

        self.assertEqual(positions[0].chord_name, "Am")
        self.assertEqual(positions[0].char_position, 0)
        self.assertEqual(positions[1].chord_name, "F")
        self.assertGreater(positions[1].char_position, 0)
        self.assertLess(positions[1].char_position, len("fighting"))


if __name__ == "__main__":
    unittest.main()
