"""Kiểm tra CLI cố định thuật toán và lựa chọn đúng một WAV."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.cli import parse_args
from app.pipeline import resolve_input_file, load_audio_file


class EntryPointTests(unittest.TestCase):
    def test_student_main_cannot_switch_algorithm_or_add_context_variant(self):
        for algorithm in ("tt1", "tt2", "tt3"):
            args = parse_args(["--file", "phone_F2.wav", "--no-show"], algorithm)
            self.assertEqual(args.algorithm, algorithm)
            self.assertFalse(args.compare_context)
            for unsupported in (["--algorithm", "all"], ["--compare-context"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args(unsupported, algorithm)
                self.assertEqual(raised.exception.code, 2)

    def test_full_context_benchmark_rejects_partial_test_input(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(["--file", "phone_F2.wav", "--compare-context"])
        self.assertEqual(raised.exception.code, 2)

    def test_resolver_supports_test_train_stem_and_project_relative_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, test = root / "data" / "train", root / "data" / "test"
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            first, second = test / "phone_F2.wav", train / "studio_F1.wav"
            first.touch()
            second.touch()
            with patch.multiple("app.pipeline", PROJECT_ROOT=root, TRAIN_DIR=train, TEST_DIR=test), \
                    patch("app.pipeline.Path.cwd", return_value=root):
                self.assertEqual(resolve_input_file("phone_F2.wav"), first.resolve())
                self.assertEqual(resolve_input_file("phone_F2"), first.resolve())
                self.assertEqual(resolve_input_file("studio_F1.wav"), second.resolve())
                self.assertEqual(resolve_input_file("data/train/studio_F1.wav"), second.resolve())
                self.assertEqual(resolve_input_file(str(first)), first.resolve())
                with self.assertRaises(FileNotFoundError):
                    resolve_input_file("missing.wav")
                with self.assertRaises(ValueError):
                    resolve_input_file("phone_F2.lab")

    def test_single_file_reports_missing_ground_truth_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "recording.wav"
            wav.touch()
            with self.assertRaisesRegex(FileNotFoundError, "LAB cùng tên"):
                load_audio_file(wav)


if __name__ == "__main__":
    unittest.main()
