"""Các bất biến quan trọng của đặc trưng, miền thời gian và phép đánh giá."""

import cmath
import math
import tempfile
import unittest
from pathlib import Path

from core.features import compute_features, fft_radix2
from core.io_utils import read_lab
from core.metrics import boundary_metrics, endpoint_metrics, frame_labels, frame_metrics, ground_truth_boundaries
from core.postprocess import boundaries, fill_short_internal_silences, mask_segments, speech_envelope


class CoreTests(unittest.TestCase):
    def test_fft_matches_independent_direct_dft(self):
        values = [1 + 2j, 3 - 1j, -4 + 1j, 0.5 + 0j, 2j, -1j, 7, -3]
        expected = [sum(value * cmath.exp(-2j * math.pi * k * n / len(values))
                        for n, value in enumerate(values)) for k in range(len(values))]
        for actual, reference in zip(fft_radix2(values), expected):
            self.assertAlmostEqual(actual.real, reference.real, places=10)
            self.assertAlmostEqual(actual.imag, reference.imag, places=10)
        with self.assertRaises(ValueError):
            fft_radix2([1, 2, 3])

    def test_full_frames_only_do_not_pad_or_include_short_tail(self):
        features = compute_features([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 99.0], 1000, frame_ms=4, hop_ms=2)
        self.assertEqual(features["starts"], [0, .002])
        self.assertEqual(features["ends"], [.004, .006])
        self.assertEqual(features["centers"], [.002, .004])
        self.assertEqual(features["ste"], [10, 26])
        self.assertEqual(features["ma"], [1.5, 2.5])
        self.assertEqual(features["energy"], [2.5, 6.5])
        self.assertEqual(features["ste_norm"], [10 / 26, 1])
        with self.assertRaises(ValueError):
            compute_features([1.0], 1000)

    def test_silence_edges_preserved_and_exact_200ms_preserved(self):
        starts = [index / 10 for index in range(8)]
        mask = [0, 1, 0, 1, 0, 0, 1, 0]
        self.assertEqual(fill_short_internal_silences(mask, starts, .8), [0, 1, 1, 1, 0, 0, 1, 0])
        self.assertEqual(mask, [0, 1, 0, 1, 0, 0, 1, 0])
        self.assertEqual(fill_short_internal_silences([0] * 8, starts, .8), [0] * 8)

    def test_support_end_is_end_of_last_active_frame(self):
        features = compute_features([1] * 60, 1000, frame_ms=20, hop_ms=10)
        mask = [0, 1, 1, 0, 0]
        self.assertEqual(speech_envelope(mask, features["starts"], .06, frame_ends=features["ends"]), (.01, .04))
        self.assertEqual(boundaries(mask, features["starts"], .06, frame_ends=features["ends"]), [.01, .04])
        self.assertEqual(mask_segments(mask, features["starts"], .06, frame_ends=features["ends"]),
                         [(0, .01, 0), (.01, .04, 1), (.04, .06, 0)])

    def test_actual_support_gap_controls_internal_silence_fill(self):
        starts = [index / 100 for index in range(30)]
        ends = [min(start + .02, .3) for start in starts]
        mask = [0, 1] + [0] * 20 + [1] + [0] * 7
        # Hai khung speech bắt đầu cách 210 ms, nhưng gap support chỉ 190 ms.
        result = fill_short_internal_silences(mask, starts, .3, frame_ends=ends)
        self.assertEqual(result, [0] + [1] * 22 + [0] * 7)

    def test_large_endpoint_error_is_never_censored(self):
        result = endpoint_metrics((0, 4), (1, 2))
        self.assertEqual(result["mae_ms"], 1500)
        self.assertAlmostEqual(result["rmse_ms"], math.sqrt(2500000))
        self.assertEqual(result["start_error_ms"], -1000)
        self.assertIsNone(endpoint_metrics(None, (1, 2))["mae_ms"])

    def test_boundary_dp_maximizes_matches_then_minimizes_error(self):
        result = boundary_metrics([.10, .20], [.18, .26], .1)
        self.assertEqual(result["matched"], 2)
        self.assertAlmostEqual(result["mae_ms"], 70)
        self.assertEqual(boundary_metrics([1], [2], .1)["missed"], 1)
        self.assertEqual(boundary_metrics([1], [2], .1)["false_positive"], 1)
        self.assertEqual(boundary_metrics([1], [.98, 1.05], .1)["pairs"], [(1, .98)])

    def test_labels_half_open_ignore_unknown_and_merge_v_uv(self):
        intervals = [(0, .1, "sil"), (.1, .2, "v"), (.2, .3, "uv"), (.3, .4, "sil")]
        labels = frame_labels([.05, .1, .2, .3, .41], intervals)
        self.assertEqual(labels, [0, 1, 1, 0, None])
        self.assertEqual(ground_truth_boundaries(intervals), [.1, .3])
        metrics = frame_metrics(labels, [0, 1, 0, 0, 1])
        self.assertEqual(metrics["ignored"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], .75)
        with self.assertRaises(ValueError):
            frame_metrics([1], [])

    def test_lab_parser_rejects_malformed_or_overlapping_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.lab"
            path.write_text("0 .2 sil\n.2 .4 v\nF0mean 122\nF0std 18\n", encoding="utf-8")
            self.assertEqual(read_lab(path, .4), [(0, .2, "sil"), (.2, .4, "v")])
            for text in ("0 .2 sil\n.1 .4 v\n", "0 .2 speech\n", "0 .2 sil\nF0std nope\n", "0 .8 sil\n"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_lab(path, .4)


if __name__ == "__main__":
    unittest.main()
