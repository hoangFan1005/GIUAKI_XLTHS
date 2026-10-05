"""Small mathematical checks independent of the supplied audio benchmark."""
import math
import unittest

from algorithms import tt1_hodgkinson as tt1
from algorithms import tt2_histogram as tt2
from algorithms import tt3_gaussian as tt3


class AlgorithmMathTests(unittest.TestCase):
    def test_histogram_plateaus_and_endpoints(self):
        self.assertEqual(tt2.local_maxima([3, 3, 1, 0, 2, 2]), [0, 4])
        self.assertEqual(tt2.local_maxima([0, 2, 2, 2, 0]), [2])
        self.assertEqual(tt2.local_maxima([0, 0, 0]), [])
        self.assertEqual(tt2.local_maxima([1, 1, 1]), [1])

    def test_histogram_first_two_low_to_high(self):
        values = [0.0] * 10 + [0.4] * 5 + [0.9] * 20
        threshold, details = tt2.threshold_from_histogram(values, 10, 0, 5.0, True)
        self.assertAlmostEqual(threshold, (5 * 0.05 + 0.45) / 6)
        self.assertEqual(details["fallback"], None)
        self.assertEqual(details["peak_indices"], [0, 4, 9])

    def test_source_padding_does_not_cascade(self):
        mask = [0] * 20
        mask[10] = 1
        padded = tt2.pad_speech(mask, 5)
        self.assertEqual(padded, [0] * 5 + [1] * 11 + [0] * 4)

    def test_histogram_weight_selection_uses_full_cleanup(self):
        values = [0.0] * 160
        values[40] = values[112] = 1.0
        record = {"features": {"energy": values, "centroid": list(values),
                               "starts": [i / 100 for i in range(160)],
                               "ends": [(i * 10 + 25) / 1000 for i in range(160)]},
                  "labels": [0] * 15 + [1] * 123 + [0] * 22, "duration": 1.62}
        # Padding thật tạo 102 speech frames; cleanup gap support 195 ms
        # nối thành 123 speech frames. Không mock predict để che sai framing.
        fitted = tt2.fit([record])
        fallback = tt2.fit([{"features": {"energy": values, "centroid": list(values)},
                             "labels": record["labels"]}])
        self.assertEqual(fitted["full_pipeline_records"], 1)
        self.assertEqual(fitted["predict_only_fallback_records"], 0)
        self.assertEqual(fitted["train_frame_f1"], 1.0)
        self.assertEqual(fallback["predict_only_fallback_records"], 1)
        self.assertAlmostEqual(fallback["train_frame_f1"], 204 / 225)

    def test_gaussian_density_crossing(self):
        params = tt3.equal_density_threshold(0.000359, 0.000715, 0.196747, 0.233472)
        threshold = params["threshold"]
        self.assertTrue(0.000359 < threshold < 0.196747)
        log_sil = -math.log(0.000715) - (threshold - 0.000359) ** 2 / (2 * 0.000715 ** 2)
        log_sp = -math.log(0.233472) - (threshold - 0.196747) ** 2 / (2 * 0.233472 ** 2)
        self.assertAlmostEqual(log_sil, log_sp, places=9)
        self.assertAlmostEqual(tt3.equal_density_threshold(0.1, 0.2, 0.5, 0.2)["threshold"], 0.3)

    def test_gaussian_degenerate_variance_is_finite(self):
        params = tt3.equal_density_threshold(0.0, 0.0, 0.1, 0.0)
        self.assertTrue(math.isfinite(params["threshold"]))
        self.assertTrue(params["sigma_was_floored"])
        self.assertAlmostEqual(params["threshold"], 0.05)

    def test_binary_area_monotonicity(self):
        silence, speech = [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]
        areas = [tt1.confusion_area(silence, speech, t) for t in [0.1, 0.2, 0.3, 0.4, 0.5]]
        self.assertTrue(all(a >= b for a, b in zip(areas, areas[1:])))

    def test_binary_source_count_stop_exact_iteration(self):
        silence, speech = [0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6, 0.8]
        result = tt1.binary_threshold(silence, speech)
        # f=g=[.2,.4,.6]; T=.4, then .3, then .35. Counts at .3 and
        # .35 repeat, so source steps return .35, not the exact area root .4.
        self.assertAlmostEqual(result["threshold"], 0.35)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["stop_reason"], "unchanged_counts_source_rule")
        self.assertNotEqual(result["area_residual"], 0.0)

    def test_binary_disjoint_midpoint(self):
        self.assertEqual(tt1.binary_threshold([0.0, 0.1], [0.3, 0.4])["threshold"], 0.2)


if __name__ == "__main__":
    unittest.main()
