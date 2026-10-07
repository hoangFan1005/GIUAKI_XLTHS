"""Regressions for Gaussian crossings near the normalized STE sigma floor."""
import math
import unittest

from algorithms.tt3_gaussian import equal_density_threshold, population_statistics


def log_density_gap(threshold, mu_sil, sigma_sil, mu_sp, sigma_sp):
    """Input: threshold and effective Gaussian parameters. Output: log pSil/pSp."""
    log_sil = -math.log(sigma_sil) - 0.5 * ((threshold - mu_sil) / sigma_sil) ** 2
    log_sp = -math.log(sigma_sp) - 0.5 * ((threshold - mu_sp) / sigma_sp) ** 2
    return log_sil - log_sp


class GaussianNumericsTests(unittest.TestCase):
    def test_floor_scale_crossing_matches_decimal_oracle(self):
        # Decimal 60-digit oracle, rounded to binary64; catches lost c/discriminant.
        params = equal_density_threshold(0.5, 1e-9, 0.50000001, 2e-9)
        self.assertEqual(params["threshold_rule"], "equal_density_between_means")
        self.assertAlmostEqual(params["threshold"], 0.5000000034705506,
                               delta=math.ulp(0.5))
        self.assertLess(abs(log_density_gap(params["threshold"], 0.5, 1e-9,
                                            0.50000001, 2e-9)), 5e-7)

    def test_identical_means_unequal_variances_use_explicit_fallback(self):
        params = equal_density_threshold(0.5, 1e-9, 0.5, 2e-9)
        self.assertEqual(params["threshold"], 0.5)
        self.assertEqual(params["threshold_rule"], "no_between_means_crossing_midpoint")
        self.assertAlmostEqual(log_density_gap(0.5, 0.5, 1e-9, 0.5, 2e-9),
                               math.log(2.0), places=13)
        expected_roots = [0.49999999864044403, 0.500000001359556]
        self.assertEqual(len(params["crossings"]), 2)
        for actual, expected in zip(sorted(params["crossings"]), expected_roots):
            self.assertAlmostEqual(actual, expected, delta=math.ulp(0.5))
            self.assertLess(abs(log_density_gap(actual, 0.5, 1e-9, 0.5, 2e-9)), 1e-7)

    def test_saved_train_parameters_keep_density_equality_and_threshold(self):
        # Frozen TRAIN fixture: independent of any subsequent artifact regeneration.
        inputs = (0.00038711079659476316, 0.0007092854827602364,
                  0.20264889816756282, 0.23562613277402947)
        params = equal_density_threshold(*inputs)
        self.assertEqual(params["threshold_rule"], "equal_density_between_means")
        self.assertAlmostEqual(params["threshold"], 0.0028777336852328582,
                               delta=8 * math.ulp(0.0028777336852328582))
        self.assertLess(abs(log_density_gap(params["threshold"], *inputs)), 1e-12)

    def test_swapping_classes_preserves_crossing_and_fallback(self):
        for inputs in [(0.5, 1e-9, 0.50000001, 2e-9),
                       (0.00038711079659476316, 0.0007092854827602364,
                        0.20264889816756282, 0.23562613277402947),
                       (0.5, 1e-9, 0.5, 2e-9)]:
            with self.subTest(inputs=inputs):
                forward = equal_density_threshold(*inputs)
                reverse = equal_density_threshold(inputs[2], inputs[3], inputs[0], inputs[1])
                self.assertEqual(forward["threshold_rule"], reverse["threshold_rule"])
                self.assertAlmostEqual(forward["threshold"], reverse["threshold"],
                                       delta=2 * math.ulp(forward["threshold"]))
                self.assertEqual(len(forward["crossings"]), len(reverse["crossings"]))

    def test_equal_variances_select_midpoint_crossing(self):
        for inputs, expected, gap_limit in [((0.1, 0.2, 0.5, 0.2), 0.3, 1e-14),
                                           ((0.5, 1e-9, 0.50000001, 1e-9),
                                            0.500000005, 7e-7)]:
            with self.subTest(inputs=inputs):
                params = equal_density_threshold(*inputs)
                self.assertEqual(params["threshold_rule"], "equal_density_between_means")
                self.assertAlmostEqual(params["threshold"], expected, delta=math.ulp(expected))
                self.assertLess(abs(log_density_gap(params["threshold"], *inputs)), gap_limit)

    def test_zero_deviations_are_floored_and_keep_midpoint_crossing(self):
        params = equal_density_threshold(0.0, 0.0, 0.1, 0.0)
        self.assertEqual(params["threshold"], 0.05)
        self.assertEqual(params["threshold_rule"], "equal_density_between_means")
        self.assertEqual(params["sigma_floor"], 1e-9)
        self.assertEqual(params["sigma_sil_effective"], 1e-9)
        self.assertEqual(params["sigma_sp_effective"], 1e-9)
        self.assertTrue(params["sigma_was_floored"])

    def test_floored_deviation_uses_effective_sigma_in_crossing(self):
        params = equal_density_threshold(0.5, 0.0, 0.50000001, 2e-9)
        self.assertTrue(params["sigma_was_floored"])
        self.assertEqual(params["sigma_sil_effective"], 1e-9)
        self.assertAlmostEqual(params["threshold"], 0.5000000034705506,
                               delta=math.ulp(0.5))
        self.assertLess(abs(log_density_gap(params["threshold"], 0.5, 1e-9,
                                            0.50000001, 2e-9)), 5e-7)

    def test_adjacent_means_have_no_unequal_variance_crossing_between_them(self):
        mu_sp = math.nextafter(0.5, math.inf)
        params = equal_density_threshold(0.5, 1e-9, mu_sp, 2e-9)
        self.assertEqual(params["threshold_rule"], "no_between_means_crossing_midpoint")
        self.assertEqual(params["threshold"], (0.5 + mu_sp) / 2.0)

    def test_near_equal_variances_still_satisfy_log_density_equality(self):
        inputs = (0.5, 1e-9, 0.50000001, 1e-9 * (1.0 + 1e-13))
        params = equal_density_threshold(*inputs)
        self.assertEqual(params["threshold_rule"], "equal_density_between_means")
        self.assertLess(abs(log_density_gap(params["threshold"], *inputs)), 7e-7)

    def test_identical_distributions_keep_explicit_midpoint_fallback(self):
        params = equal_density_threshold(0.5, 0.0, 0.5, 0.0)
        self.assertEqual(params["threshold"], 0.5)
        self.assertEqual(params["threshold_rule"], "no_between_means_crossing_midpoint")
        self.assertEqual(params["crossings"], [])

    def test_roots_just_outside_means_use_same_fallback_when_classes_swap(self):
        # Decimal100 exact-float oracles put both roots outside the mean interval.
        # Subtracting large sigma logs previously flipped the endpoint root sign.
        cases = [(1e-9, 2e-9, 2.1774100225154735e-9, 1e-9),
                 (.001, .02, .012774100225154744, .01)]
        for inputs in cases:
            expected = inputs[0] + (inputs[2] - inputs[0]) / 2.
            for orientation in (inputs, (inputs[2], inputs[3], inputs[0], inputs[1])):
                with self.subTest(inputs=orientation):
                    params = equal_density_threshold(*orientation)
                    self.assertEqual(params['threshold_rule'], 'no_between_means_crossing_midpoint')
                    self.assertEqual(params['threshold'], expected)

    def test_population_statistics_still_divide_variance_by_observation_count(self):
        self.assertEqual(population_statistics([0.0, 1.0]), (0.5, 0.5))
        self.assertEqual(population_statistics([0.5, 0.5]), (0.5, 0.0))


if __name__ == "__main__":
    unittest.main()
