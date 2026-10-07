"""FINAL speech-high domain is explicit; the raw Gaussian classifier is general."""
import unittest

from algorithms import tt3_gaussian as gaussian
from app.pipeline import detect_regions, fit_training_model


class GaussianDirectionContractTests(unittest.TestCase):
    def setUp(self):
        self.features = {
            'ste_norm': [.9, 1.] + [.1]*10 + [.9, 1.],
            'starts': [i*.01 for i in range(14)],
            'ends': [i*.01+.025 for i in range(14)],
        }
        self.params = {
            'threshold': .5,
            'endpoint_noise': {'noise_q95': .01, 'noise_upper': .02, 'high_multiplier': 1.5},
        }

    def test_low_direction_is_rejected_instead_of_silently_erasing_speech(self):
        params = dict(self.params, speech_direction='low')
        with self.assertRaisesRegex(ValueError, "FINAL.*speech_direction.*high"):
            detect_regions('tt3', self.features, .155, params)

    def test_unknown_direction_is_not_silently_treated_as_high(self):
        with self.assertRaisesRegex(ValueError, "speech_direction.*high"):
            detect_regions('tt3', self.features, .155,
                           dict(self.params, speech_direction='unknown'))

    def test_train_fit_rejects_low_direction_before_returning_a_final_model(self):
        record = {'name': 'inverted_means', 'split': 'train',
                  'features': {'ste_norm': [.9, 1., .1, .2]}, 'labels': [0, 0, 1, 1]}
        with self.assertRaisesRegex(ValueError, "FINAL.*speech_direction.*high"):
            fit_training_model('tt3', [record])

    def test_raw_gaussian_classifier_still_supports_low_direction(self):
        record = {'features': {'ste_norm': [.9, 1., .1, .2]}, 'labels': [0, 0, 1, 1]}
        raw_model = gaussian.fit([record])
        self.assertEqual(raw_model['speech_direction'], 'low')
        self.assertEqual(gaussian.predict(record['features'], raw_model), [0, 0, 1, 1])

    def test_explicit_and_legacy_high_direction_keep_same_final_result(self):
        legacy = detect_regions('tt3', self.features, .155, self.params)
        explicit = detect_regions('tt3', self.features, .155,
                                  dict(self.params, speech_direction='high'))
        self.assertEqual(legacy, explicit)
        self.assertEqual(legacy['final_regions'], [(0., .155)])


if __name__ == '__main__':
    unittest.main()
