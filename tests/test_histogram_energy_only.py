"""Teacher's energy-only TT2: centroid cannot gate speech or be calculated."""
import unittest

from algorithms import tt2_histogram as tt2
from app.pipeline import prepare_records, predict_and_score
from core.metrics import frame_labels


class HistogramEnergyOnlyTests(unittest.TestCase):
    def params(self):
        return dict(variant='source', bins=64, smooth_radius=2, W=20., padding_frames=0)

    def test_predict_accepts_energy_without_centroid(self):
        values = [0.] * 40 + [1.] + [0.] * 39
        try:
            mask, diagnostic = tt2.predict({'energy': values}, self.params())
        except KeyError as error:
            self.fail(f'Energy-only TT2 must not require {error}')
        self.assertEqual(mask, [0] * 40 + [1] + [0] * 39)
        self.assertEqual(diagnostic['feature'], 'energy')
        self.assertNotIn('centroid_threshold', diagnostic)

    def test_supplied_low_centroid_cannot_reject_high_energy(self):
        values = [0.] * 40 + [1.] + [0.] * 39
        mask, _ = tt2.predict({'energy': values, 'centroid': [0.] * 80}, self.params())
        self.assertEqual(mask[40], 1)

    def test_histogram_preparation_does_not_calculate_centroid(self):
        audio = dict(name='fixture', samples=[0.] * 30 + [1.] * 40 + [0.] * 30,
                     sample_rate=1000, duration=.1,
                     intervals=[(0., .03, 'sil'), (.03, .07, 'v'), (.07, .1, 'sil')])
        prepared = prepare_records([audio], source_histogram=True)[0]
        self.assertNotIn('centroid', prepared['features'])
        self.assertEqual(prepared['features']['frame_size'], 25)

    def test_final_regions_use_energy_seed_without_centroid(self):
        values = [0.] * 160
        values[10:35] = [.8] * 25
        values[45:70] = [.8] * 25
        values[105:135] = [.8] * 30
        starts = [i * .01 for i in range(160)]
        ends = [s + .025 for s in starts]
        intervals = [(0., .1, 'sil'), (.1, .715, 'v'), (.715, 1.05, 'sil'),
                     (1.05, 1.365, 'uv'), (1.365, 1.62, 'sil')]
        record = dict(name='two_utterances', duration=1.62, intervals=intervals,
                      features=dict(energy=values, ste_norm=values, starts=starts, ends=ends),
                      labels=frame_labels([(s + e) / 2 for s, e in zip(starts, ends)], intervals))
        model = dict(self.params(), endpoint_noise=dict(noise_q95=.02, noise_upper=.15))
        try:
            result = predict_and_score('tt2', record, model)
        except KeyError as error:
            self.fail(f'Final energy-only pipeline must not require {error}')
        self.assertEqual(len(result['final_regions']), 2)
        self.assertAlmostEqual(result['metrics']['mae_ms'], 0.)
        for actual, expected in zip(result['predicted_boundaries'], [.1, .715, 1.05, 1.365]):
            self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
