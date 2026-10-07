"""Focused support-timing coverage and a reproducible TRAIN-only diagnostic."""
import copy
import importlib
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from algorithms import tt2_histogram as tt2
from app.config import TRAIN_DIR, TEST_DIR
from app.pipeline import detect_regions, fit_training_model, load_audio_folder, prepare_records
from core.endpoints import hysteresis_regions
from core.features import compute_features
from core.metrics import region_endpoint_metrics
from core.postprocess import fill_short_internal_silences, mask_segments


def regular(values, fs):
    width = round(fs*.025)/fs
    starts = [i*.01 for i in range(len(values))]
    ends = [s+width for s in starts]
    return starts, ends, ends[-1]


class SupportContractTests(unittest.TestCase):
    def assertRegions(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_region, expected_region in zip(actual, expected):
            for actual_boundary, expected_boundary in zip(actual_region, expected_region):
                self.assertAlmostEqual(actual_boundary, expected_boundary, places=11)

    def test_support_gap_equality_and_cleanup_agree(self):
        # A shifted second run isolates the comparator from the 25/10ms lattice.
        for fs in (16000, 44100):
            width = round(fs*.025)/fs
            for gap, count in ((.190, 1), (.200, 2), (.210, 2), (.250, 2)):
                with self.subTest(fs=fs, gap=gap):
                    right = .3+width+gap
                    zeros = []
                    offset = .31
                    while offset < right-1e-12:
                        zeros.append(offset)
                        offset += .01
                    starts = [0.]+[.1+i*.01 for i in range(21)]+zeros
                    starts += [right+i*.01 for i in range(21)]+[right+.4]
                    ends = [s+width for s in starts]
                    values = [0.]+[.8]*21+[0.]*len(zeros)+[.8]*21+[0.]
                    result = hysteresis_regions(values, starts, ends, ends[-1], .1, .5, .2, .1)
                    expected = [(.1, right+.2+width)] if count == 1 else [(.1, .3+width), (right, right+.2+width)]
                    self.assertRegions(result, expected)
                    cleaned = fill_short_internal_silences([int(v>.5) for v in values], starts,
                                                          ends[-1], .2, frame_ends=ends)
                    self.assertRegions([(s,e) for s,e,k in mask_segments(cleaned, starts, ends[-1], ends) if k], expected)

    def test_inactive_frame_count_is_not_silence_duration(self):
        for fs in (16000, 44100):
            for count, wanted in ((20, 1), (21, 1), (22, 2)):
                with self.subTest(fs=fs, inactive_frames=count):
                    values = [0.]*10+[.8]*21+[0.]*count+[.8]*21+[0.]*30
                    starts, ends, duration = regular(values, fs)
                    gap = starts[31+count]-ends[30]
                    self.assertAlmostEqual(gap, (count+1)*.01-round(fs*.025)/fs)
                    self.assertEqual(len(hysteresis_regions(values, starts, ends, duration, .1, .5, .2, .1)), wanted)

    def test_minimum_span_eight_removed_nine_retained(self):
        for fs in (16000, 44100):
            for count, wanted in ((8, 0), (9, 1)):
                with self.subTest(fs=fs, active_frames=count):
                    values = [0.]*10+[.8]*count+[0.]*30
                    starts, ends, duration = regular(values, fs)
                    self.assertEqual(len(hysteresis_regions(values, starts, ends, duration, .1, .5, .2, 0)), 1)
                    self.assertEqual(len(hysteresis_regions(values, starts, ends, duration, .1, .5, .2, .1)), wanted)

    def test_bridging_short_islands_precedes_span_filter(self):
        for fs in (16000, 44100):
            width = round(fs*.025)/fs
            for gap, wanted in ((.175, 1), (.200, 0)):
                with self.subTest(fs=fs, gap=gap):
                    right = .1+width+gap
                    starts = [0., .1, .15, right, right+.3]
                    ends = [s+width for s in starts]
                    values = [0., .8, 0., .8, 0.]
                    result = hysteresis_regions(values, starts, ends, ends[-1], .1, .5, .2, .1)
                    self.assertEqual(len(result), wanted)
                    if wanted:
                        self.assertAlmostEqual(result[0][1]-result[0][0], 2*width+gap)
                    else:
                        self.assertEqual(len(hysteresis_regions(values, starts, ends, ends[-1], .1, .5, .2, 0)), 2)

    def test_multi_regions_keep_individual_boundaries_and_mae(self):
        values = [0.]*160
        values[10:35] = [.8]*25
        values[45:70] = [.8]*25
        values[105:135] = [.8]*30
        for fs in (16000, 44100):
            with self.subTest(fs=fs):
                starts, ends, duration = regular(values, fs)
                actual = hysteresis_regions(values, starts, ends, duration, .1, .5, .2, .1)
                expected = [(.1, .69+round(fs*.025)/fs), (1.05, 1.34+round(fs*.025)/fs)]
                self.assertRegions(actual, expected)
                scores = region_endpoint_metrics(expected, actual)
                self.assertEqual(scores['matched_region_count'], 2)
                self.assertAlmostEqual(scores['mae_ms'], 0.)
                mismatch = region_endpoint_metrics(expected, [(actual[0][0], actual[-1][1])])
                self.assertIsNone(mismatch['mae_ms'])
                self.assertEqual(mismatch['missing_region_count'], 1)

    def test_zero_waveform_has_no_speech_at_both_rates(self):
        for fs in (16000, 44100):
            with self.subTest(fs=fs):
                f = compute_features([0.]*round(fs*.8), fs)
                self.assertEqual(hysteresis_regions(f['ste_norm'], f['starts'], f['ends'], .8, .1, .5), [])


class DiagnosticToolTests(unittest.TestCase):
    def setUp(self):
        # RED is an assertion failure for the missing deliverable, not an import error.
        self.assertIsNotNone(importlib.util.find_spec('tools.check_endpoint_robustness'),
                             'The bounded endpoint robustness tool is required')
        self.tool = importlib.import_module('tools.check_endpoint_robustness')

    def test_48_waveforms_match_independent_integer_overlap_and_known_limits(self):
        rows = self.tool.waveform_characterization()
        self.assertEqual(len(rows), 48)
        expected_cases = {(fs,gap,phase) for fs in (16000,44100)
                          for gap in (190,200,210,250)
                          for phase in (0,1,round(fs*.01)//4,round(fs*.01)//2,
                                        3*round(fs*.01)//4,round(fs*.01)-1)}
        self.assertEqual({(r['sample_rate_hz'],r['true_gap_ms'],r['phase_samples']) for r in rows}, expected_cases)
        for row in rows:
            with self.subTest(fs=row['sample_rate_hz'], gap=row['true_gap_ms'], phase=row['phase_samples']):
                samples, speech = self.tool.rectangular_case(row['sample_rate_hz'], row['phase_samples'], row['true_gap_ms'])
                fs = row['sample_rate_hz']
                f = compute_features(samples, fs)
                n, h = round(fs*.025), round(fs*.01)
                self.assertEqual(row['frame_size'], n)
                self.assertEqual(row['hop_size'], h)
                self.assertEqual(len(f['starts']), 1+(len(samples)-n)//h)
                self.assertEqual(f['ends'][-1], ((len(f['starts'])-1)*h+n)/fs)
                independent_values = []
                active = []
                for index in range(len(f['ste_norm'])):
                    count = sum(max(0, min(index*h+n,b)-max(index*h,a)) for a,b in speech)
                    independent_values.append(count/n)
                    if count*10 >= n:
                        if active and index*h <= active[-1][1]:
                            active[-1] = (active[-1][0], index*h+n)
                        else:
                            active.append((index*h, index*h+n))
                self.assertEqual(f['ste_norm'], independent_values)
                self.assertAlmostEqual(row['support_gap_ms'], (active[1][0]-active[0][1])/fs*1000)
                self.assertEqual(row['predicted_region_count'], 2 if row['true_gap_ms']==250 else 1)
                self.assertEqual(row['ground_truth_region_count'], 2)
                if row['true_gap_ms'] in (200,210):
                    self.assertEqual(row['status'], 'known_limitation')
                    self.assertIsNone(row['mae_ms'])
                    self.assertEqual(row['missing_region_count'], 1)
                elif row['true_gap_ms']==190:
                    self.assertEqual(row['status'], 'expected_short_gap_merge')
                    self.assertIsNone(row['mae_ms'])
                else:
                    self.assertEqual(row['status'], 'estimated_gap_preserved')
                    self.assertIsNotNone(row['mae_ms'])

    def test_physical_75ms_burst_survives_estimated_span_filter(self):
        rows = self.tool.short_burst_characterization()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['true_speech_ms'], 75.)
            self.assertGreater(row['estimated_span_ms'], 100.)
            self.assertEqual(row['predicted_region_count'], 1)
            self.assertEqual(row['status'], 'physical_duration_differs_from_span')

    def test_ablation_rejects_nontrain_before_detection(self):
        for split in ('test','external',None):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'TRAIN'):
                self.tool.ablate_training([dict(split=split)], {})
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.tool.ablate_training([dict(split='train', wav_path=str(TEST_DIR/'phone_F2.wav'))], {})
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            self.tool.ablate_training([], {})

    def test_invalid_counts_accompany_summary_and_missing_mae_remains_none(self):
        rows = [dict(algorithm='tt1', arm='raw0', mae_ms=None, rmse_ms=None,
                     missing_region_count=1, extra_region_count=0),
                dict(algorithm='tt1', arm='raw0', mae_ms=10., rmse_ms=12.,
                     missing_region_count=0, extra_region_count=0)]
        summary = self.tool.ablation_summary(rows)[0]
        self.assertEqual(summary['invalid_count'], 1)
        self.assertEqual(summary['valid_count'], 1)
        self.assertIsNone(summary['mean_mae_ms'])
        self.assertEqual(summary['valid_rows_mean_mae_ms'], 10.)

    def test_histogram_raw_ablation_excludes_candidate_padding(self):
        samples = [0.]*3200+[1.]*2400+[0.]*10400
        record = prepare_records([dict(name='unpadded_fixture', split='train', samples=samples,
            sample_rate=16000, duration=1., intervals=[(0.,.2,'sil'),(.2,.35,'v'),(.35,1.,'sil')])])[0]
        noise = dict(noise_q95=.01,noise_upper=.03,high_multiplier=1.5)
        params = dict(variant='source',bins=64,smooth_radius=2,W=20.,padding_frames=25,
                      endpoint_noise=noise,frame_ms=25.,hop_ms=10.)
        models = dict(tt1=dict(threshold=.1,endpoint_noise=noise), tt2=params,
                      tt3=dict(threshold=.1,speech_direction='high',endpoint_noise=noise))
        padded, diagnostic = tt2.predict(record['features'],params)
        raw = [int(e>diagnostic['energy_threshold']) for e in record['features']['energy']]
        self.assertGreater(sum(padded),sum(raw))
        rows = self.tool.ablate_training([record],models)
        for row in (r for r in rows if r['algorithm']=='tt2'):
            self.assertEqual(row['seed_count'],sum(raw))
            self.assertEqual(row['native_threshold'],diagnostic['energy_threshold'])
            if row['arm']=='raw0':
                self.assertGreater(row['final_regions'][0][0],0.)
                self.assertLess(row['final_regions'][0][1],.4)

    def test_real_train_ablation_matches_pipeline_preserves_inputs_and_loader_scope(self):
        records = prepare_records(load_audio_folder(TRAIN_DIR))
        models = {a:fit_training_model(a,records)[0] for a in ('tt1','tt2','tt3')}
        before_records, before_models = copy.deepcopy(records), copy.deepcopy(models)
        rows = self.tool.ablate_training(records, models)
        self.assertEqual(len(rows), 48)
        self.assertEqual(records, before_records)
        self.assertEqual(models, before_models)
        self.assertEqual({row['arm'] for row in rows}, {'raw0','raw100','hysteresis0','hysteresis100'})
        for row in rows:
            self.assertIn('native_threshold', row)
            self.assertIn('base_normalized_threshold', row)
            self.assertIn('last_raw_support_end_s', row)
            self.assertIn('last_low_support_end_s', row)
            self.assertIn('last_high_seed_support_end_s', row)
            self.assertIn('signed_endpoint_errors_ms', row)
            self.assertEqual(row['W'], models[row['algorithm']].get('W'))
            if row['arm']=='hysteresis100':
                record = next(r for r in records if r['name']==row['file'])
                expected = detect_regions(row['algorithm'], record['features'], record['duration'], models[row['algorithm']])
                self.assertEqual(row['final_regions'], expected['final_regions'])
        # The I/O boundary is captured; downstream computation remains real.
        calls = []
        def captured_load(folder):
            calls.append(folder)
            self.assertEqual(folder, TRAIN_DIR)
            return records
        with patch.object(self.tool, 'load_audio_folder', side_effect=captured_load), \
                patch.object(self.tool, 'prepare_records', side_effect=lambda value:value), \
                patch.object(self.tool, 'fit_training_model', side_effect=lambda a,r:(models[a],[])), \
                patch.object(self.tool, 'write_diagnostics') as writer, redirect_stdout(io.StringIO()):
            payload = self.tool.main()
        self.assertEqual(calls, [TRAIN_DIR])
        self.assertEqual(len(payload['ablation']), 48)
        writer.assert_called_once()
        self.assertEqual(payload['manifest']['parameter_selection_set'], 'train')
        self.assertTrue(payload['manifest']['historical_test_exposure'])
        self.assertEqual(payload['manifest']['training_files'], [r['name'] for r in records])
        self.assertEqual(models['tt2']['W'], 20.)




class TimingResearchTests(unittest.TestCase):
    def setUp(self):
        self.tool=importlib.import_module('tools.check_endpoint_robustness')

    def test_cells_branch_scores_cell_membership_and_audio_edges(self):
        self.assertTrue(hasattr(self.tool,'timing_predictor'), 'Shared research predictor required')
        f=dict(centers=[.0125,.0225,.0325],starts=[0.,.01,.02],ends=[.025,.035,.045],ste_norm=[1.,0.,1.])
        record=dict(name='edge',split='train',features=f,duration=.05,labels=[1,1,1],intervals=[(0.,.05,'v')])
        params=dict(endpoint_mode='core',threshold=.5,geometry='cells',minimum_speech_ms=0.)
        result=self.tool.timing_predictor('tt1',record,params)
        self.assertEqual(result['final_regions'],[(0.,.05)])
        self.assertEqual(result['mask'],[1,1,1])
        self.assertEqual(result['metrics']['mae_ms'],0.)

    def test_trial_matrix_is_bounded_one_factor_and_rejects_nontrain(self):
        self.assertTrue(hasattr(self.tool,'TIMING_TRIALS'), 'Declared matrix required')
        trials=self.tool.TIMING_TRIALS
        self.assertEqual(len(trials),7)
        self.assertEqual(sum(len(t['algorithms']) for t in trials)*4,76)
        self.assertNotIn(('cells',0.,'enhanced'),{(t['geometry'],t['minimum_speech_ms'],t['endpoint_mode']) for t in trials})
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            self.tool.timing_training_study([dict(split='test')])

    def test_t1_zero_is_explicitly_unsupported_without_floor(self):
        self.assertTrue(hasattr(self.tool,'timing_predictor'))
        f=dict(centers=[.0125],starts=[0.],ends=[.025],ste_norm=[1.])
        record=dict(name='zero',split='train',features=f,duration=.03,labels=[1],intervals=[(0.,.03,'v')])
        params=dict(endpoint_mode='enhanced',threshold=0.,geometry='support',minimum_speech_ms=100.,low_rule='T1',
                    endpoint_noise=dict(noise_upper=.03))
        with self.assertRaisesRegex(ValueError,'unsupported LOW=T1'):
            self.tool.timing_predictor('tt1',record,params)

class TimingFoldIntegrationTests(unittest.TestCase):
    def test_real_train_folds_budget_locked_geometry_and_full_undefined_scores(self):
        tool=importlib.import_module('tools.check_endpoint_robustness')
        records=prepare_records(load_audio_folder(TRAIN_DIR))
        before=copy.deepcopy(records)
        with redirect_stdout(io.StringIO()):payload=tool.timing_training_study(records)
        self.assertEqual(records,before)
        self.assertEqual(len(payload['heldout']),76)
        self.assertEqual(len(payload['sweep']),3600)
        self.assertEqual(len(payload['models']),76)
        for entry in payload['models']:
            self.assertEqual(len(entry['fit_files']),3)
            self.assertNotIn(entry['heldout_file'],entry['fit_files'])
            model=entry['model']
            self.assertEqual(model['historical_test_exposure'],True)
            self.assertEqual(model['boundary_convention'],'nearest-frame centered decision cells' if model['geometry']=='cells' else 'union of active frame supports')
            if entry['algorithm']=='tt2':
                chosen=model['W_selection_train']
                self.assertEqual(chosen['candidate_W'],[float(i) for i in range(1,51)])
                self.assertEqual(chosen['minimum_speech_ms'],model['minimum_speech_ms'])
                self.assertEqual(chosen['boundary_convention'],model['boundary_convention'])
                self.assertEqual(model['W'],model['finalW'])
        for summary in payload['summary']:
            if summary['invalid_count']:
                self.assertIsNone(summary['mean_mae_ms'])
                self.assertIsNone(summary['mean_file_rmse_ms'])
        self.assertEqual(len(payload['comparisons']),52)

if __name__ == '__main__':
    unittest.main()
