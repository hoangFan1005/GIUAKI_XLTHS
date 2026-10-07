"""TRAIN calibration, LAB-free inference and unambiguous exported metrics."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from algorithms import tt2_histogram as tt2
from app import pipeline
from app.cli import parse_args
from app.config import TRAIN_DIR, TEST_DIR
from core.metrics import frame_labels


def synthetic_record():
    values = [0.] * 80
    values[10:50] = [.8] * 40
    starts = [i * .01 for i in range(80)]
    ends = [start + .025 for start in starts]
    intervals = [(0., .1, 'sil'), (.1, .515, 'v'), (.515, .82, 'sil')]
    return dict(name='array_fold', split='train', duration=.82, intervals=intervals,
                features=dict(energy=values, ste_norm=values, starts=starts, ends=ends),
                labels=frame_labels([(s + e) / 2 for s, e in zip(starts, ends)], intervals))


class CurrentArtifactDocumentationTests(unittest.TestCase):
    """Prevent current artifact labels from drifting to legacy TEST/schema 1."""
    root = Path(__file__).resolve().parents[1]

    def test_current_w_sweep_contains_train_names_and_unique_headers(self):
        with (self.root / 'outputs/tables/tt2_w_selection/sweep.csv').open(
                encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        self.assertEqual(len(rows), 200)
        self.assertEqual({row['filename'] for row in rows},
                         {'phone_F1', 'phone_M1', 'studio_F1', 'studio_M1'})
        self.assertEqual(len(fields), len({key.casefold() for key in fields}))
        self.assertIn('final_region_mae_ms', fields)

    def test_current_metric_exports_have_schema_two_unique_headers(self):
        for folder in ('all', 'all_all_dataset', 'tt1', 'tt2', 'tt3'):
            with self.subTest(folder=folder):
                with (self.root / 'outputs/tables' / folder / 'test_metrics.csv').open(
                        encoding='utf-8-sig', newline='') as handle:
                    reader = csv.DictReader(handle)
                    fields = reader.fieldnames
                    rows = list(reader)
                self.assertEqual(len(fields), len({key.casefold() for key in fields}))
                self.assertTrue({'mae_ms', 'rmse_ms', 'tolerance_boundary_mae_ms'} <= set(fields))
                self.assertNotIn('boundary_MAE_ms', fields)
                self.assertTrue(all(row['metrics_schema_version'] == '2' for row in rows))

    def test_outputs_readme_labels_current_sweep_as_train_and_schema_two(self):
        text = (self.root / 'outputs/README.md').read_text(encoding='utf-8')
        sweep_line = next(line for line in text.splitlines() if '`tables/tt2_w_selection/`' in line)
        self.assertIn('TRAIN', sweep_line)
        self.assertNotIn('test', sweep_line.casefold())
        self.assertIn('`mae_ms` / `rmse_ms`', text)
        self.assertNotIn('`boundary_MAE_ms`', text)
        self.assertNotIn('`boundary_mae_ms`', text)
        self.assertIn('historical_test_exposure=true', text)

    def test_submission_readme_separates_current_acceptance_from_oct05_history(self):
        text = (self.root / 'submission/README.md').read_text(encoding='utf-8')
        self.assertIn('## Lịch sử 05/10/2026', text)
        current, history = text.split('## Lịch sử 05/10/2026', 1)
        self.assertIn('07/10/2026', current)
        self.assertIn('133/133', current)
        self.assertNotIn('98/98', current)
        self.assertIn('98/98', history)
        self.assertIn('historical_test_exposure=true', current)
        self.assertIn('schema 2', current)


class DetectionContractTests(unittest.TestCase):
    def test_lab_free_detection_and_changed_lab_only_change_scoring(self):
        self.assertTrue(callable(getattr(pipeline, 'detect_regions', None)),
                        'A detector accepting features/duration/model without LAB is required')
        record = synthetic_record()
        params = dict(threshold=.1, endpoint_noise=dict(noise_q95=.02, noise_upper=.15))
        detected = pipeline.detect_regions('tt1', record['features'], record['duration'], params)
        scored = pipeline.predict_and_score('tt1', record, params)
        changed = dict(record, intervals=[(0., .2, 'sil'), (.2, .615, 'v'), (.615, .82, 'sil')],
                       labels=[0] * 80)
        rescored = pipeline.predict_and_score('tt1', changed, params)
        self.assertEqual(detected['final_regions'], [(.1, .515)])
        self.assertEqual(detected['mask'], scored['mask'])
        self.assertEqual(detected['diagnostic'], rescored['diagnostic'])
        self.assertEqual(scored['final_regions'], rescored['final_regions'])
        self.assertAlmostEqual(scored['metrics']['mae_ms'], 0.)
        self.assertAlmostEqual(rescored['metrics']['mae_ms'], 100.)

    def test_missing_noise_fails_without_loading_or_refitting(self):
        record = synthetic_record()
        with patch.object(pipeline, 'load_audio_folder', side_effect=AssertionError('inference I/O')), \
                patch.object(pipeline, 'fit_noise_floor', side_effect=AssertionError('inference refit')):
            with self.assertRaisesRegex(ValueError, 'endpoint_noise'):
                pipeline.predict_and_score('tt1', record, dict(threshold=.1))

    def test_primary_and_tolerance_metrics_have_distinct_casefold_unique_headers(self):
        record = synthetic_record()
        params = dict(threshold=.1, endpoint_noise=dict(noise_q95=.02, noise_upper=.15))
        metrics = pipeline.predict_and_score('tt1', record, params)['metrics']
        self.assertEqual(metrics.get('metrics_schema_version'), 2)
        self.assertIn('mae_ms', metrics)
        self.assertIn('rmse_ms', metrics)
        self.assertIn('matched_boundary_mae_ms', metrics)
        self.assertIn('tolerance_boundary_mae_ms', metrics)
        self.assertNotIn('boundary_MAE_ms', metrics)
        self.assertFalse(any(key.startswith('boundary_') and not key.startswith('boundary_inside_')
                             for key in metrics))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.csv'
            pipeline.write_csv(path, [metrics])
            with path.open(encoding='utf-8-sig', newline='') as handle:
                fields = next(csv.reader(handle))
            self.assertEqual(len(fields), len(set(key.casefold() for key in fields)))

    def test_csv_writer_rejects_case_insensitive_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'case|header|column'):
                pipeline.write_csv(Path(folder) / 'bad.csv', [{'boundary_MAE_ms': 1, 'boundary_mae_ms': 2}])

    def test_imported_legacy_test_calibration_is_never_labeled_train(self):
        record = synthetic_record()
        params = dict(variant='source', bins=64, smooth_radius=2, W=20.,
                      endpoint_noise=dict(noise_q95=.02, noise_upper=.15),
                      W_selection_test=dict(selection_set='test'))
        scores = pipeline.predict_and_score('tt2', record, params)['metrics']
        self.assertEqual(scores['parameter_selection_set'], 'test')
        self.assertEqual(scores['evaluation_protocol'], 'test_tuned_not_independent')
        self.assertEqual(scores['model_schema_version'], 1)
        self.assertEqual(scores['metrics_schema_version'], 2)
        self.assertTrue(scores['historical_test_exposure'])

    def test_summary_preserves_metric_version_and_test_exposure(self):
        record = synthetic_record()
        params = dict(threshold=.1, schema_version=2,
                      endpoint_noise=dict(noise_q95=.02, noise_upper=.15))
        scores = pipeline.predict_and_score('tt1', record, params)['metrics']
        summary = pipeline.summarize([dict(file='fixture', algorithm='tt1', **scores)])[0]
        self.assertEqual(summary.get('metrics_schema_version'), 2)
        self.assertEqual(summary.get('evaluation_protocol'), 'train_selected_reused_test')
        self.assertTrue(summary.get('historical_test_exposure'))


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = pipeline.prepare_records(pipeline.load_audio_folder(TRAIN_DIR))

    def require_fit(self):
        self.assertTrue(callable(getattr(pipeline, 'fit_training_model', None)),
                        'Models must be calibrated and locked from TRAIN before TEST reads')

    def test_loader_preserves_explicit_split_provenance(self):
        self.assertEqual({record.get('split') for record in self.training}, {'train'})
        testing = pipeline.load_audio_folder(TEST_DIR)
        self.assertEqual({record.get('split') for record in testing}, {'test'})

    def test_real_train_sweep_manifest_and_candidate_proposal_are_distinct(self):
        self.require_fit()
        proposal = tt2.fit(self.training)
        before = json.dumps(self.training, sort_keys=True)
        with patch.object(pipeline, 'load_audio_folder', side_effect=AssertionError('fit must use supplied TRAIN')), \
                patch.object(pipeline, 'read_wav', side_effect=AssertionError('fit must not read TEST')), \
                patch.object(pipeline, 'read_lab', side_effect=AssertionError('fit must not read LAB')):
            model, rows = pipeline.fit_training_model('tt2', self.training)
        selection = model['W_selection_train']
        self.assertEqual(len(rows), 200)
        self.assertEqual(selection['candidate_W'], list(range(1, 51)))
        self.assertEqual(selection['selection_set'], 'train')
        self.assertEqual(selection['evaluation_protocol'], 'train_final_calibration')
        self.assertEqual(selection['tie_preference_W'], 20)
        self.assertNotIn('previous_W', selection)
        self.assertEqual(set(selection['evaluated_files']), {'phone_F1', 'phone_M1', 'studio_F1', 'studio_M1'})
        self.assertEqual(model['W'], selection['selected_W'])
        self.assertEqual(model['finalW'], model['W'])
        self.assertEqual(model['candidate_frame_selected_W'], proposal['W'])
        self.assertEqual(model['candidate_frame_f1'], proposal['train_frame_f1'])
        self.assertEqual(model['candidate_cleanup_records']['full_pipeline_records'], 4)
        self.assertNotIn('train_frame_f1', model)
        self.assertNotIn('selection_scores', model)
        self.assertNotIn('W_selection_test', model)
        self.assertEqual(model['schema_version'], 2)
        self.assertEqual(model['metrics_schema_version'], 2)
        self.assertEqual(model['parameter_selection_set'], 'train')
        self.assertEqual(model['evaluation_protocol'], 'train_selected_reused_test')
        self.assertTrue(model['historical_test_exposure'])
        self.assertTrue(all('final_region_mae_ms' in row and 'boundary_MAE_ms' not in row for row in rows))
        self.assertEqual(before, json.dumps(self.training, sort_keys=True))

    def test_fit_rejects_test_external_and_spoofed_test_path_but_accepts_array_fold(self):
        self.require_fit()
        for split in ('test', 'external', None):
            record = dict(synthetic_record(), split=split)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'TRAIN|train'):
                pipeline.fit_training_model('tt1', [record])
        with self.assertRaisesRegex(ValueError, 'TRAIN|train'):
            pipeline.fit_training_model('tt1', [dict(synthetic_record(), wav_path=str(TEST_DIR / 'phone_F2.wav'))])
        model, rows = pipeline.fit_training_model('tt1', [synthetic_record()])
        self.assertEqual(rows, [])
        self.assertEqual(model['training_files'], ['array_fold'])

    def test_changed_test_lab_does_not_affect_model_noise_or_weight(self):
        self.require_fit()
        model, _ = pipeline.fit_training_model('tt2', self.training)
        record = pipeline.prepare_records(pipeline.load_audio_folder(TEST_DIR))[0]
        before = json.dumps(model, sort_keys=True)
        first = pipeline.predict_and_score('tt2', record, model)
        modified = dict(record, intervals=[(0., record['duration'], 'sil')], labels=[0] * len(record['labels']))
        second = pipeline.predict_and_score('tt2', modified, model)
        self.assertEqual(first['final_regions'], second['final_regions'])
        self.assertNotEqual(first['metrics']['mae_ms'], second['metrics']['mae_ms'])
        self.assertEqual(before, json.dumps(model, sort_keys=True))

    def test_single_file_fits_before_target_read_and_never_reloads_all_test(self):
        # Keep the real fits/scorers; suppress only artifacts and graphics.
        args = parse_args(['--algorithm', 'all', '--file', 'phone_F2.wav', '--no-show'])
        real_loader = pipeline.load_audio_file
        real_folder = pipeline.load_audio_folder
        lock = []
        self.require_fit()
        real_fit = pipeline.fit_training_model
        def fit(algorithm, records):
            value = real_fit(algorithm, records)
            lock.append(algorithm)
            return value
        def load_file(path):
            if Path(path).parent == TEST_DIR:
                self.assertEqual(set(lock), {'tt1', 'tt2', 'tt3'}, 'All models fit before TEST WAV/LAB read')
            return real_loader(path)
        def load_folder(folder):
            self.assertNotEqual(Path(folder), TEST_DIR, 'A single-file demo must not reload 4 TEST records')
            return real_folder(folder)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(pipeline, 'OUTPUT_DIR', Path(folder)), \
                patch.object(pipeline, 'load_audio_file', side_effect=load_file), \
                patch.object(pipeline, 'load_audio_folder', side_effect=load_folder), \
                patch.object(pipeline, 'fit_training_model', side_effect=fit), \
                patch.object(pipeline, 'make_file_figure', return_value=None), \
                patch.object(pipeline, 'plot_gaussian_training'), \
                patch.object(pipeline, 'plt'), contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_experiment(args)
            config = json.loads((Path(folder) / 'tables/all/single/phone_F2/run_config.json').read_text())
            self.assertEqual(config['metrics_schema_version'], 2)
            self.assertEqual(config['parameter_selection_set'], 'train')
            self.assertEqual(config['evaluation_protocol'], 'train_selected_reused_test')
            self.assertTrue(config['historical_test_exposure'])
            self.assertEqual(config['evaluated_files'], ['phone_F2'])
            with (Path(folder) / 'tables/all/single/phone_F2/test_metrics.csv').open(encoding='utf-8-sig', newline='') as handle:
                fields = next(csv.reader(handle))
            self.assertNotIn('boundary_MAE_ms', fields)
            self.assertEqual(len(fields), len(set(key.casefold() for key in fields)))


if __name__ == '__main__':
    unittest.main()
