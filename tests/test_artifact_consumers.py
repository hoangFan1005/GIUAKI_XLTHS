import unittest
from tools.export_slide_data import numeric_parameters
from tools import package_submission

class ArtifactConsumerTests(unittest.TestCase):
    def model(self):
        return {'schema_version': 2, 'evaluation_protocol': 'train_selected_reused_test',
                'parameter_selection_set': 'train', 'historical_test_exposure': True,
                'endpoint_noise': {'noise_mean': 0.1}, 'W': 7,
                'candidate_frame_selected_W': 1, 'candidate_frame_f1': 0.8,
                'candidate_frame_selection_scores': [{'W': 1, 'candidate_frame_f1': 0.8}],
                'candidate_cleanup_records': {'full_pipeline_records': 4},
                'W_selection_train': {'selected_W': 7, 'shared_optimal_W': [],
                    'candidate_W': [1, 7], 'evaluated_files': ['TRAIN_A'],
                    'selection_set': 'train', 'evaluation_protocol': 'train_final_calibration',
                    'tie_preference_W': 20}}
    def test_export_preserves_train_and_candidate_provenance(self):
        out = numeric_parameters(self.model())
        self.assertEqual(out.get('candidate_frame_selected_W'), 1)
        self.assertEqual(out.get('historical_test_exposure'), True)
        self.assertEqual(out.get('selection'), self.model()['W_selection_train'])
        self.assertEqual(out.get('candidate_frame_f1'), 0.8)
        self.assertEqual(out.get('candidate_cleanup_records'), {'full_pipeline_records': 4})
    def test_stale_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'schema'):
            numeric_parameters({'endpoint_noise': {}})
    def test_package_prose_uses_saved_metadata(self):
        self.assertTrue(hasattr(package_submission, 'model_readme'), 'missing metadata README helper')
        prose = package_submission.model_readme(2, self.model())
        for fragment in ('W = 7', 'TRAIN_A', 'candidate_frame_f1', '0.8', '20', 'previous version', 'historical_test_exposure', 'mae_ms'):
            self.assertIn(fragment, prose)
        self.assertNotIn('test-tuned', prose)
    def test_package_main_preserves_notebook_readme(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'submission').mkdir()
            readme = root / 'submission/README.md'
            readme.write_text('JUPYTER_NOTEBOOKS_CODE_ONLY.zip primary', encoding='utf-8')
            with patch.object(package_submission, 'ROOT', root), patch.object(package_submission, 'build_package') as build:
                package_submission.main()
            self.assertEqual(build.call_count, 3)
            self.assertEqual(readme.read_text(encoding='utf-8'), 'JUPYTER_NOTEBOOKS_CODE_ONLY.zip primary')

    def test_package_rejects_test_calibration_in_schema2(self):
        self.assertTrue(hasattr(package_submission, 'model_readme'))
        model = self.model()
        model['W_selection_train']['selection_set'] = 'test'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            package_submission.model_readme(2, model)
