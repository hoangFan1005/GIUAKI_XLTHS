import unittest
from app.cli import parse_args
from app.pipeline import summarize
from tools import build_notebooks as builder

class EndpointModeIntegrationTests(unittest.TestCase):
    def test_cli_modes(self):
        self.assertEqual(parse_args([]).endpoint_mode, 'enhanced')
        self.assertTrue(parse_args(['--compare-endpoint-modes']).compare_endpoint_modes)
        self.assertEqual(parse_args(['--endpoint-mode','core']).endpoint_mode, 'core')

    def test_summary_groups_modes_and_keeps_primary_undefined(self):
        common=dict(algorithm='tt1',file='a',frame_f1=1.,ground_truth_region_count=1,predicted_region_count=1,mae_ms=2.,rmse_ms=2.)
        rows=[dict(common,endpoint_mode='enhanced'),dict(common,endpoint_mode='core'),dict(common,file='b',endpoint_mode='core',predicted_region_count=2,mae_ms=None,rmse_ms=None)]
        summary=summarize(rows)
        self.assertEqual(len(summary),2)
        core=next(row for row in summary if row['endpoint_mode']=='core')
        self.assertIsNone(core['mean_file_mae_ms'])
        self.assertEqual(core['valid_files_mean_file_mae_ms'],2.)
        malformed=summarize([dict(common,endpoint_mode='core',predicted_region_count=2)])[0]
        self.assertIsNone(malformed['mean_file_mae_ms'])

    def test_notebook_locks_both_modes(self):
        self.assertIn('MODELS',builder.FIT_MODEL)
        self.assertIn('MODEL_LOCK_DIGESTS',builder.LOCK_MODEL)
        self.assertIn('ALL_RESULTS_BY_MODE',builder.EVALUATION)

    def test_schema_three_requires_self_excluded_calibration_digest(self):
        from tools import run_notebooks as runner
        model=dict(schema_version=3, metrics_schema_version=2, parameter_selection_set='train',
                   evaluation_protocol='train_selected_reused_test', historical_test_exposure=True,
                   training_files=list(runner.TRAIN_NAMES), endpoint_mode='core',minimum_speech_ms=0.,
                   minimum_silence_ms=200.,endpoint_noise=None,boundary_convention='union of active frame supports',
                   padding_stage='excluded from core FINAL')
        with self.assertRaisesRegex(ValueError,'calibration digest'):
            runner.audit_model_protocol(model,runner.model_digest(model))

    def test_weight_sweep_checks_declared_geometry(self):
        from app.weight_selection import sweep_final_weights
        model=dict(endpoint_mode='core',boundary_convention='union of active frame supports')
        def predictor(*args):
            return dict(metrics=dict(ground_truth_region_count=1,predicted_region_count=1,mae_ms=0.,status='ok',start_error_ms=0.,end_error_ms=0.), diagnostic=dict(endpoint_mode='core',geometry='frame centers',energy_threshold=1.,low_ste_threshold=None,high_ste_threshold=None,raw_speech_frames=1,candidate_regions=[]),final_regions=[])
        with self.assertRaisesRegex(ValueError,'geometry|boundary_convention'):
            sweep_final_weights([dict(name='a',split='train')],model,predictor,candidates=(1.,))

    def test_core_plot_uses_native_threshold_with_nullable_low_high(self):
        import matplotlib.pyplot as plt
        from app.plotting import _plot_axis
        record=dict(name='a',samples=[0.,1.,0.],sample_rate=100,duration=.03,
                    features=dict(centers=[.01,.02],ste_norm=[.5,1.],energy=[2.,4.]))
        result=dict(diagnostic=dict(endpoint_mode='core',native_threshold=2.,native_threshold_units='sum of squared samples',
                                    low_ste_threshold=None,high_ste_threshold=None),ground_truth_boundaries=[],predicted_boundaries=[],
                    metrics=dict(mae_ms=None),final_regions=[])
        fig,axis=plt.subplots()
        try:
            _plot_axis(axis,record,result)
            labels=[item.get_text() for item in axis.get_legend().get_texts()]
            self.assertTrue(any('Native T' in label for label in labels))
            self.assertFalse(any('High STE' in label or 'Low STE' in label for label in labels))
            self.assertIn('core',axis.get_title())
            self.assertIn('FINAL-region endpoint MAE',axis.get_title())
        finally:plt.close(fig)

    def test_cli_locks_every_algorithm_in_both_modes_before_test_read(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app import pipeline
        from tests.test_notebook_protocol import tiny_dataset
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); tiny_dataset(root)
            events=[]
            real_fit,real_load=pipeline.fit_training_model,pipeline.load_audio_folder
            def fit(algorithm,records,**kwargs):
                result=real_fit(algorithm,records,**kwargs)
                events.append(('fit',algorithm,kwargs['endpoint_mode']))
                return result
            def load(folder):
                if folder==root/'test':
                    self.assertEqual(set(events),{('fit',algorithm,mode) for algorithm in ('tt1','tt2','tt3') for mode in ('core','enhanced')})
                return real_load(folder)
            with patch.multiple(pipeline,TRAIN_DIR=root/'train',TEST_DIR=root/'test',OUTPUT_DIR=root/'outputs'), \
                 patch.object(pipeline,'fit_training_model',side_effect=fit),patch.object(pipeline,'load_audio_folder',side_effect=load), \
                 patch.object(pipeline,'make_file_figure',return_value=None),patch.object(pipeline,'plot_gaussian_training'):
                pipeline.run_experiment(parse_args(['--evaluate-all','--compare-endpoint-modes']))
            self.assertTrue((root/'outputs/endpoint_modes/comparison/tables/all_metrics.csv').is_file())
            self.assertFalse((root/'outputs/models').exists())

    def test_default_main_dispatches_once_and_demo_retains_four_test_figures(self):
        import contextlib,io,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app import cli,pipeline
        from tests.test_notebook_protocol import tiny_dataset
        with patch('matplotlib.use'),patch.object(pipeline,'run_experiment') as run:
            cli.main([])
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0].endpoint_mode,'enhanced')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tiny_dataset(root)
            with patch.multiple(pipeline,TRAIN_DIR=root/'train',TEST_DIR=root/'test',OUTPUT_DIR=root/'outputs'), \
                 patch.object(pipeline,'make_file_figure',return_value=None),patch.object(pipeline,'plot_gaussian_training'), \
                 patch.object(pipeline,'show_figures',return_value=[]) as show,contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_experiment(parse_args([]))
            show.assert_called_once()
            self.assertEqual(len(show.call_args.args[0]),4)
