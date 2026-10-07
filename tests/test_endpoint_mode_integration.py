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
        result=dict(diagnostic=dict(endpoint_mode='core',native_threshold=2.,native_threshold_units='mean squared sample amplitude',
                                    low_ste_threshold=None,high_ste_threshold=None),ground_truth_boundaries=[],predicted_boundaries=[],
                    metrics=dict(mae_ms=None),final_regions=[])
        fig,axis=plt.subplots()
        try:
            _plot_axis(axis,record,result)
            labels=[item.get_text() for item in axis.get_legend().get_texts()]
            self.assertTrue(any('Native T' in label for label in labels))
            self.assertFalse(any('High STE' in label or 'Low STE' in label for label in labels))
            self.assertIn('core',axis.get_title())
            self.assertEqual(list(fig.axes[1].lines[-1].get_ydata()),[.5,.5])
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

    def test_fixed_algorithm_all_dataset_preserves_shared_all_benchmarks(self):
        import contextlib,csv,io,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app import pipeline
        from tests.test_notebook_protocol import tiny_dataset
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tiny_dataset(root)
            with patch.multiple(pipeline,TRAIN_DIR=root/'train',TEST_DIR=root/'test',OUTPUT_DIR=root/'outputs'), \
                 patch.object(pipeline,'make_file_figure',return_value=None),patch.object(pipeline,'plot_gaussian_training'), \
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_experiment(parse_args(['--evaluate-all','--compare-endpoint-modes']))
                sentinels={}
                for mode in ('core','enhanced'):
                    target=root/'outputs/endpoint_modes'/mode/'tables/all'
                    for name in ('test_metrics.csv','summary.csv'):
                        path=target/name;sentinels[path]=path.read_bytes()
                for algorithm in ('tt1','tt2','tt3'):
                    pipeline.run_experiment(parse_args(['--evaluate-all','--compare-endpoint-modes'],fixed_algorithm=algorithm))
                    for path,original in sentinels.items():
                        self.assertEqual(path.read_bytes(),original, f'{algorithm} must preserve {path.name} shared benchmark')
                    for mode in ('core','enhanced'):
                        path=root/'outputs/endpoint_modes'/mode/'tables'/f'{algorithm}_all_dataset'/'test_metrics.csv'
                        with path.open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
                        self.assertEqual(len(rows),8)
                        self.assertEqual({row['algorithm'] for row in rows},{algorithm})

    def test_histogram_metadata_names_mean_squared_energy(self):
        from app import pipeline
        features=dict(ste_norm=[0.,.5,1.],energy=[0.,2.,4.],starts=[0.,.01,.02],ends=[.025,.035,.045])
        _,diagnostic=pipeline.algorithm_decision('tt2',features,dict(bins=1,smooth_radius=0,W=5.,padding_frames=25))
        self.assertEqual(diagnostic['native_threshold_units'],'mean squared sample amplitude')

    def test_partial_comparisons_preserve_complete_benchmark_and_export_scope(self):
        import contextlib,csv,io,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app import pipeline
        from tests.test_notebook_protocol import tiny_dataset
        def rows(path):
            with path.open(encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tiny_dataset(root)
            base=root/'outputs/endpoint_modes'
            comparison=base/'comparison/tables'
            with patch.multiple(pipeline,TRAIN_DIR=root/'train',TEST_DIR=root/'test',OUTPUT_DIR=root/'outputs'), \
                 patch.object(pipeline,'make_file_figure',return_value=None),patch.object(pipeline,'plot_gaussian_training'), \
                 contextlib.redirect_stdout(io.StringIO()):
                full=parse_args(['--evaluate-all','--compare-endpoint-modes'])
                pipeline.run_experiment(full)
                canonical=rows(comparison/'all_metrics.csv')
                self.assertEqual(len(canonical),8*3*2)
                protected=[comparison/name for name in ('all_metrics.csv','all_summary.csv','test_metrics.csv','test_summary.csv')]
                protected += [base/mode/'tables/all'/name for mode in ('core','enhanced') for name in ('test_metrics.csv','summary.csv')]
                sentinels={path:path.read_bytes() for path in protected}
                cases=[(parse_args(['--evaluate-all','--compare-endpoint-modes'],fixed_algorithm=algorithm),f'{algorithm}_all_dataset',8,{algorithm},None) for algorithm in ('tt1','tt2','tt3')]
                cases.append((parse_args(['--compare-endpoint-modes']),'all',4,{'tt1','tt2','tt3'},None))
                for algorithm in (None,'tt1','tt2','tt3'):
                    args=parse_args(['--file',str(root/'test/phone_F2.wav'),'--compare-endpoint-modes'],fixed_algorithm=algorithm)
                    cases.append((args,f'{algorithm or "all"}/single/phone_F2',1,{algorithm} if algorithm else {'tt1','tt2','tt3'},'phone_F2'))
                for args,scope,count,algorithms,file in cases:
                    with self.subTest(scope=scope):
                        pipeline.run_experiment(args)
                        for path,original in sentinels.items():self.assertEqual(path.read_bytes(),original,str(path))
                        scoped=rows(comparison/scope/'all_metrics.csv')
                        self.assertEqual(len(scoped),count*len(algorithms)*2)
                        self.assertEqual({row['algorithm'] for row in scoped},algorithms)
                        self.assertEqual({row['endpoint_mode'] for row in scoped},{'core','enhanced'})
                        self.assertEqual(len(rows(comparison/scope/'all_summary.csv')),len(algorithms)*2)
                        self.assertEqual(len(rows(comparison/scope/'test_metrics.csv')),min(count,4)*len(algorithms)*2)
                        self.assertEqual(len(rows(comparison/scope/'test_summary.csv')),len(algorithms)*2)
                        if file:self.assertEqual({row['file'] for row in scoped},{file})
                self.assertEqual(rows(comparison/'all_metrics.csv'),canonical)
                for path in protected[:4]:path.write_bytes(b'complete-benchmark-sentinel')
                pipeline.run_experiment(full)
                for path in protected[:4]:self.assertEqual(path.read_bytes(),sentinels[path])

    def test_fresh_statistics_have_unique_sources_and_actual_splits(self):
        import contextlib,csv,io,shutil,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app import pipeline
        from tests.test_notebook_protocol import tiny_dataset
        def rows(path):
            with path.open(encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);tiny_dataset(root)
            external=root/'external';external.mkdir()
            for suffix in ('.wav','.lab'):shutil.copyfile(root/'train'/('phone_F1'+suffix),external/('phone_F1'+suffix))
            with patch.multiple(pipeline,TRAIN_DIR=root/'train',TEST_DIR=root/'test',OUTPUT_DIR=root/'outputs'), \
                 patch.object(pipeline,'make_file_figure',return_value=None),patch.object(pipeline,'plot_gaussian_training'), \
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_experiment(parse_args(['--evaluate-all','--compare-endpoint-modes']))
                for mode in ('core','enhanced'):
                    statistics=rows(root/'outputs/endpoint_modes'/mode/'tables/dataset_statistics.csv')
                    self.assertEqual(len(statistics),8)
                    self.assertEqual(len({row['file'] for row in statistics}),8)
                    self.assertEqual([row['split'] for row in statistics].count('train'),4)
                    self.assertEqual([row['split'] for row in statistics].count('test'),4)
                pipeline.run_experiment(parse_args(['--file',str(external/'phone_F1.wav'),'--compare-endpoint-modes']))
                for mode in ('core','enhanced'):
                    scoped=rows(root/'outputs/endpoint_modes'/mode/'tables/all/single/phone_F1/dataset_statistics.csv')
                    self.assertEqual(len(scoped),1)
                    self.assertEqual((scoped[0]['file'],scoped[0]['split']),('phone_F1','external'))
                real_load=pipeline.load_audio_folder
                def load_with_external(folder):
                    records=real_load(folder)
                    return records+[pipeline.load_audio_file(external/'phone_F1.wav')] if folder==root/'test' else records
                with patch.object(pipeline,'load_audio_folder',side_effect=load_with_external):
                    pipeline.run_experiment(parse_args(['--compare-endpoint-modes']))
                for mode in ('core','enhanced'):
                    statistics=rows(root/'outputs/endpoint_modes'/mode/'tables/dataset_statistics.csv')
                    self.assertEqual(len(statistics),9)
                    self.assertEqual({row['split'] for row in statistics if row['file']=='phone_F1'},{'train','external'})

    def test_current_guide_metric_links_exist_and_filter_algorithm(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        for number,name in ((1,'BINARY_SEARCH'),(2,'HISTOGRAM'),(3,'GAUSSIAN')):
            guide=root/f'reports/TT{number}_{name}_GIAI_THICH.md'
            text=guide.read_text(encoding='utf-8-sig')
            target='../outputs/endpoint_modes/enhanced/tables/all/test_metrics.csv'
            self.assertIn(target,text)
            self.assertTrue((guide.parent/target).is_file())
            self.assertIn(f'algorithm=tt{number}',text)

    def test_current_handoff_links_describe_dual_mode_artifacts(self):
        import re
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        for relative in ('outputs/README.md','submission/README.md'):
            path=root/relative;text=path.read_text(encoding='utf-8-sig')
            current=text.split('## Lịch sử')[0]
            for marker in ('48','schema 3','schema 2','200','16','5 PNG','final_verification.json','KET_QUA_NGHIEM_THU_2026_10_07.md'):
                self.assertIn(marker,current,relative)
            for target in re.findall(r'\]\(([^)]+)\)',current):
                if target.endswith(('final_verification.json','KET_QUA_NGHIEM_THU_2026_10_07.md')):
                    self.assertTrue((path.parent/target).is_file(),target)
            self.assertNotIn('Chưa có model core riêng',current)
