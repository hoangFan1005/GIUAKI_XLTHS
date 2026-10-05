import json
import unittest
from app.pipeline import load_audio_file, prepare_records, predict_and_score
from app.config import PROJECT_ROOT, TEST_DIR
from core.postprocess import mask_segments
from core.metrics import frame_labels
from app.pipeline import summarize
from app.cli import parse_args

class PipelineEndpointRegression(unittest.TestCase):
    def test_noise_islands_are_not_final_regions(self):
        model=json.loads((PROJECT_ROOT/'outputs/models/tt1.json').read_text())
        record=prepare_records([load_audio_file(TEST_DIR/'phone_F2.wav')])[0]
        result=predict_and_score('tt1',record,model)
        regions=[s for s in mask_segments(result['mask'],record['features']['starts'],record['duration'],record['features']['ends']) if s[2]]
        self.assertEqual(len(regions),1,'Weak noise islands must not become final speech regions')

    def test_pipeline_scores_two_regions_not_global_envelope(self):
        values=[0.]*160
        values[10:35]=[.8]*25;values[45:70]=[.8]*25;values[105:135]=[.8]*30
        starts=[i*.01 for i in range(160)];ends=[s+.025 for s in starts]
        intervals=[(0.,.1,'sil'),(.1,.715,'v'),(.715,1.05,'sil'),(1.05,1.365,'uv'),(1.365,1.62,'sil')]
        features=dict(ste_norm=values,starts=starts,ends=ends)
        record=dict(name='synthetic_two_utterances',features=features,duration=1.62,intervals=intervals,
                    labels=frame_labels([(s+e)/2 for s,e in zip(starts,ends)],intervals))
        model=dict(threshold=.1,endpoint_noise=dict(noise_q95=.02,noise_upper=.15))
        result=predict_and_score('tt1',record,model)
        self.assertEqual(len(result['final_regions']),2)
        self.assertEqual(result['metrics']['ground_truth_region_count'],2)
        self.assertAlmostEqual(result['metrics']['mae_ms'],0.)
        self.assertEqual(result['predicted_boundaries'],[b for region in result['final_regions'] for b in region])
        row=dict(file=record['name'],algorithm='tt1',**result['metrics'])
        self.assertAlmostEqual(summarize([row])[0]['pooled_endpoint_rmse_ms'],0.)

    def test_all_dataset_cli_is_headless_and_keeps_fixed_algorithm(self):
        args=parse_args(['--evaluate-all'],'tt2')
        self.assertTrue(args.no_show);self.assertEqual(args.algorithm,'tt2')

if __name__=='__main__':unittest.main()
