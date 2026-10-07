"""Final-region W selection: balanced errors, valid counts, deterministic ties."""
import unittest
from app import pipeline
from app.config import TEST_DIR, TRAIN_DIR

def row(weight,name,error,gt=1,pred=1):
    return dict(W=weight,filename=name,final_region_mae_ms=error,
                ground_truth_region_count=gt,predicted_region_count=pred)

class FinalWeightSelectionTests(unittest.TestCase):
    def test_mixed_endpoint_policies_cannot_select_one_shared_weight(self):
        rows=[dict(row(1,'a',1),endpoint_mode='core'),
              dict(row(20,'a',0),endpoint_mode='enhanced')]
        with self.assertRaisesRegex(ValueError,'policy'):
            self.select(rows)

    def test_primary_mean_is_undefined_with_one_invalid_count(self):
        result=self.select([row(1,'a',None,pred=2),row(1,'b',10)])
        self.assertIsNone(result['selected_summary']['mean_MAE_ms'])
        self.assertEqual(result['selected_summary']['invalid_files'],1)
        self.assertEqual(result['selected_summary']['valid_files_mean_MAE_ms'],10)

    def select(self,rows,tie_preference=None):
        return pipeline.select_final_weight(rows,tie_preference)

    def test_all_four_optimal_uses_smallest_w_without_preferential_weight(self):
        rows=[row(w,n,e) for w in (1,5,20) for n,e in [('a',32.5),('b',7.5),('c',7.5),('d',7.5)]]
        result=self.select(rows)
        self.assertEqual(result['selected_W'],1)
        self.assertEqual(result['shared_optimal_W'],[1,5,20])
        self.assertIsNone(result['tie_preference_W'])
        self.assertIn('smallest W',result['selection_rule'])
        self.assertAlmostEqual(result['selected_summary']['mean_MAE_ms'],13.75)

    def test_conflicting_optima_minimize_worst_per_file_regret(self):
        rows=[row(w,n,e) for w,errors in [(1,[0,100]),(3,[20,20]),(5,[30,10])]
              for n,e in zip(('a','b'),errors)]
        result=self.select(rows)
        self.assertEqual(result['selected_W'],3)
        self.assertEqual(result['shared_optimal_W'],[])
        self.assertEqual(result['selected_summary']['max_regret_ms'],20)

    def test_extra_region_never_wins_by_lower_matched_mae(self):
        result=self.select([row(1,'a',None,pred=2),row(1,'b',0),row(3,'a',10),row(3,'b',10)])
        self.assertEqual(result['selected_W'],3)
        self.assertEqual(result['selected_summary']['invalid_files'],0)

    def test_large_errors_are_not_censored(self):
        result=self.select([row(1,'a',1000),row(1,'b',0),row(3,'a',100),row(3,'b',100)])
        self.assertEqual(result['selected_W'],3)
        self.assertEqual(result['summaries'][0]['mean_MAE_ms'],500)

    def test_input_order_does_not_change_tied_weight(self):
        rows=[row(w,n,10) for w in (1,20) for n in ('a','b')]
        self.assertEqual(self.select(rows)['selected_W'],self.select(list(reversed(rows)))['selected_W'])

    def test_duplicate_and_incomplete_candidates_rejected(self):
        with self.assertRaises(ValueError):self.select([row(1,'a',1),row(1,'a',2)])
        with self.assertRaises(ValueError):self.select([row(1,'a',1),row(1,'b',2),row(3,'a',3)])

    def test_tie_rule_cannot_be_overridden_to_prefer_w20(self):
        with self.assertRaisesRegex(ValueError,'tie|preference|smallest'):
            pipeline.select_final_weight([row(w,'a',10) for w in (1,5,20)],tie_preference_W=20)

    def test_sweep_rejects_test_and_external_records(self):
        for split in ('test','external'):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError,'TRAIN|train'):
                pipeline.sweep_final_weights([dict(name='holdout',split=split)], {},
                                             lambda *_: self.fail('predictor must not run'),
                                             candidates=(1,20))
        spoofed=dict(name='phone_F2',split='train',wav_path=str(TEST_DIR/'phone_F2.wav'))
        with self.assertRaisesRegex(ValueError,'TRAIN|train'):
            pipeline.sweep_final_weights([spoofed], {},
                                         lambda *_: self.fail('predictor must not run'),
                                         candidates=(1,20))

    def test_selected_w_is_recomputable_from_train_sweep_manifest(self):
        records=[dict(name='train_a',split='train',wav_path=str(TRAIN_DIR/'train_a.wav')),
                 dict(name='train_b',split='train',wav_path=str(TRAIN_DIR/'train_b.wav'))]
        def predictor(_algorithm,record,params):
            error=0. if params['W']==1 else 0.
            return dict(metrics=dict(ground_truth_region_count=1,predicted_region_count=1,
                                     mae_ms=error,status='ok',start_error_ms=error,end_error_ms=error),
                        diagnostic=dict(endpoint_mode='core',geometry='union of active frame supports',
                                        minimum_speech_ms=0.,minimum_silence_ms=200.,padding_stage='excluded from core FINAL',
                                        energy_threshold=.1,low_ste_threshold=None,high_ste_threshold=None,
                                        raw_speech_frames=3,candidate_regions=[(0.,.1)]),
                        final_regions=[(.0,.1)])
        selection,rows=pipeline.sweep_final_weights(records,dict(endpoint_mode='core'),predictor,candidates=(20,1))
        self.assertEqual(selection['evaluated_files'],['train_a','train_b'])
        self.assertEqual(selection['selection_set'],'train')
        self.assertEqual(selection['selected_W'],1.)
        self.assertEqual(sorted({row['W'] for row in rows}),[1.,20.])

    def test_sweep_rejects_train_claim_without_wav_provenance(self):
        with self.assertRaisesRegex(ValueError,'paths must belong to TRAIN'):
            pipeline.sweep_final_weights([dict(name='train_a',split='train')], {},
                                         lambda *_: self.fail('predictor must not run'),
                                         candidates=(1,))

    def test_real_four_train_file_sweep_scores_final_regions(self):
        self.assertTrue(callable(getattr(pipeline,'fit_training_model',None)))
        records=pipeline.prepare_records(pipeline.load_audio_folder(TRAIN_DIR),True)
        model,rows=pipeline.fit_training_model('tt2',records)
        result=model['W_selection_train']
        self.assertEqual(result['evaluation_protocol'],'train_final_calibration')
        self.assertEqual(result['selection_set'],'train')
        self.assertEqual(set(result['evaluated_files']),{r['name'] for r in records})
        self.assertEqual(len(rows),200)
        self.assertEqual(model['W'],result['selected_W'])
        for row_ in rows:
            record=next(r for r in records if r['name']==row_['filename'])
            score=pipeline.predict_and_score('tt2',record,dict(model,W=row_['W']))
            self.assertEqual(row_['final_region_mae_ms'],score['metrics']['mae_ms'])
            self.assertEqual(row_['predicted_region_count'],len(score['final_regions']))

if __name__=='__main__':unittest.main()
