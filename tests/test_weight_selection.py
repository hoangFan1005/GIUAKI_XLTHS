"""Final-region W selection: balanced errors, valid counts, deterministic ties."""
import unittest
from app import pipeline
from app.config import TRAIN_DIR

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

    def select(self,rows,tie_preference=20):
        return pipeline.select_final_weight(rows,tie_preference)

    def test_all_four_optimal_uses_declared_tie_preference(self):
        rows=[row(w,n,e) for w in (1,5,20) for n,e in [('a',32.5),('b',7.5),('c',7.5),('d',7.5)]]
        result=self.select(rows)
        self.assertEqual(result['selected_W'],20)
        self.assertEqual(result['shared_optimal_W'],[1,5,20])
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

    def test_tie_preference_is_not_the_model_candidate_frame_weight(self):
        result=pipeline.select_final_weight([row(w,'a',10) for w in (1,5,20)],tie_preference_W=20)
        self.assertEqual(result['selected_W'],20)
        self.assertEqual(result['tie_preference_W'],20)
        self.assertNotIn('previous_W',result)

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
