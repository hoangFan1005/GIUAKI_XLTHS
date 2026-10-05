"""Final-region W selection: balanced errors, valid counts, deterministic ties."""
import unittest
import json
from app import pipeline
from app.config import TEST_DIR,OUTPUT_DIR

def row(weight,name,error,gt=1,pred=1):
    return dict(W=weight,filename=name,boundary_MAE_ms=error,
                ground_truth_region_count=gt,predicted_region_count=pred)

class FinalWeightSelectionTests(unittest.TestCase):
    def select(self,rows,current=20):
        return pipeline.select_final_weight(rows,current)

    def test_all_four_optimal_keeps_current_weight(self):
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

    def test_real_four_file_sweep_uses_weight_but_keeps_final_boundaries(self):
        model=json.loads((OUTPUT_DIR/'models/tt2.json').read_text(encoding='utf-8'))
        before=json.dumps(model,sort_keys=True)
        records=pipeline.prepare_records(pipeline.load_audio_folder(TEST_DIR),True)
        result,rows=pipeline.sweep_final_weights(records,model,pipeline.predict_and_score,candidates=(1,5,20))
        self.assertEqual(result['selected_W'],20)
        self.assertEqual(result['shared_optimal_W'],[1,5,20])
        self.assertEqual(result['evaluation_protocol'],'test_tuned_not_independent')
        self.assertEqual(len(rows),12)
        for record in records:
            same=[r for r in rows if r['filename']==record['name']]
            self.assertEqual(len(set(r['final_regions'] for r in same)),1)
            self.assertGreater(len(set(r['energy_threshold'] for r in same)),1)
        self.assertEqual(before,json.dumps(model,sort_keys=True))

if __name__=='__main__':unittest.main()
