"""Behavioral regressions for final regions, train calibration and region scoring."""
import unittest
from core.endpoints import hysteresis_regions, fit_noise_floor, endpoint_thresholds
from core.metrics import ground_truth_regions, region_endpoint_metrics

def timing(n):
    return [i*.01 for i in range(n)], [i*.01+.025 for i in range(n)]

class EndpointTests(unittest.TestCase):
    def assertRegions(self,actual,expected):
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            self.assertAlmostEqual(a[0],b[0]);self.assertAlmostEqual(a[1],b[1])
    def test_low_noise_islands_cannot_confirm_speech(self):
        values=[.02]*120
        values[40:80]=[.8]*40
        starts,ends=timing(len(values))
        regions=hysteresis_regions(values,starts,ends,1.22,.1,.5,.2,.1)
        self.assertRegions(regions,[(.4,.815)])

    def test_high_and_low_both_change_real_detection(self):
        starts,ends=timing(80)
        values=[0.]*10+[.2]*10+[.8]*20+[.2]*10+[0.]*30
        a=hysteresis_regions(values,starts,ends,.82,.1,.5,.2,.1)
        b=hysteresis_regions(values,starts,ends,.82,.3,.5,.2,.1)
        c=hysteresis_regions(values,starts,ends,.82,.1,.9,.2,.1)
        self.assertRegions(a,[(.1,.515)])
        self.assertRegions(b,[(.2,.415)])
        self.assertEqual(c,[])

    def test_two_real_regions_and_short_word_pause(self):
        values=[0.]*160
        values[10:35]=[.8]*25;values[45:70]=[.8]*25
        values[105:135]=[.8]*30
        s,e=timing(len(values))
        self.assertRegions(hysteresis_regions(values,s,e,1.62,.1,.5,.2,.1),[(.1,.715),(1.05,1.365)])

    def test_long_gap_is_checked_before_new_high_sample(self):
        # Last support END=.1, next START=.3: exactly200ms must split.
        s=[i*.025 for i in range(20)];e=[v+.025 for v in s]
        v=[.8]*4+[0.]*8+[.8]*4+[0.]*4
        self.assertRegions(hysteresis_regions(v,s,e,.5,.1,.5,.2,.1),[(0.,.1),(.3,.4)])

    def test_short_region_removed_and_audio_edges_retained(self):
        s,e=timing(100);v=[0.]*100;v[:15]=[.8]*15;v[45]=.9;v[85:]=[.8]*15
        regions=hysteresis_regions(v,s,e,1.02,.1,.5,.2,.1)
        self.assertRegions(regions,[(0.,.165),(.85,1.015)])

    def test_zero_audio_never_speech(self):
        s,e=timing(100)
        self.assertRegions(hysteresis_regions([0.]*100,s,e,1.02,1e-12,2e-12,.2,.1),[])

    def test_confirmation_seed_required_but_low_speech_can_continue(self):
        s,e=timing(70);v=[0.]*10+[.8]*30+[0.]*30
        self.assertRegions(hysteresis_regions(v,s,e,.72,.1,.5,.2,.1,seed_mask=[0]*70),[])
        seed=[0]*70;seed[20]=1
        self.assertRegions(hysteresis_regions(v,s,e,.72,.1,.5,.2,.1,seed_mask=seed),[(.1,.415)])

    def test_noise_calibration_only_uses_training_silence(self):
        records=[dict(features={'ste_norm':[.1,.2,.9]},labels=[0,0,1])]
        model=fit_noise_floor(records)
        changed=fit_noise_floor([dict(features={'ste_norm':[.1,.2,100.]},labels=[0,0,1])])
        self.assertEqual(model,changed)
        self.assertAlmostEqual(model['noise_mean'],.15)
        self.assertAlmostEqual(model['noise_std'],.05)

    def test_histogram_base_is_high_confirmation_not_low_hold(self):
        noise={'noise_upper':.003,'noise_q95':.002}
        lo,hi=endpoint_thresholds(.04,noise,histogram=True)
        self.assertEqual(lo,.003);self.assertEqual(hi,.04)

    def test_contiguous_v_uv_one_gt_and_real_silence_two(self):
        gt=ground_truth_regions([(0.,.1,'sil'),(.1,.3,'v'),(.3,.5,'uv'),(.5,.8,'sil'),(.8,1.,'v')])
        self.assertEqual(gt,[(.1,.5),(.8,1.)])

    def test_multi_region_mae_does_not_collapse_envelope(self):
        z=region_endpoint_metrics([(0.,1.),(2.,3.)],[(.1,1.1),(2.1,3.2)])
        self.assertAlmostEqual(z['mae_ms'],125.)
        self.assertEqual(z['matched_region_count'],2)

    def test_extra_region_cannot_lower_primary_mae_by_matching_only_good(self):
        z=region_endpoint_metrics([(1.,2.)],[(.1,.2),(1.01,2.01)])
        self.assertIsNone(z['mae_ms'])
        self.assertEqual(z['extra_region_count'],1)
        self.assertAlmostEqual(z['matched_boundary_mae_ms'],10.)
        self.assertEqual(z['region_pairs'],[(0,1)])

    def test_large_single_region_errors_not_censored(self):
        z=region_endpoint_metrics([(1.,2.)],[(0.,4.)])
        self.assertEqual(z['mae_ms'],1500.)

    def test_large_ordered_multi_region_errors_not_censored(self):
        z=region_endpoint_metrics([(1.,2.),(3.,4.)],[(5.,6.),(7.,8.)])
        self.assertEqual(z['region_pairs'],[(0,0),(1,1)])
        self.assertEqual(z['mae_ms'],4000.)

    def test_missing_or_no_speech_explicit(self):
        self.assertIsNone(region_endpoint_metrics([(1.,2.)],[])['mae_ms'])
        self.assertEqual(region_endpoint_metrics([],[])['status'],'both_no_speech')

if __name__=='__main__':unittest.main()

