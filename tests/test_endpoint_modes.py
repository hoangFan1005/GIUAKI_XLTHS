"""Native inequalities and policy behavior, with hand-derived support fixtures."""
import copy
import unittest

from app import pipeline
from core.features import compute_features


NOISE = dict(noise_q95=.01, noise_upper=.03, high_multiplier=1.5)


def features(values, starts=None, ends=None):
    starts = starts if starts is not None else [i*.01 for i in range(len(values))]
    ends = ends if ends is not None else [s+.025 for s in starts]
    return dict(ste_norm=list(values), energy=[v*100 for v in values], starts=starts, ends=ends)


class EndpointModesTests(unittest.TestCase):
    def decision(self, algorithm, f, params):
        self.assertTrue(callable(getattr(pipeline, 'algorithm_decision', None)),
                        'Native classifier API must exist without endpoint policy dependencies')
        return pipeline.algorithm_decision(algorithm, f, params)

    def test_native_tt1_and_tt3_include_equality_and_support_low_direction(self):
        # Changing >= / <= to strict comparisons loses the equality frame.
        for a, direction, want in (('tt1','high',[0,1,1]), ('tt3','high',[0,1,1]),
                                   ('tt3','low',[1,1,0])):
            with self.subTest(algorithm=a, direction=direction):
                mask, d = self.decision(a, dict(ste_norm=[.2,.5,.8]),
                                        dict(threshold=.5,speech_direction=direction))
                self.assertEqual(mask, want)
                self.assertEqual(d['native_threshold'], .5)
                self.assertEqual(d['native_threshold_units'], 'normalized STE')

    def test_tt2_native_energy_is_strict_and_excludes_candidate_padding(self):
        mask, d = self.decision('tt2', dict(energy=[0.,2.,4.]),
                               dict(bins=1,smooth_radius=0,W=5.,padding_frames=25))
        self.assertEqual(mask, [0,0,1])  # one peak -> midpoint 2, equality is silence
        self.assertEqual(d['native_threshold'], 2.)
        self.assertEqual(d['native_threshold_units'], 'mean squared sample amplitude')

    def test_context_native_decision_uses_normalized_ste(self):
        mask, d = self.decision('tt2-context', dict(ste_norm=[0.,.5,1.]),
                               dict(variant='context',bins=1,smooth_radius=0,W=5.))
        self.assertEqual(mask, [0,0,1])
        self.assertEqual(d['native_threshold_units'], 'normalized STE')

    def test_core_support_gap_threshold_keeps_exact_200ms(self):
        # Wrong gap comparator or using frame count would merge the 200ms case.
        for gap, want in ((.19,[(.1,.605)]),(.2,[(.1,.325),(.525,.615)]),
                          (.21,[(.1,.325),(.535,.625)]),(.25,[(.1,.325),(.575,.665)])):
            with self.subTest(gap=gap):
                right=.325+gap
                f=features([0,1,0,1,0], [0,.1,.35,right,.8], [.025,.325,.375,right+.09,.825])
                before=copy.deepcopy(f)
                r=pipeline.detect_regions('tt1',f,.825,dict(threshold=.5,endpoint_mode='core'))
                self.assertEqual(len(r['final_regions']),len(want))
                for actual,expected in zip(r['final_regions'],want):
                    for x,y in zip(actual,expected): self.assertAlmostEqual(x,y)
                self.assertEqual(f,before)

    def test_core_retains_80ms_support_without_noise_and_labels_metadata(self):
        f=features([0,1,0],[0,.1,.3],[.025,.18,.325])
        params=dict(threshold=.5,endpoint_mode='core')
        before=copy.deepcopy(params)
        r=pipeline.detect_regions('tt1',f,.325,params)
        self.assertEqual(r['final_regions'],[(.1,.18)])
        self.assertEqual(r['predicted_boundaries'],[.1,.18])
        for key in ('low_ste_threshold','high_ste_threshold','endpoint_noise'):
            self.assertIsNone(r['diagnostic'][key])
        self.assertEqual(r['diagnostic']['endpoint_mode'],'core')
        self.assertEqual(r['diagnostic']['minimum_speech_ms'],0.)
        self.assertEqual(r['diagnostic']['final_padding_ms'],0.)
        self.assertEqual(params,before)
        enhanced=pipeline.detect_regions('tt1',f,.325,dict(params,endpoint_mode='enhanced',endpoint_noise=NOISE))
        self.assertEqual(enhanced['final_regions'],[])

    def test_core_low_gaussian_keeps_native_inequality(self):
        r=pipeline.detect_regions('tt3', features([.8,.5,.2,.8]), .055,
                                  dict(threshold=.5,speech_direction='low',endpoint_mode='core'))
        self.assertEqual(r['final_regions'],[(.01,.045)])

    def test_unknown_mode_and_gaussian_direction_are_rejected(self):
        f=features([0.,1.])
        with self.assertRaisesRegex(ValueError,'endpoint_mode'):
            pipeline.detect_regions('tt1',f,.035,dict(threshold=.5,endpoint_mode='typo',endpoint_noise=NOISE))
        for mode,direction in (('core','unknown'),('enhanced','unknown'),('enhanced','low')):
            with self.subTest(mode=mode,direction=direction), self.assertRaisesRegex(ValueError,'direction'):
                pipeline.detect_regions('tt3',f,.035,dict(threshold=.5,endpoint_mode=mode,
                    speech_direction=direction,endpoint_noise=NOISE))

    def test_legacy_mode_resolves_enhanced_and_noise_remains_required(self):
        f=features([0.]*10+[1.]*20+[0.]*30)
        p=dict(threshold=.5,endpoint_noise=NOISE)
        self.assertEqual(pipeline.detect_regions('tt1',f,.615,p),
                         pipeline.detect_regions('tt1',f,.615,dict(p,endpoint_mode='enhanced')))
        with self.assertRaisesRegex(ValueError,'endpoint_noise'):
            pipeline.detect_regions('tt1',f,.615,dict(threshold=.5))

    def test_valid_train_models_do_not_detect_zero_waveform_or_empty_features(self):
        train=[dict(features=dict(ste_norm=[.001,.002,.7,.9],energy=[.1,.2,70.,90.]), labels=[0,0,1,1])]
        models={'tt1':pipeline.tt1.fit(train), 'tt3':pipeline.tt3.fit(train),
                'tt2':dict(bins=64,smooth_radius=2,W=20.,padding_frames=25)}
        for fs in (16000,44100):
            f=compute_features([0.]*round(fs*.8),fs)
            for a,p in models.items():
                with self.subTest(fs=fs,algorithm=a):
                    self.assertEqual(pipeline.detect_regions(a,f,.8,dict(p,endpoint_mode='core'))['final_regions'],[])
                    self.assertEqual(pipeline.detect_regions(a,features([]),0.,dict(p,endpoint_mode='core'))['final_regions'],[])


if __name__ == '__main__':
    unittest.main()
