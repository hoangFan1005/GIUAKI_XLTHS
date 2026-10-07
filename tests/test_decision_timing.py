"""Independent midpoint and duration-rule oracles for diagnostic decision cells."""
import importlib
import unittest

class DecisionTimingTests(unittest.TestCase):
    def helper(self):
        self.assertIsNotNone(importlib.util.find_spec('core.decision_timing'), 'Decision-cell helper is required')
        return importlib.import_module('core.decision_timing').decision_cells

    def test_midpoints_cover_audio_and_extend_unanalysed_tail(self):
        cells = self.helper()
        for fs, centers, middle in ((16000, [.0125,.0225,.0325], [.0175,.0275]),
                                    (44100, [551/44100,992/44100,1433/44100], [771.5/44100,1212.5/44100])):
            with self.subTest(fs=fs):
                starts, ends = cells(dict(centers=centers), .049)
                self.assertEqual(starts, [0.,*middle])
                self.assertEqual(ends, [*middle,.049])
                self.assertAlmostEqual(sum(e-s for s,e in zip(starts,ends)), .049)

    def test_empty_one_frame_and_invalid_centers(self):
        cells = self.helper()
        self.assertEqual(cells(dict(centers=[]), 0.), ([],[]))
        self.assertEqual(cells(dict(centers=[.0125]), .03), ([0.],[.03]))
        for centers,duration in (([.02,.01],.1),([.01,.01],.1),([float('nan')],.1),([-.01],.1),([.2],.1),([.01],0.),([],float('inf'))):
            with self.subTest(centers=centers), self.assertRaises(ValueError):
                cells(dict(centers=centers),duration)

    def test_cell_gap_190_merges_exact_200_and_210_survive(self):
        from core.endpoints import hysteresis_regions, regions_to_mask
        cells = self.helper()
        for fs in (16000,44100):
            offset = round(fs*.025)/(2*fs)-.0125
            for gap,count in ((.19,1),(.2,2),(.21,2)):
                # Literal midpoints yield [.1,.2] and [.2+gap,.3+gap].
                centers = [x+offset for x in [.05,.15,.25,.15+2*gap,.45+2*gap]]
                starts,ends = cells(dict(centers=centers), 1.)
                values=[0.,.8,0.,.8,0.]
                result=hysteresis_regions(values,starts,ends,1.,.1,.5,.2,0.)
                expected=[(.1+offset,.3+2*gap+offset)] if count==1 else [(.1+offset,.2+offset),(.2+gap+offset,.3+2*gap+offset)]
                self.assertEqual(len(result),count)
                for pair,want in zip(result,expected):
                    for x,y in zip(pair,want):self.assertAlmostEqual(x,y)
                self.assertEqual(regions_to_mask(result,starts,ends),[0,1,1,1,0] if count==1 else [0,1,0,1,0])




class EstimatedCellSpanTests(unittest.TestCase):
    def test_estimated_95_removed_exact_100_and_105_retained(self):
        from core.decision_timing import decision_cells
        from core.endpoints import hysteresis_regions
        for span,expected in ((.095,[]),(.100,[(.1,.2)]),(.105,[(.1,.205)])):
            centers=[.05,.15,.05+2*span,.35+2*span]
            starts,ends=decision_cells(dict(centers=centers),.8)
            result=hysteresis_regions([0.,.8,0.,0.],starts,ends,.8,.1,.5,.2,.1)
            self.assertEqual(len(result),len(expected))
            for pair,want in zip(result,expected):
                for a,b in zip(pair,want):self.assertAlmostEqual(a,b)

    def test_speech_at_edges_uses_0_duration_and_last_cell_tail(self):
        from core.decision_timing import decision_cells
        from core.endpoints import hysteresis_regions,regions_to_mask
        for centers in ([.0125,.0225,.0325],[551/44100,992/44100,1433/44100]):
            starts,ends=decision_cells(dict(centers=centers),.049)
            for values,want_mask in (([.8,0.,0.],[1,0,0]),([0.,0.,.8],[0,0,1])):
                result=hysteresis_regions(values,starts,ends,.049,.1,.5,.2,0.)
                self.assertEqual(regions_to_mask(result,starts,ends),want_mask)
                if values[0]:self.assertEqual(result[0][0],0.)
                else:self.assertEqual(result[0][1],.049)

if __name__=='__main__':unittest.main()
