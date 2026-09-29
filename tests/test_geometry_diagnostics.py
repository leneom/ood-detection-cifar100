import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from geometry_diagnostics_core import describe, transitions, mean_difference, STATE_NAMES


class GeometryDiagnosticsTests(unittest.TestCase):
    def components(self, v, e):
        v,e = np.array(v,float),np.array(e,float)
        return dict(residual=np.ones(len(v)),virtual_logit=v,affine_energy=e,vim=v+e,feature_norm=np.ones(len(v)))

    def test_anticorrelation_cancels_variance_and_signed_mean_shifts(self):
        a = describe(self.components([1,2,3,4],[-1,-2,-3,-4]))
        self.assertEqual(a['joint']['pearson_virtual_energy'], -1.)
        self.assertEqual(a['joint']['vim_variance'], 0.)
        b = describe(self.components([11,12,13,14],[-9,-10,-11,-12]))
        shift = mean_difference(b,a)
        self.assertEqual(shift['virtual_logit'],10.)
        self.assertEqual(shift['affine_energy'],-8.)
        self.assertEqual(shift['vim'],2.)

    def test_zero_variance_correlation_is_undefined_and_wrong_identity_rejected(self):
        x=self.components([2,2,2],[1,2,3])
        self.assertIsNone(describe(x)['joint']['pearson_virtual_energy'])
        x['vim'][1]+=1.
        with self.assertRaisesRegex(ValueError,'identity'):
            describe(x)

    def test_all_transition_states_keep_correctness_denominators(self):
        a=np.array([True,True,False,False])
        r=np.array([True,False,True,False])
        result,codes=transitions(a,r,np.array([True,False,False,False]))
        np.testing.assert_array_equal(codes,[0,1,2,3])
        self.assertEqual(list(result['state_counts'].values()),[1,1,1,1])
        self.assertEqual(result['correct']['n'],1)
        self.assertEqual(result['incorrect']['n'],3)
        self.assertEqual(result['incorrect']['state_fractions'][STATE_NAMES[1]],1/3)
        self.assertEqual(result['acceptance_change'],0.)

    def test_zero_correct_denominator_and_nonboolean_input(self):
        a=np.array([True,False])
        result,_=transitions(a,a,np.array([False,False]))
        self.assertEqual(result['correct']['n'],0)
        self.assertTrue(all(v is None for v in result['correct']['state_fractions'].values()))
        with self.assertRaisesRegex(ValueError,'boolean'):
            transitions(np.array([1,0]),a)


if __name__ == '__main__':
    unittest.main()
