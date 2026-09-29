import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from dependence_core import moments,decomposition,json_finite

class DependenceTests(unittest.TestCase):
    def test_large_exact_constant_has_zero_variance_and_undefined_correlation(self):
        m=moments(np.full((7,2),1e12+.125))
        np.testing.assert_array_equal(m['covariance'],np.zeros((2,2)))
        self.assertFalse(m['correlation_defined'].any());self.assertEqual(json_finite(m['correlation']),[[None,None],[None,None]])
    def test_identical_three_components_sd_ratio_sqrt_three(self):
        x=np.arange(1,6,dtype=float);m,d=decomposition(np.column_stack([x,x,x]))
        self.assertAlmostEqual(d['observed_mean_variance'],2.5);self.assertAlmostEqual(d['diagonal_contribution'],2.5/3)
        self.assertAlmostEqual(d['cross_covariance_contribution'],5/3);self.assertAlmostEqual(d['sd_ratio_to_diagonal_only'],np.sqrt(3))
        np.testing.assert_allclose(m['correlation'],np.ones((3,3)))
    def test_negative_covariance_is_retained(self):
        x=np.arange(1,6,dtype=float);m,d=decomposition(np.column_stack([x,-2*x,np.full(5,9.)]))
        self.assertEqual(m['covariance'][0,1],-5);self.assertAlmostEqual(d['observed_mean_variance'],2.5/9)
        self.assertAlmostEqual(d['cross_covariance_contribution'],-10/9);self.assertAlmostEqual(d['sd_ratio_to_diagonal_only'],np.sqrt(1/5))
        self.assertIsNone(json_finite(m['correlation'])[2][2])
    def test_four_shared_components_macro_sd_ratio_two(self):
        _,d=decomposition(np.tile(np.arange(8,dtype=float)[:,None],(1,4)))
        self.assertAlmostEqual(d['sd_ratio_to_diagonal_only'],2);self.assertEqual(d['components'],4)
        self.assertAlmostEqual(d['signed_identity_residual'],0)
    def test_nonfinite_column_never_uses_complete_cases(self):
        x=np.array([[1,1],[2,np.nan],[3,3]],dtype=float);m,d=decomposition(x)
        np.testing.assert_array_equal(m['finite_counts'],[3,2]);self.assertEqual(m['covariance'][0,0],1)
        self.assertTrue(np.isnan(m['covariance'][1]).all());self.assertFalse(d['defined']);self.assertIsNone(d['observed_mean_variance'])
        self.assertEqual(json_finite(m['means']),[2,None])
    def test_invalid_draw_shape_rejected(self):
        for x in [np.ones((1,3)),np.ones(3),np.ones((4,0))]:
            with self.assertRaises(ValueError):moments(x)

if __name__=='__main__':unittest.main()
