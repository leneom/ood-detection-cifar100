import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ash_core import shape_ash,ash_scores


class ASHMathTests(unittest.TestCase):
    def test_nearest_even_rounding_and_pruning(self):
        f=np.arange(1.,11.)[None,:]
        h,s=shape_ash(f)
        np.testing.assert_array_equal(s['selected_indices'],[[9,8,7,6]])
        self.assertEqual(s['original_sum'][0],55.)
        self.assertEqual(s['retained_sum'][0],34.)
        np.testing.assert_array_equal(h['ash_p_energy'],[[0,0,0,0,0,0,7,8,9,10]])
        self.assertEqual(h['ash_b_energy'].sum(),55.)
        np.testing.assert_allclose(h['ash_s_energy'],h['ash_p_energy']*np.exp(55/34))

    def test_equal_values_choose_original_coordinate_order(self):
        f=np.ones((1,10))
        h,s=shape_ash(f)
        np.testing.assert_array_equal(s['selected_indices'],[[0,1,2,3]])
        self.assertEqual(s['cutoff_equal_count'][0],10)
        self.assertEqual(s['cutoff_selected_equal_count'][0],4)
        np.testing.assert_array_equal(f,np.ones((1,10)))

    def test_exponential_is_not_linear_scaling(self):
        h,s=shape_ash(np.ones((1,10)))
        self.assertAlmostEqual(h['ash_s_energy'][0,0],np.exp(2.5))
        self.assertNotEqual(h['ash_s_energy'][0,0],2.5)
        self.assertAlmostEqual(h['ash_b_energy'][0,0],2.5)

    def test_score_sign_and_original_bias(self):
        f=np.arange(1.,11.)[None,:]
        w=np.vstack([np.ones(10),-np.ones(10)])
        b=np.array([2.,-3.])
        scores,s=ash_scores(f,w,b)
        np.testing.assert_allclose(s['ash_p_energy_logits'],[[36.,-37.]])
        self.assertAlmostEqual(scores['ash_p_energy'][0],-np.log(np.exp(36.)+np.exp(-37.)))

    def test_invalid_rows_and_settings_fail(self):
        for f in [np.zeros((1,10)),np.ones((0,10)),np.ones((1,10))*np.nan,-np.ones((1,10))]:
            with self.assertRaises(ValueError): shape_ash(f)
        with self.assertRaises(ValueError): shape_ash(np.ones((1,10)),90)
