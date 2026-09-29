import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from bootstrap_core import RankingPlan,operational_counts
class WeightedBootstrapMathTests(unittest.TestCase):
 def test_weighted_half_credit_ties_and_grouped_average_precision(self):
  a=RankingPlan([0.,1.],[1.,2.]).evaluate(np.array([2,1]),np.array([1,2]))
  np.testing.assert_allclose(a,[8.5/9,11/12,11/12,1/3,1.,1.],atol=1e-14)
 def test_all_ties_preserve_weighted_prevalence(self):
  a=RankingPlan([1.,1.],[1.,1.]).evaluate(np.array([2,1]),np.array([1,1]))
  np.testing.assert_allclose(a,[.5,.4,.6,1.,1.,1.])
 def test_zero_mass_extremes_do_not_change_threshold(self):
  a=RankingPlan([-99.,0.,1.,99.],[0.,2.]).evaluate(np.array([0,2,1,0]),np.array([1,2]))
  b=RankingPlan([0.,1.],[0.,2.]).evaluate(np.array([2,1]),np.array([1,2]));np.testing.assert_array_equal(a,b)
 def test_weighted_operating_denominators_and_undefined_risk(self):
  masks={'id_accepted':np.array([[[1,1,1],[0,0,0]]],bool),'x_accepted':np.array([[[1,0],[0,0]]],bool)}
  a=operational_counts(np.array([2,0,1]),{'x':np.array([1,2])},masks,np.array([1,0,0],bool),['x'])
  np.testing.assert_array_equal(a[0],[[3,0]]);np.testing.assert_array_equal(a[1],[[1,0]]);self.assertAlmostEqual(a[4][0,0],1/3);self.assertTrue(np.isnan(a[4][0,1]));self.assertAlmostEqual(a[5][0,0,0],1/3)
 def test_invalid_weights_rejected(self):
  plan=RankingPlan([0.,1.],[2.])
  for w in [np.array([-1,2]),np.array([0,0]),np.array([1.,1.]),np.array([1])]:
   with self.assertRaises(ValueError):plan.evaluate(w,np.array([1]))
if __name__=='__main__':unittest.main()
