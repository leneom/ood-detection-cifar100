import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ash_diagnostics_core import assemble,describe,classification,transitions,COMPONENTS
class ASHDiagnosticMathTests(unittest.TestCase):
 def test_bias_aware_scaled_logits_can_change_argmax(self):
  b=np.array([10.,0.]);zp=np.array([[1.,0.]]);c=3.
  zs=c*zp+(1-c)*b
  self.assertEqual(zp.argmax(1)[0],0);self.assertEqual(zs.argmax(1)[0],1)
  self.assertEqual((c*zp).argmax(1)[0],0)
 def test_gains_losses_preserve_original_correctness_denominator(self):
  labels=np.array([0,1,2,3]);original=np.array([0,0,2,0]);shaped=np.array([0,1,0,1])
  r,c=classification(shaped,original,labels)
  np.testing.assert_array_equal(c,[0,1,2,3]);self.assertEqual(r['accuracy_change'],0);self.assertEqual(r['argmax_changes'],3)
 def test_acceptance_change_and_risk_have_distinct_denominators(self):
  a=np.array([1,1,0,0],bool);r=np.array([1,0,1,0],bool);c=np.array([1,0,1,0],bool)
  out,codes=transitions(a,r,c)
  np.testing.assert_array_equal(codes,[0,1,2,3]);self.assertEqual(out['acceptance_change'],0);self.assertEqual(out['retained_error_risk_change'],.5)
  self.assertEqual(out['incorrect']['n'],2)
 def test_no_acceptance_or_zero_variance_is_undefined(self):
  a=np.array([False,False]);out,_=transitions(a,a,np.array([True,True]));self.assertIsNone(out['addition_retained_error_risk']);self.assertIsNone(out['incorrect']['state_fractions']['both_accept'])
  arrays={k:np.ones(2) for k in COMPONENTS};d=describe(arrays,0.)
  self.assertIsNone(d['joint']['ash_p_energy']['pearson_fraction_delta'])
 def test_alignment_and_boolean_types_rejected(self):
  with self.assertRaises(ValueError):transitions(np.array([1,0]),np.array([True,False]))
  with self.assertRaises(ValueError):classification(np.array([0]),np.array([0,1]),np.array([0]))
if __name__=='__main__':unittest.main()
