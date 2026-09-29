import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rmd_core import fit_rmd,rmd_scores,nonnegative_distances


class RMDTests(unittest.TestCase):
    def data(self):
        return np.array([[-3,-1],[-2,0],[-1,1],[1,1],[2,0],[3,-1]],float),np.array([0,0,0,1,1,1])

    def test_population_covariance_and_own_trace_regularization(self):
        f,y=self.data();meta,s=fit_rmd(f,y)
        within=f-s['class_means'][y];bg=f-f.mean(0)
        np.testing.assert_allclose(s['within_covariance_raw'],within.T@within/6)
        np.testing.assert_allclose(s['background_covariance_raw'],bg.T@bg/6)
        between=s['class_means']-s['background_mean']
        np.testing.assert_allclose(s['background_covariance_raw'],s['within_covariance_raw']+between.T@between/2)
        for name in ['within','background']:
            c=s[name+'_covariance_raw'];expected=.99*c+.01*np.trace(c)/2*np.eye(2)
            np.testing.assert_allclose(s[name+'_covariance'],expected)
        self.assertEqual(meta['covariance_denominator'],'N')

    def test_direct_distances_and_signed_negative_RMD_are_preserved(self):
        f,y=self.data();_,s=fit_rmd(f,y)
        query=np.r_[s['class_means'],np.array([[0,3],[8,-2]])]
        scores,middle,_=rmd_scores(query,s)
        direct=np.array([[((q-m)@s['within_precision']@(q-m)) for m in s['class_means']] for q in query])
        bg=np.array([(q-s['background_mean'])@s['background_precision']@(q-s['background_mean']) for q in query])
        np.testing.assert_allclose(middle['class_distances'],direct,atol=1e-10)
        np.testing.assert_allclose(scores['rmd'],direct.min(1)-bg,atol=1e-10)
        self.assertTrue((scores['rmd'][:2]<0).all())
        np.testing.assert_array_equal(middle['nearest_class'][:2],[0,1])

    def test_distance_is_invariant_to_common_feature_translation(self):
        f,y=self.data();_,s=fit_rmd(f,y)
        delta=np.array([1e5,-2e5]);_,t=fit_rmd(f+delta,y)
        a,_,_=rmd_scores(f,s);b,_,_=rmd_scores(f+delta,t)
        np.testing.assert_allclose(a['rmd'],b['rmd'],atol=1e-8)

    def test_tiny_negative_distance_correction_has_fixed_gate(self):
        value,n,correction=nonnegative_distances(np.array([-1e-9,2.]))
        np.testing.assert_array_equal(value,[0,2])
        self.assertEqual(n,1);self.assertEqual(correction,1e-9)
        with self.assertRaisesRegex(ValueError,'too-negative'):
            nonnegative_distances(np.array([-2e-8]))

    def test_invalid_labels_and_changed_shrinkage_are_rejected(self):
        f,y=self.data()
        for bad in [y.astype(float),y+1,np.array([0,0,0,2,2,2])]:
            with self.assertRaises(ValueError):fit_rmd(f,bad)
        with self.assertRaisesRegex(ValueError,'fixed'):
            fit_rmd(f,y,shrinkage=.1)


if __name__=='__main__':unittest.main()
