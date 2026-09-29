import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from rmd_diagnostics_core import describe, mean_difference, transitions, STATE_NAMES


class RMDDiagnosticMathTests(unittest.TestCase):
    def arrays(self):
        m = np.array([2., 5., 7., 9.])
        b = np.array([6., 7., 5., 4.])
        return dict(mahalanobis_cpu=m, background_distance=b, rmd=m-b)

    def test_signed_difference_and_population_variance(self):
        a = self.arrays()
        s = describe(a)
        self.assertEqual(s['rmd']['min'], -4.)
        self.assertEqual(s['rmd']['mean'], .25)
        self.assertEqual(s['rmd']['quantiles']['0.5'], 0.)
        self.assertEqual(s['joint']['MD_variance'], 6.6875)
        self.assertEqual(s['joint']['background_variance'], 1.25)
        self.assertEqual(s['joint']['covariance_MD_background'], -2.125)
        self.assertEqual(s['joint']['rmd_variance'], 12.1875)
        np.testing.assert_array_equal(a['rmd'], [-4., -2., 2., 5.])

    def test_zero_variance_has_null_correlation(self):
        a = dict(mahalanobis_cpu=np.ones(4), background_distance=np.ones(4)*3, rmd=np.ones(4)*-2)
        s = describe(a)
        self.assertIsNone(s['joint']['pearson_MD_background'])
        self.assertEqual(s['joint']['variance_component_difference'], 0.)

    def test_mean_change_subtracts_background(self):
        a = self.arrays()
        b = dict(mahalanobis_cpu=a['mahalanobis_cpu']+3, background_distance=a['background_distance']+7, rmd=a['rmd']-4)
        d = mean_difference(describe(b), describe(a))
        self.assertEqual(d['mahalanobis_cpu'], 3.)
        self.assertEqual(d['background_distance'], 7.)
        self.assertEqual(d['rmd'], -4.)
        self.assertEqual(d['identity_error'], 0.)

    def test_all_four_states_and_correctness_denominators(self):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        c = np.array([True, False, True, False])
        s, codes = transitions(a, b, c)
        np.testing.assert_array_equal(codes, [0, 1, 2, 3])
        self.assertEqual(s['state_counts'], dict.fromkeys(STATE_NAMES, 1))
        self.assertEqual(s['acceptance_change'], 0.)
        self.assertEqual(s['correct']['n'], 2)
        self.assertEqual(s['incorrect']['state_fractions']['rmd_only_accept'], .5)
        self.assertEqual(s['correct']['state_fractions']['reference_only_accept'], .5)

    def test_invalid_components_or_nonboolean_masks_fail(self):
        a = self.arrays()
        a['rmd'] = np.maximum(a['rmd'], 0)
        with self.assertRaises(ValueError):
            describe(a)
        a = self.arrays(); a['mahalanobis_cpu'][0] = np.nan
        with self.assertRaises(ValueError):
            describe(a)
        with self.assertRaises(ValueError):
            transitions(np.array([1, 0]), np.array([True, False]))


if __name__ == '__main__':
    unittest.main()
