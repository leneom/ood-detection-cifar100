import importlib.util
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from calibration_core import (stratified_calibration_rows, logit_scores,
                              calibration_metrics, nll, fit_temperature)


class CalibrationTests(unittest.TestCase):
    def test_split_is_balanced_disjoint_reproducible_and_covers_validation(self):
        labels = np.repeat(np.arange(100), 50)
        fit, threshold = stratified_calibration_rows(labels)
        other = stratified_calibration_rows(labels)
        np.testing.assert_array_equal(fit, other[0])
        self.assertEqual(len(np.intersect1d(fit, threshold)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[fit, threshold]), np.arange(5000))
        for rows in [fit, threshold]:
            np.testing.assert_array_equal(np.bincount(labels[rows]), np.full(100, 25))

    def test_split_rejects_wrong_population(self):
        with self.assertRaises(ValueError):
            stratified_calibration_rows(np.zeros(5000, dtype=int))

    def test_uniform_probabilities_have_known_metrics(self):
        metrics = calibration_metrics(np.zeros((3, 2)), np.array([0, 1, 0]))
        self.assertAlmostEqual(metrics['nll'], np.log(2))
        self.assertAlmostEqual(metrics['brier'], .5)
        self.assertAlmostEqual(metrics['ece15'], 1 / 6)
        self.assertEqual(sum(b['count'] for b in metrics['bins']), 3)

    def test_temperature_preserves_argmax_and_maxlogit_not_probability_confidence(self):
        z = np.array([[1000., 999., -999.], [-2., 7., 0.], [1., 1., -3.]])
        y = np.array([0, 1, 2])
        original, calibrated = logit_scores(z), logit_scores(z, 2.5)
        np.testing.assert_array_equal(original['maxlogit'], calibrated['maxlogit'])
        self.assertFalse(np.array_equal(original['msp'], calibrated['msp']))
        self.assertEqual(calibration_metrics(z, y)['accuracy'], calibration_metrics(z, y, 2.5)['accuracy'])
        self.assertTrue(all(np.isfinite(v).all() for v in calibrated.values()))

    def test_invalid_temperature_and_labels_are_rejected(self):
        for temperature in [0, -1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                logit_scores(np.ones((2, 2)), temperature)
        with self.assertRaises(ValueError):
            nll(np.zeros((2, 2)), np.array([0, 2]))

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy available in CPU analysis environment')
    def test_fit_reduces_nll_for_overconfident_mistakes(self):
        z = np.tile([4., 0.], (20, 1))
        y = np.r_[np.zeros(15, dtype=int), np.ones(5, dtype=int)]
        fit = fit_temperature(z, y)
        self.assertIsNone(fit['fallback_reason'])
        self.assertAlmostEqual(fit['temperature'], 4 / np.log(3), places=5)
        self.assertLess(fit['selected_fit_nll'], fit['fit_nll_T1'])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy available in CPU analysis environment')
    def test_failed_or_worse_optimizer_retains_T1(self):
        z, y = np.array([[2., 0.], [0., 2.]]), np.array([0, 1])
        for success, objective, reason in [(False, 0., 'optimizer_failure'),
                                           (True, 9., 'fit_NLL_increased'),
                                           (True, np.nan, 'nonfinite_optimizer_result')]:
            bad = SimpleNamespace(x=0., fun=objective, success=success, message='test', nit=1, nfev=1)
            with patch('scipy.optimize.minimize_scalar', return_value=bad):
                fitted = fit_temperature(z, y)
            self.assertEqual(fitted['temperature'], 1.)
            self.assertEqual(fitted['fallback_reason'], reason)


if __name__ == '__main__':
    unittest.main()
