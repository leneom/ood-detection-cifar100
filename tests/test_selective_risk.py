import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from selective_risk_core import threshold_for_target, id_decisions, ood_decisions


class SelectiveRiskTests(unittest.TestCase):
    def test_calibration_threshold_keeps_ties_and_does_not_guarantee_test_coverage(self):
        threshold, calibration, _ = threshold_for_target([1, 1, 2, 2, 2], .8)
        self.assertEqual(threshold, 2.)
        self.assertEqual(calibration['achieved_acceptance'], 1.)
        self.assertEqual(calibration['equal_threshold_count'], 3)
        result, _ = id_decisions([0, 3, 4, 5], np.ones(4, dtype=bool), threshold)
        self.assertEqual(result['coverage'], .25)
        self.assertEqual(result['retained_error_risk'], 0.)

    def test_error_risk_and_class_conditional_rejection_have_distinct_denominators(self):
        correct = np.array([True, False, True, False, True])
        result, accepted = id_decisions(np.arange(5), correct, 2)
        np.testing.assert_array_equal(accepted, [True, True, True, False, False])
        self.assertEqual(result['accepted'], 3)
        self.assertEqual(result['retained_error_risk'], 1 / 3)
        self.assertEqual(result['accepted_accuracy'], 2 / 3)
        self.assertEqual(result['correct_rejection_fraction'], 1 / 3)
        self.assertEqual(result['incorrect_rejection_fraction'], 1 / 2)
        self.assertEqual(result['correct_rejected'] + result['incorrect_rejected'], result['rejected'])

    def test_empty_denominators_remain_null(self):
        result, _ = id_decisions([2, 3], np.ones(2, dtype=bool), 1.)
        self.assertIsNone(result['retained_error_risk'])
        self.assertIsNone(result['accepted_accuracy'])
        self.assertIsNone(result['incorrect_rejection_fraction'])
        self.assertEqual(result['correct_rejection_fraction'], 1.)

    def test_foreign_ood_counts_do_not_define_classification_correctness(self):
        result, accepted = ood_decisions([0, 1, 2, 3], 1.)
        self.assertEqual(result, {'n': 4, 'accepted': 2, 'rejected': 2, 'acceptance': .5, 'equal_threshold_count': 1})
        np.testing.assert_array_equal(accepted, [True, True, False, False])

    def test_fixed_target_grid_has_nested_acceptance(self):
        calibration = np.arange(2500)
        previous = -1
        for target in [.8, .9, .95, .99]:
            threshold, record, _ = threshold_for_target(calibration, target)
            self.assertEqual(record['accepted'], int(np.ceil(target * 2500)))
            self.assertGreater(threshold, previous)
            previous = threshold

    def test_invalid_targets_scores_and_correctness_are_rejected(self):
        for target in [0, -1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                threshold_for_target([1, 2], target)
        for values in [[], [1, np.inf], [[1, 2]]]:
            with self.assertRaises(ValueError):
                ood_decisions(values, 1.)
        with self.assertRaises(ValueError):
            id_decisions([1, 2], np.array([1, 0]), 1.)
        with self.assertRaises(ValueError):
            id_decisions([1, 2], np.ones(1, dtype=bool), 1.)


if __name__ == '__main__':
    unittest.main()
