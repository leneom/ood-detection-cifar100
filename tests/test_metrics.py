import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from ood.metrics import detection_metrics, id_threshold


class MetricsTests(unittest.TestCase):
    def test_perfect_separation(self):
        metrics = detection_metrics([0, 1, 2], [3, 4, 5], [0, 1, 2])
        self.assertEqual(metrics["auroc"], 1.)
        self.assertEqual(metrics["aupr_out"], 1.)
        self.assertEqual(metrics["aupr_in"], 1.)
        self.assertEqual(metrics["fpr95_id"], 0.)

    def test_ties_are_not_optimistically_broken(self):
        metrics = detection_metrics([1, 1], [1, 1, 1])
        self.assertEqual(metrics["auroc"], 0.5)
        self.assertEqual(metrics["aupr_out"], 0.6)
        self.assertEqual(metrics["aupr_in"], 0.4)
        self.assertEqual(metrics["fpr95_id"], 1.)

    def test_matches_pairwise_definition(self):
        inside = np.array([0, 0.3, 1, 1, 2])
        outside = np.array([0.3, 1, 3])
        pairwise = np.mean((outside[:, None] > inside) + .5 * (outside[:, None] == inside))
        self.assertAlmostEqual(detection_metrics(inside, outside)["auroc"], pairwise)

    def test_fpr95_and_calibration_are_distinct(self):
        metrics = detection_metrics(np.arange(100), [94, 95, 100], np.arange(20))
        self.assertEqual(metrics["benchmark_threshold"], 94.)
        self.assertEqual(metrics["fpr95_id"], 1 / 3)
        self.assertEqual(metrics["id_tpr_at_fpr95"], .95)
        self.assertEqual(metrics["validation_threshold"], 18.)
        self.assertEqual(metrics["id_test_acceptance_at_validation_threshold"], .19)

    def test_invalid_scores_fail(self):
        for invalid in ([], [float("nan")], [[1, 2]], [float("inf")]):
            with self.assertRaises(ValueError):
                detection_metrics(invalid, [0, 1])
        with self.assertRaises(ValueError):
            id_threshold([0, 1], 0)


if __name__ == "__main__":
    unittest.main()
