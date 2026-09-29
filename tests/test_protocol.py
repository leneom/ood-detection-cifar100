import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from ood.data import stratified_split, preprocess
from ood.scores import score_logits, METHODS
from ood.train import parser, run


class ProtocolTests(unittest.TestCase):
    def test_split_is_stratified_disjoint_and_repeatable(self):
        labels = np.repeat(np.arange(4), 20)
        train, val = stratified_split(labels, 5, 42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(80)))
        np.testing.assert_array_equal(np.bincount(labels[val]), [5] * 4)
        np.testing.assert_array_equal(val, stratified_split(labels, 5, 42)[1])
        self.assertFalse(np.array_equal(val, stratified_split(labels, 5, 43)[1]))

    def test_energy_shift_preserves_softmax_but_changes_energy(self):
        logits = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
        torch.testing.assert_close(score_logits(logits, "msp"), score_logits(logits + 100, "msp"))
        torch.testing.assert_close(score_logits(logits + 100, "energy"), score_logits(logits, "energy") - 100)

    def test_score_orientation_and_stability(self):
        logits = torch.tensor([[1000., 0., 0.], [0., 0., 0.]])
        for method in METHODS:
            scores = score_logits(logits, method)
            self.assertTrue(torch.isfinite(scores).all())
            self.assertLess(float(scores[0]), float(scores[1]))
        for temperature in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                score_logits(logits, "energy", temperature)

    def test_batch_augmentation_shape_and_finiteness(self):
        images = torch.randint(0, 255, (7, 32, 32, 3), dtype=torch.uint8)
        for augment in (False, True):
            transformed = preprocess(images, augment)
            self.assertEqual(transformed.shape, (7, 3, 32, 32))
            self.assertTrue(torch.isfinite(transformed).all())

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            base = ["--dataset", "synthetic", "--model", "tiny", "--epochs", "3", "--device", "cpu", "--batch-size", "32"]
            whole, resumed = Path(folder) / "whole", Path(folder) / "resumed"
            run(parser().parse_args(base + ["--output-dir", str(whole)]))
            run(parser().parse_args(base + ["--output-dir", str(resumed), "--stop-after", "1"]))
            run(parser().parse_args(base + ["--output-dir", str(resumed), "--resume"]))
            a = torch.load(whole / "last.pt", weights_only=False)
            b = torch.load(resumed / "last.pt", weights_only=False)
            self.assertEqual(a["epoch"], 3)
            self.assertEqual(a["scheduler"], b["scheduler"])
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                run(parser().parse_args(base + ["--output-dir", str(resumed), "--resume", "--lr", "0.2"]))


if __name__ == "__main__":
    unittest.main()
