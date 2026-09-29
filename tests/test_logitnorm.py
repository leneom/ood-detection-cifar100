import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ood.logitnorm_train import TAU, logitnorm_loss


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogitNormLossTests(unittest.TestCase):
    def test_matches_reference_formula(self):
        logits = torch.tensor([[3., 1., 0.], [10., 8., 6.], [-2., 0.5, 0.1]])
        targets = torch.tensor([0, 2, 1])
        norms = logits.norm(p=2, dim=1, keepdim=True) + 1e-7
        expected = torch.nn.functional.cross_entropy(logits / norms / TAU, targets)
        self.assertTrue(torch.allclose(logitnorm_loss(logits, targets), expected))

    def test_invariant_to_logit_scale(self):
        logits = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
        targets = torch.arange(8) % 5
        self.assertTrue(torch.allclose(logitnorm_loss(7.5 * logits, targets), logitnorm_loss(logits, targets)))

    def test_computed_in_float32(self):
        logits = torch.randn(4, 3).half()
        self.assertEqual(logitnorm_loss(logits, torch.tensor([0, 1, 2, 0])).dtype, torch.float32)


class LogitNormDecisionTests(unittest.TestCase):
    def setUp(self):
        self.build = load("build_logitnorm_report")

    def test_eight_population_rule(self):
        overall = self.build.overall
        self.assertEqual(overall(["supported"] * 8), "confirmed")
        self.assertEqual(overall(["supported"] * 7 + ["mixed"]), "partially confirmed")
        self.assertEqual(overall(["supported"] * 7 + ["contradicted"]), "not confirmed")
        self.assertEqual(self.build.decide([0.01, 0.02, -0.001]), "mixed")


if __name__ == "__main__":
    unittest.main()
