import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DecisionRuleTests(unittest.TestCase):
    """The predeclared sign rule in docs/POSTHOC_CONFIRMATION.md."""

    def setUp(self):
        self.build = load("build_posthoc_confirmation_report")

    def test_population_sign_patterns(self):
        self.assertEqual(self.build.decide([0.1, 0.2, 1e-9]), "supported")
        self.assertEqual(self.build.decide([-0.1, -0.2, -1e-9]), "contradicted")
        self.assertEqual(self.build.decide([0.1, -0.2, 0.3]), "mixed")
        self.assertEqual(self.build.decide([0.1, 0.0, 0.3]), "mixed")

    def test_overall_decision(self):
        overall = self.build.overall
        self.assertEqual(overall(["supported"] * 4), "confirmed")
        self.assertEqual(overall(["supported", "mixed", "mixed", "supported"]), "partially confirmed")
        self.assertEqual(overall(["supported", "contradicted", "supported", "supported"]), "not confirmed")
        self.assertEqual(overall(["mixed"] * 4), "not confirmed")


if __name__ == "__main__":
    unittest.main()
