import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('aggregate_results', Path(__file__).resolve().parents[1] / 'tools' / 'aggregate_results.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class AggregationTests(unittest.TestCase):
    def test_sample_sd_and_single_seed_boundary(self):
        self.assertEqual(module.statistics([1, 2, 3])['sample_sd'], 1.)
        self.assertIsNone(module.statistics([1])['sample_sd'])

    def test_rejects_unfinished_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evaluation = root / 'eval'
            evaluation.mkdir()
            (evaluation / 'results.json').write_text(json.dumps({'training_config': {'seed': 0, 'epochs': 100}}))
            (root / 'status.json').write_text(json.dumps({'state': 'paused', 'epoch': 10}))
            with self.assertRaises(ValueError):
                module.collect([evaluation])

    def test_rejects_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evaluation = root / 'eval'
            evaluation.mkdir()
            record = {'training_config': {'seed': 0, 'epochs': 100}, 'temperature': 1., 'datasets': {}}
            (evaluation / 'results.json').write_text(json.dumps(record))
            (root / 'status.json').write_text(json.dumps({'state': 'complete', 'epoch': 100}))
            with self.assertRaises(ValueError):
                module.collect([evaluation, evaluation])


if __name__ == '__main__':
    unittest.main()
