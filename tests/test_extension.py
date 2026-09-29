import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('run_extension', Path(__file__).resolve().parents[1] / 'tools' / 'run_extension.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ExtensionTests(unittest.TestCase):
    def test_live_matrix_cannot_release_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'status.json').write_text(json.dumps({'state': 'complete', 'completed_seeds': [0]}))
            (root / 'controller.lock').write_text(str(os.getpid()))
            self.assertFalse(module.matrix_ready(root, [0]))

    def test_incomplete_failed_and_paused_matrices_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for state in ('running', 'failed', 'paused'):
                (root / 'status.json').write_text(json.dumps({'state': state}))
                with self.assertRaises(InterruptedError):
                    module.matrix_ready(root, [0])

    def test_completed_seed_set_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'status.json').write_text(json.dumps({'state': 'complete', 'completed_seeds': [0]}))
            self.assertTrue(module.matrix_ready(root, [0]))
            with self.assertRaises(ValueError):
                module.matrix_ready(root, [0, 1])

    def test_extension_rejects_changed_validation_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, extension = root / 'base', root / 'extension'
            baseline.mkdir()
            extension.mkdir()
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'locally generated test placeholder')
            record = {'training_config': {'seed': 0}, 'checkpoint_epoch': 89,
                      'checkpoint_sha256': module.hash_file(checkpoint), 'temperature': 1., 'feature_fit': None,
                      'datasets': {'noise': {'methods': {'msp': {}}}}}
            for path in (baseline, extension):
                (path / 'results.json').write_text(json.dumps(record))
                np.savez(path / 'noise_scores.npz', msp_id=[.1, .2], msp_val=[.2, .3])
                for filename in ('id_logits.npz', 'validation_logits.npz', 'features.npz'):
                    np.savez(path / filename, values=[.1, .2])
            module.verify_extension(baseline, extension, checkpoint, ['noise'])
            np.savez(extension / 'noise_scores.npz', msp_id=[.1, .2], msp_val=[.2, .4])
            with self.assertRaisesRegex(ValueError, 'calibration population'):
                module.verify_extension(baseline, extension, checkpoint, ['noise'])


if __name__ == '__main__':
    unittest.main()
