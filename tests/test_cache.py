import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from ood.cache import load_svhn_cache, SVHN_TEST_MD5


class SVHNCacheTests(unittest.TestCase):
    def test_verified_cache_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'svhn-test.npz'
            np.savez_compressed(archive, images=np.zeros((26032, 32, 32, 3), dtype=np.uint8),
                                labels=np.arange(26032, dtype=np.int64) % 10)
            manifest = {'schema_version': 1, 'dataset': 'svhn', 'split': 'test', 'count': 26032,
                        'source_md5': SVHN_TEST_MD5, 'dtype': 'uint8', 'shape': [26032, 32, 32, 3],
                        'cache_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            (root / 'svhn-test.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            images, labels, _ = load_svhn_cache(root)
            self.assertEqual(images.shape, (26032, 32, 32, 3))
            self.assertEqual(int(labels.max()), 9)
            with archive.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_svhn_cache(root)

    def test_train_split_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'svhn-test.manifest.json').write_text(json.dumps({'split': 'train'}))
            with self.assertRaisesRegex(ValueError, 'official test schema'):
                load_svhn_cache(root)


if __name__ == '__main__':
    unittest.main()
