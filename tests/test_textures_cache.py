import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ood.textures_cache import DTD_TEST_MD5, TEXTURES_TRANSFORM, load_textures_cache, sha256


class TexturesCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='textures-cache-test-')
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to((ROOT / 'runs').resolve())
        self.target = self.root / 'textures-test1.npz'
        self.manifest_path = self.root / 'textures-test1.manifest.json'
        self.audit_path = self.root / 'audit.json'
        self.classes = [f'category{i:02d}' for i in range(47)]
        self.names = [f'{name}/image{j:03d}.jpg' for name in self.classes for j in range(40)]
        self.images = np.zeros((1880, 32, 32, 3), dtype=np.uint8)
        self.labels = np.repeat(np.arange(47, dtype=np.int64), 40)
        self.write_cache()
        self.manifest = {'schema_version': 1, 'dataset': 'textures', 'release': 'r1.0.1',
                         'split': 'test', 'partition': 1, 'count': 1880, 'source_md5': DTD_TEST_MD5,
                         'shape': [1880, 32, 32, 3], 'dtype': 'uint8', 'transform': TEXTURES_TRANSFORM,
                         'source_sha256': 'test source placeholder', 'cache_sha256': sha256(self.target),
                         'images_sha256': hashlib.sha256(self.images.tobytes()).hexdigest(),
                         'labels_sha256': hashlib.sha256(self.labels.tobytes()).hexdigest(),
                         'selected_filenames': self.names, 'class_names': self.classes,
                         'class_counts': [40] * 47, 'pillow': 'test version'}
        self.audit = {'state': 'passed', 'dataset': 'textures', 'split': 'test', 'partition': 1,
                      'all_pixel_equality_checks': 1880, 'class_label_and_order_checks': 1880,
                      'original_split_isolation': True, 'source_md5': DTD_TEST_MD5,
                      'source_sha256': self.manifest['source_sha256'],
                      'canonical_decoder': {'pillow': 'test version'}}
        self.write_metadata()

    def tearDown(self):
        self.temporary.cleanup()

    def write_cache(self):
        np.savez_compressed(self.target, images=self.images, labels=self.labels,
                            filenames=np.asarray(self.names), class_names=np.asarray(self.classes),
                            original_hw=np.full((1880, 2), 300, dtype=np.int64))

    def write_metadata(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.audit['manifest_sha256'] = sha256(self.manifest_path)
        self.audit['cache_sha256'] = sha256(self.target)
        self.audit_path.write_text(json.dumps(self.audit))

    def load(self):
        return load_textures_cache(self.root, self.audit_path)

    def test_valid_full_cache_preserves_pixels_order_and_audit(self):
        images, labels, provenance = self.load()
        self.assertTrue(np.array_equal(images, self.images))
        self.assertTrue(np.array_equal(labels, self.labels))
        self.assertEqual(provenance['selected_filenames'], self.names)
        self.assertEqual(provenance['independent_audit']['sha256'], sha256(self.audit_path))

    def test_wrong_partition_transform_and_incomplete_audit_are_rejected(self):
        self.manifest['partition'] = 2
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            self.load()
        self.manifest['partition'] = 1
        self.manifest['transform'] = {'resize': 'different'}
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            self.load()
        self.manifest['transform'] = TEXTURES_TRANSFORM
        self.audit['all_pixel_equality_checks'] = 1879
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'complete independent'):
            self.load()

    def test_container_or_manifest_corruption_is_rejected(self):
        with self.target.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'cache hash'):
            self.load()
        self.write_cache()
        self.write_metadata()
        self.manifest_path.write_text(self.manifest_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'audited provenance'):
            self.load()

    def test_changed_pixel_bytes_rejected_even_if_container_hash_updated(self):
        self.images[0, 0, 0, 0] = 1
        self.write_cache()
        self.manifest['cache_sha256'] = sha256(self.target)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'pixel-byte'):
            self.load()

    def test_filename_label_disagreement_rejected(self):
        self.labels[0], self.labels[40] = self.labels[40], self.labels[0]
        self.write_cache()
        self.manifest['cache_sha256'] = sha256(self.target)
        self.manifest['labels_sha256'] = hashlib.sha256(self.labels.tobytes()).hexdigest()
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'filename categories'):
            self.load()


if __name__ == '__main__':
    unittest.main()
