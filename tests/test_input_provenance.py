import hashlib
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ood.evaluate import image_fingerprint


class InputProvenanceTests(unittest.TestCase):
    def test_hashes_exact_pixel_order_without_mutation_and_rejects_float_input(self):
        original = torch.arange(2 * 8 * 12 * 3).reshape(2, 8, 12, 3).to(torch.uint8)
        saved = original.clone()
        for pixels in (original, original[:, ::2]):
            record = image_fingerprint(pixels)
            expected = hashlib.sha256(pixels.numpy().tobytes(order='C')).hexdigest()
            self.assertEqual(record['sha256'], expected)
            self.assertEqual(record['shape'], list(pixels.shape))
            self.assertEqual(record['dtype'], 'uint8')
        self.assertTrue(torch.equal(original, saved))
        with self.assertRaisesRegex(ValueError, 'NHWC uint8'):
            image_fingerprint(original.float())


if __name__ == '__main__':
    unittest.main()
