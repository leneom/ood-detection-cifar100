import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from ood.features import KNN, Mahalanobis, extract
from ood.models import build_model
from ood.data import preprocess


class FeatureTests(unittest.TestCase):
    def test_knn_matches_brute_force(self):
        torch.manual_seed(10)
        bank, queries = torch.randn(40, 8), torch.randn(13, 8)
        detector = KNN(k=5, chunk_size=3).fit(bank)
        norm_bank = torch.nn.functional.normalize(bank, dim=1)
        norm_queries = torch.nn.functional.normalize(queries, dim=1)
        expected = torch.cdist(norm_queries, norm_bank).square().sort(dim=1).values[:, 4]
        torch.testing.assert_close(detector.score(queries), expected)

    def test_mahalanobis_matches_direct_quadratic_form(self):
        torch.manual_seed(11)
        labels = torch.arange(3).repeat_interleave(20)
        features = torch.randn(60, 4) + labels[:, None] * 3
        detector = Mahalanobis(shrinkage=0.1, chunk_size=2).fit(features, labels)
        queries = torch.randn(7, 4).double()
        differences = queries[:, None] - detector.means[None]
        expected = torch.einsum('ncd,de,nce->nc', differences, detector.precision, differences).amin(dim=1)
        torch.testing.assert_close(detector.score(queries), expected)

    def test_feature_hook_preserves_predictions_and_is_removed(self):
        model = build_model('tiny', 4).eval()
        images = torch.randint(0, 255, (8, 32, 32, 3), dtype=torch.uint8)
        expected = model(preprocess(images))
        logits, features = extract(model, images, batch_size=3)
        torch.testing.assert_close(logits, expected)
        self.assertEqual(features.shape, (8, 32))
        self.assertFalse(model[-1]._forward_pre_hooks)

    def test_invalid_knn_inputs(self):
        with self.assertRaises(ValueError):
            KNN(k=5).fit(torch.ones(4, 2))
        with self.assertRaises(ValueError):
            KNN(k=1).fit(torch.zeros(4, 2))


if __name__ == '__main__':
    unittest.main()
