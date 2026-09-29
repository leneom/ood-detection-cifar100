import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from geometry_core import fit_geometry, geometry_scores


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.f = np.random.default_rng(30).normal(size=(150, 4)) + [10, 3, 1, 2]
        self.w = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
        self.b = np.array([.5, .25])
        self.meta, self.state = fit_geometry(self.f, self.w, self.b, principal_dim=2)

    def test_origin_and_second_moment_do_not_center_on_empirical_mean(self):
        np.testing.assert_allclose(self.state['origin'], [-.5, -.25, 0, 0])
        x = self.f - self.state['origin']
        np.testing.assert_allclose(self.state['second_moment'], x.T @ x / len(x))
        self.assertGreater(self.state['second_moment'][0, 0], 100)
        self.assertFalse(self.meta['sample_mean_subtracted'])

    def test_orthonormal_subspaces_and_residual_projection_identity(self):
        p, r = self.state['principal_basis'], self.state['residual_basis']
        np.testing.assert_allclose(r.T @ r, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(p.T @ r, np.zeros((2, 2)), atol=1e-12)
        x = self.f - self.state['origin']
        expected = np.sqrt(np.maximum(np.sum(x*x, axis=1) - np.sum((x @ p)**2, axis=1), 0))
        np.testing.assert_allclose(self.state['training_residual'], expected, atol=1e-12)

    def test_vim_is_scaled_residual_plus_affine_energy(self):
        scores, intermediate = geometry_scores(self.f, self.w, self.b, self.meta, self.state)
        expected = (self.f @ self.w.T + self.b).max(axis=1).mean() / scores['residual'].mean()
        self.assertAlmostEqual(self.meta['alpha'], expected)
        np.testing.assert_allclose(scores['vim'], self.meta['alpha'] * scores['residual'] + scores['affine_energy'])
        np.testing.assert_array_equal(intermediate['virtual_logit'], self.meta['alpha'] * scores['residual'])

    def test_global_clip_quantile_and_upper_only_cap(self):
        ordered = sorted(self.f.flatten())
        index = .99 * (len(ordered) - 1)
        low = int(np.floor(index))
        expected = ordered[low] + (index - low) * (ordered[low + 1] - ordered[low])
        self.assertAlmostEqual(self.meta['clipping_level'], expected)
        query = np.array([[-7., 1000., -5., 1000.]])
        _, intermediate = geometry_scores(query, self.w, self.b, self.meta, self.state)
        np.testing.assert_allclose(intermediate['react_logits'], [[-6.5, self.meta['clipping_level'] + .25]])
        self.assertEqual(intermediate['clipped_coordinate_count'][0], 2)

    def test_unclipped_rows_have_identical_react_and_affine_energy(self):
        query = np.full((3, 4), -5.)
        scores, _ = geometry_scores(query, self.w, self.b, self.meta, self.state)
        np.testing.assert_array_equal(scores['react_energy'], scores['affine_energy'])

    def test_invalid_or_zero_residual_fits_fail_without_sign_changes(self):
        with self.assertRaises(ValueError):
            fit_geometry(np.tile([2., 0, 0, 0], (20, 1)), self.w, np.zeros(2), principal_dim=2)
        with self.assertRaises(ValueError):
            fit_geometry(self.f, self.w, self.b, principal_dim=4)
        with self.assertRaises(ValueError):
            fit_geometry(self.f, self.w, self.b, principal_dim=2, quantile=.95)
        with self.assertRaises(ValueError):
            fit_geometry(self.f, self.w, np.array([np.inf, 0]), principal_dim=2)


if __name__ == '__main__':
    unittest.main()
