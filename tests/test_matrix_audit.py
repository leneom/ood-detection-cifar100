import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module('audit_matrix', 'audit_matrix.py')
aggregate = module('aggregate_results_audit_reference', 'aggregate_results.py')


def fixture():
    results = []
    for seed in range(3):
        datasets = {}
        for name in ('cifar10', 'mnist', 'noise'):
            methods = {}
            for method in ('msp', 'energy'):
                row = {metric: .5 for metric in audit.METRICS}
                row.update(n_id=10000, n_ood=10000, auroc=.4 + seed * .1 + (.1 if method == 'energy' else 0),
                           fpr95_id=.8 - (.1 if method == 'energy' else 0))
                methods[method] = row
            datasets[name] = {'methods': methods}
        results.append({'training_config': {'seed': seed, 'epochs': 100}, 'id_test_accuracy': .7 + seed * .01,
                        'checkpoint_epoch': 90, 'checkpoint_sha256': f'diagnostic-placeholder-{seed}', 'datasets': datasets})
    return results


class MatrixAuditTests(unittest.TestCase):
    def test_population_sd_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sample SD'):
            audit.verify_stat({'mean': 2., 'sample_sd': (2 / 3) ** .5, 'n_seeds': 3, 'per_seed': [1., 2., 3.]}, [1., 2., 3.])
        self.assertEqual(audit.verify_stat({'mean': 2., 'sample_sd': 1., 'n_seeds': 3, 'per_seed': [1., 2., 3.]}, [1., 2., 3.]), 0.)

    def test_requires_all_predeclared_seeds(self):
        results = fixture()[:2]
        with self.assertRaisesRegex(ValueError, 'predeclared seeds'):
            audit.compare(aggregate.aggregate(results), results, [0, 1, 2])

    def test_paired_difference_preserves_classifier_correlation(self):
        results = fixture()
        report = audit.compare(aggregate.aggregate(results), results, [0, 1, 2])
        paired = report['paired_method_minus_msp']['cifar10']['energy']['auroc']
        self.assertAlmostEqual(paired['mean'], .1)
        self.assertLess(paired['sample_sd'], 1e-15)
        self.assertAlmostEqual(audit.stats([.4, .5, .6])['sample_sd'], .1)

    def test_synthetic_diagnostics_cannot_enter_natural_macro(self):
        results = fixture()
        summary = copy.deepcopy(aggregate.aggregate(results))
        summary['natural_macro_datasets'].append('noise')
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            audit.compare(summary, results, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
