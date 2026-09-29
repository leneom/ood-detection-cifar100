"""Local geometry artifacts and protocol bindings, compatible with both Python environments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/geometry-phase3'
CALIBRATION = ROOT / 'runs/calibration-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
ADDITIONS = ['vim', 'residual', 'react_energy', 'affine_energy']
DATASETS = ['cifar10', 'mnist', 'svhn', 'textures', 'noise']
NATURAL = DATASETS[:4]
FOLDERS = {'cifar10': 'eval-final', 'mnist': 'eval-final', 'noise': 'eval-final',
           'svhn': 'eval-extension-svhn', 'textures': 'eval-extension-textures'}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def save_arrays(path, **data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_inputs():
    pre = read(RUN / 'predeclaration.json')
    require(digest(ROOT / pre['protocol']) == pre['protocol_sha256'], 'geometry protocol changed')
    require(digest(RUN / 'official-benchmark-reference.py') == pre['reference_sha256'], 'author reference changed')
    for name, expected in [('report/baseline-complete/overview.json', pre['baseline_overview_sha256']),
                           ('report/calibration-phase2/summary.json', pre['calibration_summary_sha256']),
                           ('report/selective-risk-phase2/summary.json', pre['selective_risk_summary_sha256']),
                           ('report/selective-risk-phase2/summary-audit.json', pre['selective_risk_summary_audit_sha256'])]:
        require(digest(ROOT / name) == expected, 'predeclared completed summary changed')
    prior = read(ROOT / 'runs/selective-risk-phase2/inputs.json')
    paths = {ROOT / name for name in prior['original_artifacts']}
    for name, expected in prior['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'frozen original artifact changed: ' + name)
    paths.update([RUN / 'predeclaration.json', ROOT / pre['protocol'], RUN / 'reference.json', RUN / 'official-benchmark-reference.py',
                  ROOT / 'runs/selective-risk-phase2/inputs.json', ROOT / 'runs/selective-risk-phase2/status.json'])
    for name in ['summary.json', 'audit.json', 'summary-audit.json', 'completion-audit.json']:
        paths.add(ROOT / 'report/selective-risk-phase2' / name)
    require(read(ROOT / 'runs/selective-risk-phase2/status.json')['state'] == 'complete', 'operating-grid producer incomplete')
    for seed in [0, 1, 2]:
        paths.add(ROOT / f'runs/selective-risk-phase2/seed{seed}/results.json')
        paths.add(ROOT / f'runs/selective-risk-phase2/seed{seed}/decisions.npz')
    return {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in sorted(paths)}
