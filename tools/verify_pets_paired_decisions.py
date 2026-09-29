"""Write once, then verify the hash-bound Pet paired-decision diagnostic."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/oxford-pets-paired-decisions'
REPORT = ROOT / 'report/oxford-pets-paired-decisions'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate():
    from tools.verify_pets_completion import validate as original_validate
    previous = read(ROOT / 'report/oxford-pets-extension/completion.json')
    require(previous['state'] == 'complete' and
            previous['inputs_sha256'] == original_validate(),
            'original 105-file Pet completion')
    declaration = read(BASE / 'declaration.json')
    require(declaration['state'] == 'frozen_before_pairwise_calculation' and
            len(declaration['input_sha256']) == 15,
            'frozen declaration')
    for rel, expected in declaration['input_sha256'].items():
        require(sha(ROOT / rel) == expected, 'frozen source changed ' + rel)
    summary = read(REPORT / 'summary.json')
    audit = read(REPORT / 'audit.json')
    require(summary['declaration_sha256'] == sha(BASE / 'declaration.json') and
            summary['tables'] == 720 and len(summary['aggregates']) == 120 and
            len(summary['pairs']) == 15 and
            audit['source_sha256'] == sha(BASE / 'declaration.json') and
            audit['checked_tables'] == 720 and
            audit['checked_row_codes'] == 4920840 and
            audit['checked_source_masks'] == 288 and
            audit['checked_aggregates'] == 120 and
            audit['checked_display_rows'] == 15,
            'independent per-row and report audits')
    log = (REPORT / 'regression.log').read_text(encoding='utf-8', errors='replace')
    require(re.search(r'Ran 91 tests in [0-9.]+s', log) and
            log.rstrip().endswith('OK'), '91-test regression')
    names = list(declaration['input_sha256'])
    names += ['runs/oxford-pets-paired-decisions/declaration.json',
              'tools/build_pets_paired_decisions.py',
              'tools/audit_pets_paired_decisions.py',
              'tools/verify_pets_paired_decisions.py']
    names += [f'report/oxford-pets-paired-decisions/{name}' for name in
              ('tables.csv', 'summary.json', 'REPORT.md', 'audit.json', 'regression.log')]
    names += [f'report/oxford-pets-paired-decisions/codes-seed{seed}-{variant}.npz'
              for seed in (0, 1, 2) for variant in ('T1', 'calibrated')]
    require(len(names) == len(set(names)) == 30, 'complete 30-file census')
    return {rel: sha(ROOT / rel) for rel in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    target = REPORT / 'completion.json'
    if args.write:
        require(not target.exists(), 'completion already exists')
        hashes = validate()
        result = {'state': 'complete',
                  'completed_at': datetime.now(timezone.utc).isoformat(),
                  'tables': 720, 'pairs': 15, 'row_codes': 4920840,
                  'regression': {'tests': 91, 'exit_code': 0},
                  'boundary': 'Post-result descriptive decisions only; original ID-only thresholds and scores unchanged.',
                  'inputs_sha256': hashes}
        target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    result = read(target)
    require(result['state'] == 'complete' and result['tables'] == 720 and
            result['pairs'] == 15 and result['row_codes'] == 4920840 and
            result['inputs_sha256'] == validate(),
            'read-only hash-bound completion')
    print(json.dumps({'state': 'verified_complete',
                      'files': len(result['inputs_sha256']),
                      'tables': result['tables']}))


if __name__ == '__main__':
    main()