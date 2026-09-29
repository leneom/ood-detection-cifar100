"""Write once or independently verify the Places365 extension completion record."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/places365-extension'
OUT = ROOT / 'report/places365-extension'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def inputs(declaration):
    names = [
        'docs/PLACES365_EXTENSION.md', 'docs/PLACES365_AMENDMENT_1.md',
        'runs/places365-extension/declaration.json',
        'runs/places365-extension/amendment-1.json',
        'runs/places365-extension/download-launch.json',
        'runs/places365-extension/download-status.json',
        'runs/places365-extension/download.stdout.log',
        'runs/places365-extension/download.stderr.log',
        'runs/places365-extension/cache-status.json',
        'runs/places365-extension/cache-manifest.json',
        'runs/places365-extension/cache-audit.json',
        'runs/places365-extension/extraction-launch.json',
        'runs/places365-extension/extraction-status.json',
        'runs/places365-extension/controller.stdout.log',
        'runs/places365-extension/controller.stderr.log',
        'data/places365-val_256.tar',
        'data/places365-filelist_places365-standard.tar',
        'data/places365-places365_val.txt',
        'data/places365-categories_places365.txt',
        'data/places365-validation.npz',
        'report/places365-extension/summary.json',
        'report/places365-extension/REPORT.md',
        'report/places365-extension/auroc.png',
        'report/places365-extension/auroc.svg',
        'report/places365-extension/fixed95-acceptance.png',
        'report/places365-extension/fixed95-acceptance.svg',
        'report/places365-extension/display-audit.json',
        'report/places365-extension/figure-inspection.json',
        'report/places365-extension/regression.log',
        'tools/freeze_places365.py', 'tools/download_places365.py',
        'tools/prepare_places365.py', 'tools/audit_places365_cache.py',
        'tools/extract_places365.py', 'tools/run_places365_extraction.py',
        'tools/audit_places365_features.py', 'tools/score_places365.py',
        'tools/audit_places365_scores.py', 'tools/build_places365_report.py',
        'tools/audit_places365_report.py',
        'tools/verify_places365_completion.py',
    ]
    names.extend(declaration['input_sha256'])
    for seed in range(3):
        names.extend([
            f'runs/places365-extension/extract-seed{seed}/status.json',
            f'runs/places365-extension/extract-seed{seed}/manifest.json',
            f'runs/places365-extension/extract-seed{seed}/raw.npz',
            f'runs/places365-extension/extract-seed{seed}/T1_scores.npz',
            f'runs/places365-extension/extract-seed{seed}/feature_scores.npz',
            f'runs/places365-extension/extract-seed{seed}.stdout.log',
            f'runs/places365-extension/extract-seed{seed}.stderr.log',
            f'runs/places365-extension/feature-audit-seed{seed}/audit.json',
            f'runs/places365-extension/score-seed{seed}/status.json',
            f'runs/places365-extension/score-seed{seed}/manifest.json',
            f'runs/places365-extension/score-seed{seed}/scores.npz',
            f'runs/places365-extension/score-seed{seed}/acceptance_masks.npz',
            f'runs/places365-extension/score-audit-seed{seed}/audit.json',
        ])
    return list(dict.fromkeys(names))


def validate():
    d = read(RUN / 'declaration.json')
    amendment = read(RUN / 'amendment-1.json')
    download = read(RUN / 'download-status.json')
    cache = read(RUN / 'cache-audit.json')
    cm = read(RUN / 'cache-manifest.json')
    controller = read(RUN / 'extraction-status.json')
    summary = read(OUT / 'summary.json')
    display = read(OUT / 'display-audit.json')
    visual = read(OUT / 'figure-inspection.json')
    require(d['state'] == 'protocol_frozen_before_places365_archive_access' and
            d['protocol_sha256'] == digest(ROOT / 'docs/PLACES365_EXTENSION.md'),
            'frozen protocol')
    for path, expected in d['input_sha256'].items():
        require(digest(ROOT / path) == expected, 'original declaration input '+path)
    require(amendment['original_declaration_sha256'] == digest(RUN/'declaration.json') and
            amendment['amendment_sha256'] == digest(ROOT/'docs/PLACES365_AMENDMENT_1.md'),
            'pre-download amendment')
    require(download['state'] == 'verified_archives_and_metadata_pending_pixel_cache' and
            download['declaration_sha256'] == digest(RUN/'declaration.json') and
            download['amendment_sha256'] == digest(RUN/'amendment-1.json'),
            'verified official download')
    for role in ('images','devkit','annotations','categories'):
        item = download['files'][role]
        require(item['state'].startswith('verified_') and
                digest(ROOT/item['relative_path']) == item['sha256'] and
                item['md5'] == d['expected_md5'][role], 'official asset '+role)
    require(cm['count'] == cache['images_checked'] == cache['labels_checked'] ==
            cache['resized_pixel_arrays_checked'] == 36500 and
            cache['state'] == 'passed' and
            cm['cache_sha256'] == cache['cache_sha256'] ==
            digest(ROOT/'data/places365-validation.npz'), 'all-image independent pixel audit')
    require(controller['state'] == 'three_seeds_extracted_pending_audit_and_scoring' and
            controller['completed_seeds'] == [0,1,2], 'sequential controller completed')
    require(summary['state'] == 'three_seed_places365_scored_and_audited' and
            summary['id_test_count'] == 10000 and summary['ood_count'] == 36500 and
            summary['ood_prevalence'] == 36500/46500 and
            display['state'] == visual['state'] == 'passed' and
            display['summary_sha256'] == digest(OUT/'summary.json') and
            display['report_sha256'] == digest(OUT/'REPORT.md'), 'audited report')
    for image, info in visual['images'].items():
        require(info['sha256'] == digest(OUT/image) and
                info['size'][0] >= 1500 and info['size'][1] >= 800,
                'visually checked image '+image)
    regression = (OUT/'regression.log').read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests', regression)
    require(match is not None and int(match.group(1)) == 91 and
            re.search(r'(?m)^OK(?: \(skipped=2\))?\s*$', regression) and
            'FAILED' not in regression, 'complete CPU regression')
    for seed in range(3):
        epath = RUN/f'extract-seed{seed}'
        spath = RUN/f'score-seed{seed}'
        e = read(epath/'manifest.json')
        s = read(spath/'manifest.json')
        fa = read(RUN/f'feature-audit-seed{seed}/audit.json')
        sa = read(RUN/f'score-audit-seed{seed}/audit.json')
        record = summary['seed_records'][seed]
        require(e['seed'] == s['seed'] == fa['seed'] == sa['seed'] == seed and
                e['count'] == s['n_ood'] == 36500 and s['n_id_test'] == 10000 and
                e['checkpoint_sha256'] == s['checkpoint_sha256'] == record['checkpoint_sha256'] and
                fa['state'] == sa['state'] == 'passed' and
                fa['extraction_manifest_sha256'] == digest(epath/'manifest.json') and
                sa['scoring_manifest_sha256'] == digest(spath/'manifest.json') and
                s['prior_id_score_manifest_sha256'] ==
                digest(ROOT/f'runs/validation-isolation/score-seed{seed}/manifest.json'),
                'seed lineage/audits '+str(seed))
        for stage, manifest in ((epath,e),(spath,s)):
            for name, value in manifest['outputs'].items():
                require(digest(stage/name) == value, 'stage output '+str(stage/name))
            for name, value in manifest['source_hashes'].items():
                require(digest(ROOT/name) == value, 'stage source '+name)
    return {name:digest(ROOT/name) for name in inputs(d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    target = OUT/'completion.json'
    if args.write:
        require(not target.exists(), 'completion already written')
        hashes = validate()
        record = {
            'state':'complete', 'completed_at':datetime.now(timezone.utc).isoformat(),
            'population':'Places365-Standard official validation, all 36500 images',
            'id_test_count':10000, 'ood_count':36500, 'seeds':[0,1,2],
            'primary_cells':12, 'independent_score_metric_fields':468,
            'independent_score_masks':432,
            'independent_summary_seed_values':828,
            'regression':{'total':91,'exit_code':0},
            'inputs_sha256':hashes,
            'boundary':'Sequentially chosen single OOD population; frozen models and ID-only fitting/thresholds; no OOD-based selection or multi-dataset benchmark claim.'
        }
        target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    record = read(target)
    require(record['state'] == 'complete' and record['seeds'] == [0,1,2],
            'completion header')
    require(record['inputs_sha256'] == validate(), 'hash-bound completion inputs changed')
    print(json.dumps({'state':'verified_complete','files':len(record['inputs_sha256']),
                      'seeds':record['seeds']}))

if __name__ == '__main__':
    main()