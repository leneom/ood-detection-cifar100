"""Fixed post-test descriptive stratification of Tiny ImageNet OOD scores."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/tinyimagenet-confirmation"
OUT = ROOT / "report/tinyimagenet-class-heterogeneity"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def distribution(values):
    q = np.quantile(values, [0, .25, .5, .75, 1], method="linear")
    return {key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), q)}


def plot(values, title, xlabel, stem):
    labels = ("MSP", "Energy", "MaxLogit", "Entropy", "KNN-50", "Pooled Mahalanobis")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for vi, (variant, color, offset) in enumerate((('T1', '#247ba0', -.14),
                                                    ('calibrated', '#dc5f3b', .14))):
        stats = []
        for mi, label in enumerate(labels):
            v = values[vi, mi] * 100
            q = np.quantile(v, [0, .25, .5, .75, 1], method="linear")
            stats.append({"label": label, "whislo": q[0], "q1": q[1],
                          "med": q[2], "q3": q[3], "whishi": q[4], "fliers": []})
        ax.bxp(stats, positions=np.arange(6) + offset, vert=False, widths=.23,
               showfliers=False, patch_artist=True,
               boxprops={"facecolor": color, "edgecolor": color, "alpha": .65},
               medianprops={"color": "black", "linewidth": 1.25},
               whiskerprops={"color": color}, capprops={"color": color})
        ax.plot([], [], color=color, linewidth=7, alpha=.65, label=variant)
    ax.set_yticks(np.arange(6), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='x', alpha=.25)
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(OUT / f'{stem}.png', dpi=180)
    fig.savefig(OUT / f'{stem}.svg')
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError("class heterogeneity output already exists")
    declaration = read(RUN / "class-heterogeneity-declaration.json")
    require(declaration["state"] == "declared_before_per_synset_scoring" and
            declaration["protocol_sha256"] == sha(ROOT / 'docs/TINYIMAGENET_CLASS_HETEROGENEITY.md') and
            declaration["aggregate_completion_sha256"] == sha(ROOT / 'report/tinyimagenet-confirmation/completion.json') and
            declaration["cache_sha256"] == sha(ROOT / 'data/tinyimagenet-validation.npz'),
            "frozen post-test descriptive declaration")
    cache = arrays(ROOT / 'data/tinyimagenet-validation.npz')
    synsets = np.unique(cache['synsets'])
    require(len(synsets) == 200 and len(cache['synsets']) == 10000, "fixed class census")
    rows = np.stack([np.flatnonzero(cache['synsets'] == s) for s in synsets])
    require(rows.shape == (200, 50) and np.array_equal(np.sort(rows.ravel()), np.arange(10000)),
            "exact 50-image strata partition")
    auroc = np.empty((3, 2, 6, 200), dtype=np.float64)
    accepted = np.empty((3, 2, 6, 4, 200), dtype=np.int16)
    source_hashes = {}
    for seed in range(3):
        extract = RUN / f'extract-seed{seed}/raw.npz'
        score = RUN / f'score-seed{seed}/scores.npz'
        mask = RUN / f'score-seed{seed}/acceptance_masks.npz'
        id_score = ROOT / f'runs/validation-isolation/score-seed{seed}/id_test_scores.npz'
        raw, y, m, x = (arrays(p) for p in (extract, score, mask, id_score))
        require(np.array_equal(raw['synsets'], cache['synsets']) and
                np.array_equal(raw['filenames'], cache['filenames']), "source row alignment")
        for path in (extract, score, mask, id_score):
            source_hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                key = f'{variant}_{method}'
                inside, outside = x[key], y[key]
                require(inside.shape == outside.shape == (10000,), "fixed score shape")
                labels = np.r_[np.zeros(10000, dtype=np.uint8), np.ones(50, dtype=np.uint8)]
                for ci, indices in enumerate(rows):
                    auroc[seed, vi, mi, ci] = roc_auc_score(labels, np.r_[inside, outside[indices]])
                for ti, target in enumerate(TARGETS):
                    key_mask = f'{key}_target{target}_OOD'
                    for ci, indices in enumerate(rows):
                        accepted[seed, vi, mi, ti, ci] = int(m[key_mask][indices].sum())
                    require(int(accepted[seed, vi, mi, ti].sum()) == int(m[key_mask].sum()),
                            "class OOD counts equal original total")
                aggregate = read(RUN / f'score-seed{seed}/manifest.json')['results'][variant][method]
                require(abs(float(auroc[seed, vi, mi].mean()) - aggregate['metrics']['auroc']) < 2e-12,
                        "equal-size class mean equals aggregate AUROC")
    OUT.mkdir(parents=True)
    with (OUT / 'class_values.npz').open('wb') as f:
        np.savez_compressed(f, synsets=synsets, row_indices=rows, auroc=auroc,
                            accepted=accepted, targets=np.array(TARGETS),
                            variants=np.array(VARIANTS), methods=np.array(METHODS))
    summary = {"state": "descriptive_complete_pending_independent_audit",
               "class_count": 200, "images_per_class": 50,
               "id_test_count_per_class_comparison": 10000,
               "variants": VARIANTS, "methods": METHODS, "targets": TARGETS,
               "protocol_sha256": declaration['protocol_sha256'],
               "declaration_sha256": sha(RUN / 'class-heterogeneity-declaration.json'),
               "aggregate_completion_sha256": declaration['aggregate_completion_sha256'],
               "source_hashes": source_hashes, "class_values_sha256": sha(OUT / 'class_values.npz'),
               "producer_source_sha256": sha(__file__),
               "created_at": datetime.now(timezone.utc).isoformat(),
               "distributions": {}}
    for vi, variant in enumerate(VARIANTS):
        summary['distributions'][variant] = {}
        for mi, method in enumerate(METHODS):
            summary['distributions'][variant][method] = {
                'class_mean_auroc': distribution(auroc[:, vi, mi].mean(axis=0)),
                'class_mean_fixed95_ood_acceptance':
                    distribution(accepted[:, vi, mi, 2].mean(axis=0) / 50)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Tiny ImageNet class-heterogeneity diagnostic', '',
             'Post-test descriptive stratification of the already completed Tiny ImageNet evaluation. The 200 official synsets each contribute exactly 50 OOD images; each class AUROC compares those 50 against the same complete 10,000 CIFAR-100 ID-test scores. All classifiers, methods, temperature values and operating thresholds remain frozen. This is not an additional held-out confirmation or method-selection exercise.', '',
             'The tables show the distribution across **200 classes of each class’s three-seed mean**. Values are percent; columns are min, 25th percentile, median, 75th percentile and max (linear percentile convention). These are neither confidence intervals nor train-seed variability.', '']
    names = {'msp':'MSP','energy':'Energy','maxlogit':'MaxLogit','entropy':'Entropy',
             'knn':'KNN-50','mahalanobis':'Pooled Mahalanobis'}
    for variant in VARIANTS:
        lines += [f'## {variant}', '',
                  '| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |',
                  '| --- | ---: | ---: |']
        for method in METHODS:
            d = summary['distributions'][variant][method]
            a = d['class_mean_auroc']
            b = d['class_mean_fixed95_ood_acceptance']
            fmt = lambda x: ' / '.join(f'{100*x[k]:.2f}' for k in ('min','q25','median','q75','max'))
            lines.append(f'| {names[method]} | {fmt(a)} | {fmt(b)} |')
        lines.append('')
    lines += ['## Interpretation and data', '',
              'All class IDs and exact 50 source rows, 7,200 class AUROCs, and 28,800 class operating counts are in [class_values.npz](class_values.npz). Each of the 200 equal-size class AUROCs averages exactly to the corresponding original all-image AUROC; class accepted counts sum to each original all-image OOD count. Full independent checks are in [audit.json](audit.json).', '',
              'The 50-image class acceptance rate moves in 2-percentage-point steps. Shared ID scores and shared images make these class and method values dependent. No synset was hand-labeled “near”, “far”, “hard” or “easy”; class extremes have no selection role. AUPR is omitted for class comparisons because the 50:10,000 OOD:ID prevalence differs sharply from the original 10,000:10,000 evaluation. This diagnostic cannot separate semantic overlap from image-source and 64→32 resolution effects.', '',
              'Figures: [class AUROC distributions](class-auroc.png) and [fixed95 OOD acceptance distributions](class-fixed95-acceptance.png).', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(auroc.mean(axis=0), 'Per-class AUROC, 200 Tiny ImageNet synsets (3-seed class means)',
         'Class AUROC (%)', 'class-auroc')
    plot(accepted[:, :, :, 2].mean(axis=0) / 50,
         'Per-class OOD acceptance at frozen 95% ID target (3-seed class means)',
         'Class OOD acceptance (%)', 'class-fixed95-acceptance')
    print(json.dumps({'state':summary['state'],'class_auroc_cells':int(auroc.size),
                      'class_operating_counts':int(accepted.size)}), flush=True)


if __name__ == '__main__':
    main()
