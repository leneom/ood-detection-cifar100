"""Independent CPU audit of the LogitNorm representation study.

Recomputes every LogitNorm score from raw logits/features without the project's
scoring code (SciPy logits, scikit-learn KNN, Cholesky Mahalanobis), every AUROC
with scikit-learn, every fixed threshold, and every decision in the report.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "logitnorm-representation"
ISOLATED = ROOT / "runs" / "validation-isolation"
REPORT = ROOT / "report" / "logitnorm-representation"
SEEDS = (0, 1, 2)
ORIGINAL = ("cifar10", "mnist", "svhn", "textures")
LATER = {"tinyimagenet": "tinyimagenet-confirmation", "fashionmnist": "fashionmnist-extension",
         "places365": "places365-extension", "oxford_pets": "oxford-pets-extension"}
POPULATIONS = ORIGINAL + tuple(LATER)
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)
# Saved scores come from float32 GPU/CPU kernels; the audit recomputes in float64.
TOLERANCE = {"msp": (1e-4, 1e-6), "entropy": (1e-4, 1e-5), "energy": (1e-5, 1e-4), "maxlogit": (1e-6, 1e-5),
             "knn": (1e-4, 1e-5), "mahalanobis": (1e-7, 1e-6)}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def logit_scores(logits):
    z = logits.astype(np.float64)
    p = softmax(z, axis=1)
    return {"msp": 1 - p.max(1), "energy": -logsumexp(z, axis=1), "maxlogit": -z.max(1),
            "entropy": -(p * np.log(np.clip(p, 1e-300, None))).sum(1)}


class FeatureScores:
    def __init__(self, features, labels):
        f = features.astype(np.float64)
        bank = f / np.linalg.norm(f, axis=1, keepdims=True)
        self.knn = NearestNeighbors(n_neighbors=50, algorithm="brute", metric="euclidean").fit(bank)
        classes = np.unique(labels)
        self.means = np.stack([f[labels == k].mean(0) for k in classes])
        centred = np.concatenate([f[labels == k] - self.means[i] for i, k in enumerate(classes)])
        cov = centred.T @ centred / len(f)
        cov = 0.99 * cov + 0.01 * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
        self.lower = cholesky(cov, lower=True)
        self.white_means = solve_triangular(self.lower, self.means.T, lower=True).T

    def __call__(self, features):
        f = features.astype(np.float64)
        distances, _ = self.knn.kneighbors(f / np.linalg.norm(f, axis=1, keepdims=True))
        white = solve_triangular(self.lower, f.T, lower=True).T
        return {"knn": distances[:, -1] ** 2,
                "mahalanobis": cdist(white, self.white_means, "sqeuclidean").min(1)}


def ce_saved(seed, role):
    if role in ("id_test", "id_validation") or role in ORIGINAL:
        saved = npz(ISOLATED / f"score-seed{seed}" / f"{role}_scores.npz")
    else:
        saved = npz(ROOT / "runs" / LATER[role] / f"score-seed{seed}" / "scores.npz")
    return {m: saved[f"T1_{m}"].astype(np.float64) for m in METHODS}


def order_threshold(values, target):
    ranked = np.sort(values)
    return float(ranked[math.ceil(target / 100 * len(ranked)) - 1])


def main():
    failures = []
    counts = {"score_vectors": 0, "auroc_values": 0, "operating_cells": 0, "contrast_values": 0}

    def check(ok, message):
        if not ok:
            failures.append(message)

    summary = load_json(REPORT / "summary.json")
    evaluation = load_json(PHASE / "evaluation-declaration.json")
    check(summary["evaluation_declaration_sha256"] == digest(PHASE / "evaluation-declaration.json"),
          "summary not bound to evaluation declaration")
    training = load_json(PHASE / "declaration.json")
    check(evaluation["training_declaration_sha256"] == digest(PHASE / "declaration.json"),
          "evaluation declaration not bound to training declaration")
    rows = npz(ISOLATED / "partition.npz")["threshold_rows"]
    auroc, operating, accuracy = {}, {}, {}
    for seed in SEEDS:
        for name, expected in training["ce_comparators"][f"seed{seed}"].items():
            check(digest(ROOT / name) == expected, f"CE comparator changed: {name}")
        manifest = load_json(PHASE / f"score-seed{seed}" / "manifest.json")
        check(summary["seed_manifests"][f"seed{seed}"] == digest(PHASE / f"score-seed{seed}" / "manifest.json"),
              f"summary/manifest mismatch seed {seed}")
        extract = PHASE / f"extract-seed{seed}"
        train = npz(extract / "id_train_raw.npz")
        features = FeatureScores(train["features"], train["labels"])

        def ln(role):
            raw = npz(extract / f"{role}_raw.npz")
            independent = {**logit_scores(raw["logits"]), **features(raw["features"])}
            saved = npz(extract / f"{role}_scores.npz")
            for m in METHODS:
                rtol, atol = TOLERANCE[m]
                check(np.allclose(saved[m].astype(np.float64), independent[m], rtol=rtol, atol=atol),
                      f"LogitNorm {m} score differs {role} seed {seed}")
                counts["score_vectors"] += 1
            return {m: saved[m].astype(np.float64) for m in METHODS}, raw

        ln_test, ln_raw = ln("id_test")
        ln_val, _ = ln("id_validation")
        ce_raw = npz(ISOLATED / f"extract-seed{seed}" / "id_test_raw.npz")
        accuracy[seed] = {"ce": float((ce_raw["logits"].argmax(1) == ce_raw["labels"]).mean()),
                          "logitnorm": float((ln_raw["logits"].argmax(1) == ln_raw["labels"]).mean())}
        for objective in accuracy[seed]:
            check(abs(accuracy[seed][objective] - manifest["id_test_accuracy"][objective]) < 1e-12,
                  f"accuracy differs {objective} seed {seed}")
        sources = {"ce": (ce_saved(seed, "id_test"), ce_saved(seed, "id_validation"), lambda p: ce_saved(seed, p)),
                   "logitnorm": (ln_test, ln_val, lambda p: ln(p)[0])}
        for objective, (inside, validation, outside_of) in sources.items():
            thresholds = {m: {t: order_threshold(validation[m][rows], t) for t in TARGETS} for m in METHODS}
            for p in POPULATIONS:
                outside = outside_of(p)
                for m in METHODS:
                    labels = np.r_[np.zeros(len(inside[m])), np.ones(len(outside[m]))]
                    value = roc_auc_score(labels, np.r_[inside[m], outside[m]])
                    reported = manifest["results"][objective][p][m]
                    check(abs(value - reported["metrics"]["auroc"]) < 1e-10, f"AUROC differs {objective} {p} {m} {seed}")
                    auroc[(seed, objective, p, m)] = value
                    counts["auroc_values"] += 1
                    for t in TARGETS:
                        cell = reported["operating_grid"][f"target{t}"]
                        ood = float(np.mean(outside[m] <= thresholds[m][t]))
                        idt = float(np.mean(inside[m] <= thresholds[m][t]))
                        check(math.isclose(cell["threshold"], thresholds[m][t], rel_tol=1e-12, abs_tol=1e-12) and
                              abs(cell["ood_acceptance"] - ood) < 1e-12 and
                              abs(cell["id_test_acceptance"] - idt) < 1e-12,
                              f"operating point differs {objective} {p} {m} target{t} seed {seed}")
                        operating[(seed, objective, p, m, t)] = ood
                        counts["operating_cells"] += 1
            if objective == "ce":
                for p in LATER:
                    earlier = load_json(ROOT / "runs" / LATER[p] / f"score-seed{seed}" / "manifest.json")
                    for m in METHODS:
                        check(abs(earlier["results"]["T1"][m]["metrics"]["auroc"] - auroc[(seed, "ce", p, m)]) < 1e-10,
                              f"CE AUROC differs from the earlier {p} report: {m} seed {seed}")
    decisions = {}
    for m in METHODS:
        for p in POPULATIONS:
            deltas = [auroc[(s, "logitnorm", p, m)] - auroc[(s, "ce", p, m)] for s in SEEDS]
            entry = summary["contrasts"][m][p]
            check(np.allclose(deltas, entry["auroc_difference"]["seeds"], rtol=0, atol=1e-10),
                  f"contrast differs {m} {p}")
            ood = [operating[(s, "logitnorm", p, m, 95)] - operating[(s, "ce", p, m, 95)] for s in SEEDS]
            check(np.allclose(ood, entry["fixed95_ood_acceptance_difference"]["seeds"], rtol=0, atol=1e-12),
                  f"fixed95 contrast differs {m} {p}")
            pattern = ("supported" if all(d > 0 for d in deltas) else
                       "contradicted" if all(d < 0 for d in deltas) else "mixed")
            check(pattern == entry["sign_pattern"], f"sign pattern differs {m} {p}")
            decisions.setdefault(m, {})[p] = pattern
            counts["contrast_values"] += 6
    overall = {}
    report_text = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    for name, m, label in (("H3", "msp", "H3 MSP improves"), ("H4", "knn", "H4 KNN-50 improves")):
        per = [decisions[m][p] for p in POPULATIONS]
        supported = per.count("supported")
        overall[name] = ("confirmed" if supported == len(per) else
                         "partially confirmed" if supported and "contradicted" not in per else "not confirmed")
        check(overall[name] == summary["decisions"][name]["overall"], f"overall decision differs {name}")
        row = f"| {label} | " + " | ".join(per) + f" | **{overall[name]}** |"
        check(row in report_text, f"report decision row differs {name}")
    result = {"state": "passed" if not failures else "failed", "failure_count": len(failures),
              "failures": failures[:50], "counts": counts, "overall": overall,
              "per_population": {name: decisions[m] for name, m in (("H3", "msp"), ("H4", "knn"))},
              "id_test_accuracy": accuracy, "summary_sha256": digest(REPORT / "summary.json"),
              "report_sha256": digest(REPORT / "REPORT.md"),
              "independent_implementations": {"logit_scores": "scipy softmax/logsumexp in float64",
                                              "knn": "sklearn NearestNeighbors brute force on L2-normalised features",
                                              "mahalanobis": "per-class centring, Cholesky whitening, cdist",
                                              "auroc": "sklearn.metrics.roc_auc_score",
                                              "threshold": "sorted order statistic ceil(q*n)-1"},
              "audited_at": datetime.now(timezone.utc).isoformat()}
    (REPORT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "failure_count", "counts", "overall")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
