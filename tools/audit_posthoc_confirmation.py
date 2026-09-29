"""Independent CPU audit of the ASH-S / RMD contrast confirmation.

Shares no scoring code with tools/score_posthoc_confirmation.py, ash_core.py,
rmd_core.py or src/ood/metrics.py: ASH uses lexsort top-k and SciPy logsumexp,
RMD uses explicit covariances with Cholesky whitening and cdist, AUROC uses
scikit-learn. Every primary difference and every decision is recomputed.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "posthoc-confirmation"
REPORT = ROOT / "report" / "posthoc-confirmation"
SEEDS = (0, 1, 2)
PRIMARY = ("tinyimagenet", "fashionmnist", "places365", "oxford_pets")
SECONDARY = ("cifar10", "mnist", "svhn", "textures")
POPULATIONS = PRIMARY + SECONDARY
RECOMPUTED = ("ash_p_energy", "ash_b_energy", "ash_s_energy", "rmd", "matched_cpu_md")
SAVED = {"energy": "T1_energy", "mahalanobis": "T1_mahalanobis", "knn": "T1_knn"}
CONTRASTS = {"H1": ("ash_s_energy", "energy"), "H2": ("rmd", "matched_cpu_md"),
             "ASH-S - KNN": ("ash_s_energy", "knn"), "RMD - KNN": ("rmd", "knn"),
             "RMD - Energy": ("rmd", "energy")}
TARGETS = (80, 90, 95, 99)
# RMD is a difference of two large quadratic forms computed by different algebra, so it gets a
# looser (still far below any ranking-relevant scale) tolerance than ASH. Fixed before the first run.
TOLERANCE = {"ash_p_energy": (1e-9, 1e-8), "ash_b_energy": (1e-9, 1e-8), "ash_s_energy": (1e-9, 1e-8),
             "rmd": (1e-7, 1e-6), "matched_cpu_md": (1e-7, 1e-6)}
RETAINED_FRACTION = 0.35
SHRINKAGE = 0.01


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


def raw_path(seed, population):
    if population in SECONDARY:
        return ROOT / f"runs/validation-isolation/extract-seed{seed}/{population}_raw.npz"
    folder = {"tinyimagenet": "tinyimagenet-confirmation", "fashionmnist": "fashionmnist-extension",
              "places365": "places365-extension", "oxford_pets": "oxford-pets-extension"}[population]
    return ROOT / f"runs/{folder}/extract-seed{seed}/raw.npz"


def saved_path(seed, population):
    if population in SECONDARY:
        return ROOT / f"runs/validation-isolation/score-seed{seed}/{population}_scores.npz"
    return raw_path(seed, population).parent.parent / f"score-seed{seed}" / "scores.npz"


def ash_independent(features, weight, bias):
    f = features.astype(np.float64)
    n, d = f.shape
    keep = int(round(d * RETAINED_FRACTION))
    columns = np.broadcast_to(np.arange(d), f.shape)
    order = np.lexsort((columns, -f), axis=-1)[:, :keep]
    mask = np.zeros(f.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    pruned = np.where(mask, f, 0.)
    total, retained = f.sum(1), pruned.sum(1)
    shaped = {"ash_p_energy": pruned,
              "ash_b_energy": np.where(mask, (total / keep)[:, None], 0.),
              "ash_s_energy": pruned * np.exp(total / retained)[:, None]}
    out = {}
    for name, value in shaped.items():
        logits = value @ weight.T + bias
        out[name] = -logsumexp(logits, axis=1)
        if name == "ash_s_energy":
            out["ash_s_predictions"] = logits.argmax(1)
    return out


class IndependentRMD:
    def __init__(self, features, labels):
        f = features.astype(np.float64)
        n, d = f.shape
        classes = np.unique(labels)
        counts = np.bincount(labels, minlength=len(classes)).astype(np.float64)
        onehot = np.zeros((n, len(classes)))
        onehot[np.arange(n), labels] = 1.
        self.means = (onehot.T @ f) / counts[:, None]
        # Within-class covariance from raw second moments minus weighted mean outer products.
        within = (f.T @ f - (self.means * counts[:, None]).T @ self.means) / n
        background = np.cov(f, rowvar=False, bias=True)
        self.origin = f.mean(0)
        self.lower = {name: cholesky(self.shrink(cov), lower=True)
                      for name, cov in (("within", within), ("background", background))}
        self.white_means = self.whiten("within", self.means)

    @staticmethod
    def shrink(cov):
        d = cov.shape[0]
        return (1 - SHRINKAGE) * cov + SHRINKAGE * (np.trace(cov) / d) * np.eye(d)

    def whiten(self, name, x):
        return solve_triangular(self.lower[name], x.T, lower=True).T

    def scores(self, features):
        f = features.astype(np.float64)
        md = cdist(self.whiten("within", f), self.white_means, "sqeuclidean").min(1)
        bg = (self.whiten("background", f - self.origin) ** 2).sum(1)
        return {"matched_cpu_md": md, "rmd": md - bg}


def order_threshold(values, target):
    ranked = np.sort(np.asarray(values, dtype=np.float64))
    return float(ranked[math.ceil(target / 100 * len(ranked)) - 1])


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    import torch
    failures, counts = [], {"score_vectors": 0, "auroc_values": 0, "operating_cells": 0, "contrast_values": 0}
    declaration = load_json(RUN / "declaration.json")
    declaration_sha = digest(RUN / "declaration.json")
    summary = load_json(REPORT / "summary.json")
    check(summary["declaration_sha256"] == declaration_sha, "summary not bound to declaration", failures)
    partition = npz(ROOT / "runs/validation-isolation/partition.npz")["threshold_rows"]
    auroc, operating = {}, {}
    for seed in SEEDS:
        for path, expected in declaration["inputs"][f"seed{seed}"].items():
            check(digest(ROOT / path) == expected, f"input hash differs: {path}", failures)
        folder = RUN / f"score-seed{seed}"
        manifest = load_json(folder / "manifest.json")
        check(summary["seed_manifests"][f"seed{seed}"] == digest(folder / "manifest.json"),
              f"summary/manifest mismatch seed {seed}", failures)
        saved_new = npz(folder / "scores.npz")
        state = torch.load(ROOT / f"runs/validation-isolation/seed{seed}/final-best.pt",
                           map_location="cpu", weights_only=False)["model"]
        weight, bias = state["fc.weight"].double().numpy(), state["fc.bias"].double().numpy()
        train = npz(ROOT / f"runs/validation-isolation/extract-seed{seed}/id_train_raw.npz")
        rmd = IndependentRMD(train["features"], train["labels"])

        def all_scores(raw, saved_file):
            values = {**ash_independent(raw["features"], weight, bias), **rmd.scores(raw["features"])}
            saved = npz(saved_file)
            values.update({name: saved[key].astype(np.float64) for name, key in SAVED.items()})
            return values

        id_test_raw = npz(ROOT / f"runs/validation-isolation/extract-seed{seed}/id_test_raw.npz")
        inside = all_scores(id_test_raw, ROOT / f"runs/validation-isolation/score-seed{seed}/id_test_scores.npz")
        validation = all_scores(npz(ROOT / f"runs/validation-isolation/extract-seed{seed}/id_validation_raw.npz"),
                                ROOT / f"runs/validation-isolation/score-seed{seed}/id_validation_scores.npz")
        labels = id_test_raw["labels"]
        side = manifest["id_side_effects"]
        check(abs(side["ash_s_accuracy"] - float((inside["ash_s_predictions"] == labels).mean())) < 1e-12,
              f"ASH-S ID accuracy differs seed {seed}", failures)
        check(side["ash_s_changed_predictions"] ==
              int((inside["ash_s_predictions"] != id_test_raw["logits"].argmax(1)).sum()),
              f"ASH-S changed-prediction count differs seed {seed}", failures)
        for name in RECOMPUTED:
            for prefix, value in (("id_test", inside[name]), ("id_threshold", validation[name][partition])):
                check(np.allclose(saved_new[f"{prefix}_{name}"], value, *TOLERANCE[name]),
                      f"{prefix} {name} differs seed {seed}", failures)
                counts["score_vectors"] += 1
        thresholds = {m: {t: order_threshold(validation[m][partition], t) for t in TARGETS} for m in inside
                      if m != "ash_s_predictions"}
        for population in POPULATIONS:
            outside = all_scores(npz(raw_path(seed, population)), saved_path(seed, population))
            for name in RECOMPUTED:
                check(np.allclose(saved_new[f"{population}_{name}"], outside[name], *TOLERANCE[name]),
                      f"{population} {name} differs seed {seed}", failures)
                counts["score_vectors"] += 1
            for method, per_target in thresholds.items():
                value = roc_auc_score(np.r_[np.zeros(len(inside[method])), np.ones(len(outside[method]))],
                                      np.r_[inside[method], outside[method]])
                reported = manifest["results"][population][method]
                check(abs(value - reported["metrics"]["auroc"]) < 1e-10,
                      f"AUROC differs {population} {method} seed {seed}", failures)
                auroc[(seed, population, method)] = value
                counts["auroc_values"] += 1
                for t, threshold in per_target.items():
                    cell = reported["operating_grid"][f"target{t}"]
                    ood = float(np.mean(outside[method] <= threshold))
                    idt = float(np.mean(inside[method] <= threshold))
                    check(math.isclose(cell["threshold"], threshold, rel_tol=1e-9, abs_tol=1e-9) and
                          abs(cell["ood_acceptance"] - ood) < 1e-12 and abs(cell["id_test_acceptance"] - idt) < 1e-12,
                          f"operating point differs {population} {method} target{t} seed {seed}", failures)
                    operating[(seed, population, method, t)] = (ood, idt)
                    counts["operating_cells"] += 1

    decisions = {}
    for name, (new, reference) in CONTRASTS.items():
        for population in POPULATIONS:
            deltas = [auroc[(s, population, new)] - auroc[(s, population, reference)] for s in SEEDS]
            reported = summary["contrasts"][name][population]
            check(np.allclose(deltas, reported["auroc_difference"]["seeds"], rtol=0, atol=1e-10),
                  f"contrast differs {name} {population}", failures)
            ood = [operating[(s, population, new, 95)][0] - operating[(s, population, reference, 95)][0] for s in SEEDS]
            check(np.allclose(ood, reported["fixed95_ood_acceptance_difference"]["seeds"], rtol=0, atol=1e-12),
                  f"fixed95 contrast differs {name} {population}", failures)
            counts["contrast_values"] += 6
            if name in ("H1", "H2"):
                pattern = ("supported" if all(d > 0 for d in deltas) else
                           "contradicted" if all(d < 0 for d in deltas) else "mixed")
                check(pattern == reported["sign_pattern"], f"sign pattern differs {name} {population}", failures)
                decisions.setdefault(name, {})[population] = pattern
    overall = {}
    for name in ("H1", "H2"):
        per = [decisions[name][p] for p in PRIMARY]
        supported = per.count("supported")
        overall[name] = ("confirmed" if supported == 4 else
                         "partially confirmed" if supported and "contradicted" not in per else "not confirmed")
        check(overall[name] == summary["decisions"][name]["overall"], f"overall decision differs {name}", failures)
    report_text = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    for name, label in (("H1", "H1 ASH-S > Energy"), ("H2", "H2 RMD > matched MD")):
        row = f"| {label} | " + " | ".join(decisions[name][p] for p in PRIMARY) + f" | **{overall[name]}** |"
        check(row in report_text, f"report decision row differs {name}", failures)
    with open(REPORT / "metrics.csv", encoding="utf-8") as stream:
        check(sum(1 for _ in csv.reader(stream)) - 1 == 3 * len(POPULATIONS) * 8, "metrics.csv row count", failures)
    with open(REPORT / "contrasts.csv", encoding="utf-8") as stream:
        check(sum(1 for _ in csv.reader(stream)) - 1 == 3 * len(POPULATIONS) * len(CONTRASTS),
              "contrasts.csv row count", failures)
    result = {"state": "passed" if not failures else "failed", "failures": failures[:50],
              "failure_count": len(failures), "counts": counts, "decisions": decisions, "overall": overall,
              "declaration_sha256": declaration_sha, "summary_sha256": digest(REPORT / "summary.json"),
              "report_sha256": digest(REPORT / "REPORT.md"),
              "independent_implementations": {"ash": "numpy lexsort top-k + scipy.special.logsumexp",
                                              "rmd": "raw second moments, numpy.cov, Cholesky whitening, cdist",
                                              "auroc": "sklearn.metrics.roc_auc_score",
                                              "threshold": "sorted order statistic ceil(q*n)-1"},
              "audited_at": datetime.now(timezone.utc).isoformat()}
    (REPORT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "failure_count", "counts", "overall")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
