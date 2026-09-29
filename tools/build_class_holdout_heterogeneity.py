"""Describe every frozen CIFAR-100 held-out class without refitting methods."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout"
DIAG = ROOT / "runs/cifar100-class-holdout-classes"
OUT = ROOT / "report/cifar100-class-holdout-classes"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)
QKEYS = ("min", "q25", "median", "q75", "max")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0, .25, .5, .75, 1], method="linear")
    return {key: float(value) for key, value in zip(QKEYS, quantiles)}


def stat(values: np.ndarray) -> dict:
    points = [float(value) for value in values]
    return {"per_seed": points, "mean": statistics.fmean(points),
            "sample_sd": statistics.stdev(points)}


def main() -> None:
    require(not OUT.exists(), "class diagnostic output exists; preserve it")
    declaration = read(DIAG / "declaration.json")
    require(declaration["state"] == "frozen_before_per_class_metric_access" and
            declaration["seeds"] == [0, 1, 2] and
            declaration["variants"] == list(VARIANTS) and
            declaration["methods"] == list(METHODS) and
            declaration["targets"] == list(TARGETS) and
            declaration["all_held_out_fine_classes"] == 20 and
            declaration["ood_images_per_class"] == 100 and
            declaration["shared_id_test_count"] == 8000 and
            declaration["ood_test_count"] == 2000,
            "complete frozen per-class declaration required")
    for item in declaration["source_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "frozen source changed: " + item["path"])
    completion = read(ROOT / "report/cifar100-class-holdout/completion.json")
    require(completion["state"] == "complete" and
            completion["seeds"] == [0, 1, 2],
            "audited full class-holdout baseline required")
    test = arrays(RUN / "test-population.npz")
    partition = read(RUN / "partition.json")
    fine = test["ood_original_fine_labels"]
    classes = np.array(sorted(map(int, partition["ood_fine_labels"])),
                       dtype=np.int64)
    require(fine.shape == (2000,) and len(classes) == 20 and
            np.array_equal(classes, np.unique(fine)) and
            np.array_equal(classes, np.array(read(RUN / "test-population.json")
                                               ["ood_fine_labels"])),
            "complete ascending held-out fine-class census")
    positions = np.stack([np.flatnonzero(fine == value) for value in classes])
    original_rows = test["ood_test_indices"][positions]
    coarse = np.asarray(partition["fine_to_coarse"], dtype=np.int64)[classes]
    require(positions.shape == (20, 100) and
            np.array_equal(np.sort(positions.ravel()), np.arange(2000)) and
            np.array_equal(test["ood_original_coarse_labels"][positions],
                           np.broadcast_to(coarse[:, None], (20, 100))),
            "exact class rows and coarse labels")
    names = [partition["fine_label_names"][int(value)] for value in classes]
    auroc = np.empty((3, 2, 6, 20), dtype=np.float64)
    accepted = np.empty((3, 2, 6, 4, 20), dtype=np.int16)
    id_accepted = np.empty((3, 2, 6, 4), dtype=np.int32)
    for seed in range(3):
        score_dir = RUN / f"score-seed{seed}"
        id_scores = arrays(score_dir / "id_test_scores.npz")
        ood_scores = arrays(score_dir / "ood_test_scores.npz")
        masks = arrays(score_dir / "acceptance_masks.npz")
        raw = arrays(RUN / f"extract-seed{seed}" / "ood_test_raw.npz")
        manifest = read(score_dir / "manifest.json")
        audit = read(RUN / f"score-audit-seed{seed}" / "audit.json")
        require(audit["state"] == "passed" and audit["seed"] == seed and
                audit["scoring_manifest_sha256"] == sha(score_dir / "manifest.json") and
                np.array_equal(raw["original_indices"], test["ood_test_indices"]) and
                np.array_equal(raw["original_fine_labels"], fine) and
                np.array_equal(raw["original_coarse_labels"],
                               test["ood_original_coarse_labels"]),
                f"seed {seed} original OOD rows or score audit differ")
        labels = np.r_[np.zeros(8000, dtype=np.uint8),
                       np.ones(100, dtype=np.uint8)]
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                key = f"{variant}_{method}"
                inside, outside = id_scores[key], ood_scores[key]
                require(inside.shape == (8000,) and outside.shape == (2000,) and
                        np.isfinite(inside).all() and np.isfinite(outside).all(),
                        "full finite ID/OOD scores required: " + key)
                for ci, row in enumerate(positions):
                    auroc[seed, vi, mi, ci] = roc_auc_score(
                        labels, np.r_[inside, outside[row]])
                original = manifest["method_results"][variant][method]
                require(abs(float(auroc[seed, vi, mi].mean()) -
                            original["metrics"]["auroc"]) < 2e-12,
                        "equal-size per-class AUROC differs from overall " + key)
                for ti, target in enumerate(TARGETS):
                    item = original["operating_grid"][f"target{target}"]
                    mask = masks[f"{key}_target{target}_OOD"]
                    id_mask = masks[f"{key}_target{target}_test_ID"]
                    require(mask.shape == (2000,) and id_mask.shape == (8000,) and
                            np.array_equal(mask, np.asarray(
                                outside <= item["threshold"], dtype=np.uint8)) and
                            int(id_mask.sum()) == item["id_test_accepted"],
                            "original fixed ID-only threshold/mask differs")
                    accepted[seed, vi, mi, ti] = [int(mask[row].sum())
                                                  for row in positions]
                    id_accepted[seed, vi, mi, ti] = int(id_mask.sum())
                    require(int(accepted[seed, vi, mi, ti].sum()) ==
                            item["ood_accepted"],
                            "class accepted counts differ from overall " + key)
    OUT.mkdir(parents=True)
    values_path = OUT / "class_values.npz"
    with values_path.open("wb") as stream:
        np.savez_compressed(stream, fine_ids=classes, coarse_ids=coarse,
                            fine_names=np.asarray(names), row_positions=positions,
                            original_test_rows=original_rows, auroc=auroc,
                            accepted_counts=accepted,
                            id_test_accepted_counts=id_accepted,
                            variants=np.asarray(VARIANTS),
                            methods=np.asarray(METHODS), targets=np.asarray(TARGETS))
    summary = {
        "state": "descriptive_complete_pending_independent_audit",
        "declaration_sha256": sha(DIAG / "declaration.json"),
        "baseline_completion_sha256": sha(ROOT / "report/cifar100-class-holdout/completion.json"),
        "class_values_sha256": sha(values_path),
        "producer_source_sha256": sha(Path(__file__)),
        "class_count": 20, "ood_images_per_class": 100,
        "shared_id_test_count": 8000,
        "class_fine_ids": classes.tolist(),
        "variants": list(VARIANTS), "methods": list(METHODS),
        "targets": list(TARGETS), "distributions": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Post-hoc description of all 20 classes; no test-driven fitting/selection or independent confirmation.",
    }
    for vi, variant in enumerate(VARIANTS):
        summary["distributions"][variant] = {}
        for mi, method in enumerate(METHODS):
            summary["distributions"][variant][method] = {
                "three_seed_class_mean_auroc": distribution(
                    auroc[:, vi, mi].mean(axis=0)),
                "three_seed_class_mean_fixed95_ood_acceptance": distribution(
                    accepted[:, vi, mi, 2].mean(axis=0) / 100),
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                  allow_nan=False) + "\n",
                                      encoding="utf-8")
    with (OUT / "class_auroc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fine_id", "fine_name", "coarse_id", "variant", "method",
                         "seed0", "seed1", "seed2", "mean", "sample_sd"])
        for ci, class_id in enumerate(classes):
            for vi, variant in enumerate(VARIANTS):
                for mi, method in enumerate(METHODS):
                    row = stat(auroc[:, vi, mi, ci])
                    writer.writerow([int(class_id), names[ci], int(coarse[ci]),
                                     variant, method, *row["per_seed"],
                                     row["mean"], row["sample_sd"]])
    with (OUT / "class_acceptance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fine_id", "fine_name", "coarse_id", "variant", "method",
                         "id_target", "seed0_count", "seed1_count", "seed2_count",
                         "mean_acceptance", "sample_sd_acceptance",
                         "id_test_acceptance_seed0", "id_test_acceptance_seed1",
                         "id_test_acceptance_seed2"])
        for ci, class_id in enumerate(classes):
            for vi, variant in enumerate(VARIANTS):
                for mi, method in enumerate(METHODS):
                    for ti, target in enumerate(TARGETS):
                        counts = accepted[:, vi, mi, ti, ci]
                        rates = stat(counts / 100)
                        id_rates = id_accepted[:, vi, mi, ti] / 8000
                        writer.writerow([int(class_id), names[ci], int(coarse[ci]),
                                         variant, method, target, *map(int, counts),
                                         rates["mean"], rates["sample_sd"],
                                         *id_rates.tolist()])
    fmt = lambda values: " / ".join(
        f"{100 * values[key]:.2f}" for key in QKEYS)
    lines = [
        "# CIFAR-100 held-out-class heterogeneity — descriptive diagnostic", "",
        "All 20 frozen held-out fine classes contribute exactly 100 official-test OOD images each. Every class AUROC uses the same complete 8,000 ID-test scores. The six detectors, three checkpoints, temperature fits and ID-only thresholds are inherited unchanged from the completed class-holdout study. This is a post-hoc description of the same test set, not an extra confirmation or method selection.", "",
        "The following are distributions across **20 class means over three training seeds**: min / Q1 / median / Q3 / max, in percent. Quantiles use linear interpolation. They are not confidence intervals or a measure of training-seed variation.", "",
    ]
    labels = ("MSP", "Energy", "MaxLogit", "Entropy", "KNN-50", "Pooled Mahalanobis")
    for variant in VARIANTS:
        lines += [f"## {variant}", "",
                  "| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |",
                  "|---|---:|---:|"]
        for label, method in zip(labels, METHODS):
            item = summary["distributions"][variant][method]
            lines.append(f"| {label} | {fmt(item['three_seed_class_mean_auroc'])} | "
                         f"{fmt(item['three_seed_class_mean_fixed95_ood_acceptance'])} |")
        lines.append("")
    lines += [
        "## Full class data and limits", "",
        "[class_auroc.csv](class_auroc.csv) lists all 240 class/method/temperature rows and each seed value. [class_acceptance.csv](class_acceptance.csv) lists all 960 class/method/temperature/ID-target rows, exact accepted counts, training-seed mean/sample SD and actual full ID-test acceptance. [class_values.npz](class_values.npz) retains all 720 class AUROCs, 2,880 class acceptance counts, exact original test rows, fine/coarse IDs and names. The independent [audit](audit.json) checks every value against frozen source scores and masks.", "",
        "Each class contributes only 100 OOD images; a one-seed class acceptance rate moves in 1-percentage-point steps. Classes, methods and seeds share images and the full ID reference, so the distribution is dependent and cannot be interpreted as a confidence interval. The per-class OOD:ID ratio is 100:8,000, so per-class AUPR is deliberately omitted. No class was selected or labeled by semantic distance, and no per-class result is used to tune a detector. The original aggregate report remains unchanged.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": summary["state"],
                      "class_auroc_cells": int(auroc.size),
                      "class_acceptance_counts": int(accepted.size)}), flush=True)


if __name__ == "__main__":
    main()
