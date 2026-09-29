"""Independently audit every class-holdout diagnostic from frozen raw scores."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout"
DIAG = ROOT / "runs/cifar100-class-holdout-classes"
OUT = ROOT / "report/cifar100-class-holdout-classes"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)
LABELS = ("MSP", "Energy", "MaxLogit", "Entropy", "KNN-50", "Pooled Mahalanobis")
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


def close(value: float, reference: float, message: str, atol: float = 2e-12) -> None:
    require(math.isclose(float(value), float(reference), rel_tol=0, abs_tol=atol),
            message + f": {value} vs {reference}")


def quantile(values: np.ndarray, probability: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    output = OUT / "audit.json"
    require(not output.exists(), "independent class audit exists; preserve it")
    declaration = read(DIAG / "declaration.json")
    summary = read(OUT / "summary.json")
    saved = arrays(OUT / "class_values.npz")
    require(declaration["state"] == "frozen_before_per_class_metric_access" and
            summary["state"] == "descriptive_complete_pending_independent_audit" and
            summary["declaration_sha256"] == sha(DIAG / "declaration.json") and
            summary["baseline_completion_sha256"] ==
            sha(ROOT / "report/cifar100-class-holdout/completion.json") and
            summary["class_values_sha256"] == sha(OUT / "class_values.npz") and
            summary["producer_source_sha256"] ==
            sha(ROOT / "tools/build_class_holdout_heterogeneity.py") and
            summary["class_count"] == 20 and
            summary["ood_images_per_class"] == 100 and
            summary["shared_id_test_count"] == 8000 and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            summary["targets"] == list(TARGETS),
            "frozen declaration or complete diagnostic binding differs")
    for item in declaration["source_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "frozen source changed: " + item["path"])
    require(saved["auroc"].shape == (3, 2, 6, 20) and
            saved["accepted_counts"].shape == (3, 2, 6, 4, 20) and
            saved["id_test_accepted_counts"].shape == (3, 2, 6, 4) and
            saved["row_positions"].shape == (20, 100) and
            saved["original_test_rows"].shape == (20, 100) and
            tuple(saved["variants"]) == VARIANTS and
            tuple(saved["methods"]) == METHODS and
            tuple(saved["targets"]) == TARGETS,
            "full fixed-axis raw class arrays missing")
    test = arrays(RUN / "test-population.npz")
    partition = read(RUN / "partition.json")
    fine = test["ood_original_fine_labels"]
    classes = np.array(sorted(partition["ood_fine_labels"]), dtype=np.int64)
    coarse = np.asarray(partition["fine_to_coarse"], dtype=np.int64)[classes]
    names = [partition["fine_label_names"][int(class_id)] for class_id in classes]
    require(np.array_equal(saved["fine_ids"], classes) and
            np.array_equal(saved["coarse_ids"], coarse) and
            saved["fine_names"].tolist() == names and
            summary["class_fine_ids"] == classes.tolist(),
            "all original fine/coarse IDs and class names")
    positions = np.stack([np.where(fine == class_id)[0] for class_id in classes])
    require(positions.shape == (20, 100) and
            np.array_equal(np.sort(positions.ravel()), np.arange(2000)) and
            np.array_equal(saved["row_positions"], positions) and
            np.array_equal(saved["original_test_rows"],
                           test["ood_test_indices"][positions]) and
            np.array_equal(test["ood_original_coarse_labels"][positions],
                           np.broadcast_to(coarse[:, None], (20, 100))),
            "exact official original rows for every class")
    checks = {"class_auroc": 0, "class_acceptance": 0,
              "id_acceptance": 0, "aggregate_identities": 0,
              "csv_auroc_rows": 0, "csv_acceptance_rows": 0,
              "distribution_quantiles": 0, "display_rows": 0}
    expected_auroc = np.empty_like(saved["auroc"])
    expected_accept = np.empty_like(saved["accepted_counts"])
    expected_id_accept = np.empty_like(saved["id_test_accepted_counts"])
    for seed in range(3):
        score_dir = RUN / f"score-seed{seed}"
        inside = arrays(score_dir / "id_test_scores.npz")
        outside = arrays(score_dir / "ood_test_scores.npz")
        masks = arrays(score_dir / "acceptance_masks.npz")
        raw = arrays(RUN / f"extract-seed{seed}" / "ood_test_raw.npz")
        manifest = read(score_dir / "manifest.json")
        require(np.array_equal(raw["original_indices"],
                               test["ood_test_indices"]) and
                np.array_equal(raw["original_fine_labels"], fine),
                "source OOD rows changed")
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                key = f"{variant}_{method}"
                x = np.sort(inside[key])
                y = outside[key]
                require(x.shape == (8000,) and y.shape == (2000,) and
                        np.isfinite(x).all() and np.isfinite(y).all(),
                        "finite full scores: " + key)
                class_scores = y[positions]
                below = np.searchsorted(x, class_scores, side="left")
                at_or_below = np.searchsorted(x, class_scores, side="right")
                expected = ((below + at_or_below) / (2 * len(x))).mean(axis=1)
                expected_auroc[seed, vi, mi] = expected
                require(np.allclose(saved["auroc"][seed, vi, mi], expected,
                                    rtol=0, atol=2e-14),
                        "independent tied-pair AUROC: " + key)
                checks["class_auroc"] += 20
                item = manifest["method_results"][variant][method]
                close(expected.mean(), item["metrics"]["auroc"],
                      "equal-class aggregate AUROC: " + key)
                checks["aggregate_identities"] += 1
                for ti, target in enumerate(TARGETS):
                    original = item["operating_grid"][f"target{target}"]
                    threshold = original["threshold"]
                    class_counts = np.count_nonzero(class_scores <= threshold,
                                                    axis=1)
                    full_mask = np.asarray(y <= threshold, dtype=np.uint8)
                    id_mask = np.asarray(inside[key] <= threshold,
                                         dtype=np.uint8)
                    expected_accept[seed, vi, mi, ti] = class_counts
                    expected_id_accept[seed, vi, mi, ti] = int(id_mask.sum())
                    require(np.array_equal(saved["accepted_counts"][seed, vi, mi, ti],
                                           class_counts) and
                            np.array_equal(masks[f"{key}_target{target}_OOD"],
                                           full_mask) and
                            int(class_counts.sum()) == original["ood_accepted"],
                            "class acceptance or complete OOD mask: " + key)
                    require(np.array_equal(masks[f"{key}_target{target}_test_ID"],
                                           id_mask) and
                            int(id_mask.sum()) == original["id_test_accepted"] ==
                            saved["id_test_accepted_counts"][seed, vi, mi, ti],
                            "actual ID-test coverage: " + key)
                    checks["class_acceptance"] += 20
                    checks["id_acceptance"] += 1
                    checks["aggregate_identities"] += 1
    auroc_rows = list(csv.DictReader((OUT / "class_auroc.csv").open(
        encoding="utf-8", newline="")))
    accept_rows = list(csv.DictReader((OUT / "class_acceptance.csv").open(
        encoding="utf-8", newline="")))
    require(len(auroc_rows) == 240 and len(accept_rows) == 960,
            "all class CSV rows required")
    cursor = 0
    for ci, class_id in enumerate(classes):
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                row = auroc_rows[cursor]
                require((int(row["fine_id"]), row["fine_name"],
                         int(row["coarse_id"]), row["variant"], row["method"]) ==
                        (int(class_id), names[ci], int(coarse[ci]), variant, method),
                        "class AUROC CSV axes")
                values = expected_auroc[:, vi, mi, ci]
                for seed in range(3):
                    close(row[f"seed{seed}"], values[seed], "class AUROC CSV seed")
                close(row["mean"], statistics.fmean(map(float, values)),
                      "class AUROC CSV mean")
                close(row["sample_sd"], statistics.stdev(map(float, values)),
                      "class AUROC CSV sample SD")
                cursor += 1
                checks["csv_auroc_rows"] += 1
    cursor = 0
    for ci, class_id in enumerate(classes):
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                for ti, target in enumerate(TARGETS):
                    row = accept_rows[cursor]
                    require((int(row["fine_id"]), row["fine_name"],
                             int(row["coarse_id"]), row["variant"], row["method"],
                             int(row["id_target"])) ==
                            (int(class_id), names[ci], int(coarse[ci]),
                             variant, method, target),
                            "class acceptance CSV axes")
                    counts = expected_accept[:, vi, mi, ti, ci]
                    for seed in range(3):
                        require(int(row[f"seed{seed}_count"]) == int(counts[seed]),
                                "class acceptance CSV count")
                        close(row[f"id_test_acceptance_seed{seed}"],
                              expected_id_accept[seed, vi, mi, ti] / 8000,
                              "class acceptance CSV actual ID coverage")
                    rates = counts / 100
                    close(row["mean_acceptance"], statistics.fmean(map(float, rates)),
                          "class acceptance CSV mean")
                    close(row["sample_sd_acceptance"],
                          statistics.stdev(map(float, rates)),
                          "class acceptance CSV sample SD")
                    cursor += 1
                    checks["csv_acceptance_rows"] += 1
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    for vi, variant in enumerate(VARIANTS):
        section = report.split(f"## {variant}\n", 1)[1].split("## ", 1)[0]
        for mi, method in enumerate(METHODS):
            item = summary["distributions"][variant][method]
            distributions = (
                ("three_seed_class_mean_auroc",
                 expected_auroc[:, vi, mi].mean(axis=0)),
                ("three_seed_class_mean_fixed95_ood_acceptance",
                 expected_accept[:, vi, mi, 2].mean(axis=0) / 100),
            )
            printed = []
            for title, values in distributions:
                observed = item[title]
                for key, probability in zip(QKEYS, (0, .25, .5, .75, 1)):
                    close(observed[key], quantile(values, probability),
                          "independent class distribution percentile", atol=2e-15)
                    checks["distribution_quantiles"] += 1
                printed.append(" / ".join(f"{100 * observed[key]:.2f}"
                                          for key in QKEYS))
            display = f"| {LABELS[mi]} | {printed[0]} | {printed[1]} |"
            require(section.count(display) == 1,
                    "displayed all-method class distribution row")
            checks["display_rows"] += 1
    record = {
        "state": "passed", "checked": checks,
        "declaration_sha256": sha(DIAG / "declaration.json"),
        "class_values_sha256": sha(OUT / "class_values.npz"),
        "summary_sha256": sha(OUT / "summary.json"),
        "class_auroc_csv_sha256": sha(OUT / "class_auroc.csv"),
        "class_acceptance_csv_sha256": sha(OUT / "class_acceptance.csv"),
        "report_sha256": sha(OUT / "REPORT.md"),
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "All 20 held-out classes independently recomputed from frozen raw scores; descriptive same-test analysis only.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checks}), flush=True)


if __name__ == "__main__":
    main()
