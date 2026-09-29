"""Independent audit of every class-holdout bootstrap report statistic."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-bootstrap"
REPORT = ROOT / "report/cifar100-class-holdout-bootstrap"
BASE = ROOT / "runs/cifar100-class-holdout"
BASELINE = ROOT / "report/cifar100-class-holdout/summary.json"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
FIELDS = ("auroc", "aupr_out", "aupr_in", "fpr95_id",
          "id_test_acceptance_target95", "ood_acceptance_target95")
NUMBERS = ("original_seed0", "original_seed1", "original_seed2",
           "original_mean", "training_seed_sample_sd",
           "paired_image_bootstrap_q025", "paired_image_bootstrap_median",
           "paired_image_bootstrap_q975")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def require(value: bool, label: str) -> None:
    if not value:
        raise AssertionError(label)


def percentile_sorted(values: np.ndarray, proportion: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = proportion * (len(ordered) - 1)
    lower = int(math.floor(position))
    higher = int(math.ceil(position))
    return float(ordered[lower] * (higher - position) +
                 ordered[higher] * (position - lower)) if higher != lower else float(ordered[lower])


def main() -> None:
    output = REPORT / "audit.json"
    require(not output.exists(), "write-once report audit already exists")
    summary = read(REPORT / "summary.json")
    raw_audit = read(RUN / "raw-audit.json")
    baseline = read(BASELINE)
    require(raw_audit["state"] == "passed" and
            summary["state"] == "report_pending_independent_audit" and
            summary["seeds"] == [0, 1, 2] and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            summary["fields"] == list(FIELDS) and
            summary["replicates"] == 1000 and
            summary["id_images_per_draw"] == 8000 and
            summary["ood_images_per_draw"] == 2000 and
            summary["ood_prevalence"] == .2,
            "complete fixed bootstrap report schema")
    source_paths = {
        "declaration": RUN / "declaration.json",
        "launch": RUN / "launch.json",
        "raw_audit": RUN / "raw-audit.json",
        "baseline_summary": BASELINE,
        "report_builder": ROOT / "tools/build_class_holdout_bootstrap_report.py",
    }
    require(summary["source_sha256"] ==
            {label: sha(path) for label, path in source_paths.items()},
            "all report source hashes")
    require(summary["statistics_csv_sha256"] == sha(REPORT / "statistics.csv"),
            "report CSV hash")
    ranked, inside_counts, outside_counts = [], [], []
    for block in range(10):
        path = RUN / "blocks" / f"block-{block:03d}-metrics.npz"
        require(raw_audit["block_hashes"][path.relative_to(ROOT).as_posix()] ==
                sha(path), "audited bootstrap block hash")
        part = arrays(path)
        require(np.array_equal(part["replicates"],
                               np.arange(block * 100, (block + 1) * 100)),
                "audited replicate order")
        ranked.append(part["ranking"])
        inside_counts.append(part["id_accepted_counts"])
        outside_counts.append(part["ood_accepted_counts"])
    ranked = np.concatenate(ranked)
    inside_counts = np.concatenate(inside_counts)
    outside_counts = np.concatenate(outside_counts)
    require(ranked.shape == (1000, 3, 2, 6, 4) and
            inside_counts.shape == outside_counts.shape == (1000, 3, 2, 6),
            "all 1000 already audited draws")
    with (REPORT / "statistics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    rows = summary["rows"]
    require(len(rows) == len(csv_rows) == 72, "all 72 report statistics")
    original_cells = {(cell["variant"], cell["method"]): cell
                      for cell in baseline["cells"]}
    require(len(original_cells) == 12, "all original cells")
    original_raw = {}
    for seed in range(3):
        folder = BASE / f"score-seed{seed}"
        original_raw[seed] = (arrays(folder / "id_test_scores.npz"),
                              arrays(folder / "ood_test_scores.npz"),
                              arrays(folder / "acceptance_masks.npz"))
    checks = {"original_seed_ranking_fields": 0, "original_fixed_mask_fields": 0,
              "bootstrap_quantiles": 0, "report_numeric_fields": 0,
              "csv_fields": 0, "display_rows": 0}
    seen = set()
    for index, (row, csv_row) in enumerate(zip(rows, csv_rows)):
        vi, mi, fi = np.unravel_index(index, (2, 6, 6))
        variant, method, field = VARIANTS[vi], METHODS[mi], FIELDS[fi]
        require((row["variant"], row["method"], row["field"]) ==
                (variant, method, field), "complete fixed cell ordering")
        require(set(row) == set(csv_row) and
                csv_row["variant"] == variant and
                csv_row["method"] == method and
                csv_row["field"] == field and
                int(csv_row["bootstrap_draws"]) == row["bootstrap_draws"] == 1000,
                "all report/CSV columns and labels")
        original = original_cells[variant, method]
        if fi < 4:
            draws = ranked[:, :, vi, mi, fi]
            ref = original["metrics"][field]
        else:
            denominator = 8000 if fi == 4 else 2000
            counts = inside_counts if fi == 4 else outside_counts
            draws = counts[:, :, vi, mi] / denominator
            label = "id_test_acceptance" if fi == 4 else "ood_acceptance"
            ref = original["operating"]["target95"][label]
        points = np.asarray(ref["per_seed"], dtype=np.float64)
        require(points.shape == (3,), "all original seed points")
        for seed in range(3):
            x, y, masks = original_raw[seed]
            key = f"{variant}_{method}"
            inside, outside = x[key], y[key]
            require(inside.shape == (8000,) and outside.shape == (2000,),
                    "original full official test scores")
            if fi == 0:
                labels = np.r_[np.zeros(8000), np.ones(2000)]
                direct = roc_auc_score(labels, np.r_[inside, outside])
            elif fi == 1:
                labels = np.r_[np.zeros(8000), np.ones(2000)]
                direct = average_precision_score(labels, np.r_[inside, outside])
            elif fi == 2:
                labels = np.r_[np.zeros(8000), np.ones(2000)]
                direct = average_precision_score(1 - labels,
                                                  -np.r_[inside, outside])
            elif fi == 3:
                threshold = np.partition(inside, math.ceil(.95 * 8000) - 1)[
                    math.ceil(.95 * 8000) - 1]
                direct = float(np.mean(outside <= threshold))
            else:
                suffix = "test_ID" if fi == 4 else "OOD"
                actual_mask = masks[f"{key}_target95_{suffix}"]
                require(actual_mask.shape == (8000 if fi == 4 else 2000,),
                        "original fixed ID-validation acceptance mask")
                direct = float(actual_mask.mean())
            require(abs(float(points[seed]) - direct) < 1e-12,
                    "original seed value independently recomputed")
            checks["original_seed_ranking_fields" if fi < 4 else
                   "original_fixed_mask_fields"] += 1
        mean = float(sum(points) / 3)
        sd = float(math.sqrt(sum((v - mean) ** 2 for v in points) / 2))
        require(abs(mean - ref["mean"]) < 1e-12 and
                abs(sd - ref["sample_sd"]) < 1e-12,
                "original training-seed mean/sample SD")
        paired = np.sum(draws, axis=1) / 3
        quantiles = [percentile_sorted(paired, p) for p in (.025, .5, .975)]
        expected = dict(zip(NUMBERS, [*points, mean, sd, *quantiles]))
        for key in NUMBERS:
            require(math.isfinite(float(row[key])) and
                    abs(float(row[key]) - float(expected[key])) < 1e-12,
                    "independent full-1000-draw statistic: " + key)
            require(abs(float(csv_row[key]) - float(expected[key])) < 1e-12,
                    "independent CSV statistic: " + key)
            checks["report_numeric_fields"] += 1
            checks["csv_fields"] += 1
        checks["bootstrap_quantiles"] += 3
        seen.add((variant, method, field))
    require(len(seen) == 72, "no omitted or duplicated fields")
    display = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    indexed = {(r["variant"], r["method"], r["field"]): r for r in rows}
    displayed = [line for line in display.splitlines() if
                 line.startswith("| T1 |") or line.startswith("| calibrated |")]
    require(len(displayed) == 12, "exactly 12 all-method display rows")
    for variant in VARIANTS:
        for method in METHODS:
            a = indexed[variant, method, "auroc"]
            o = indexed[variant, method, "ood_acceptance_target95"]
            expected_line = (
                f"| {variant} | {method} | {100*a['original_mean']:.2f} ± {100*a['training_seed_sample_sd']:.2f}% | "
                f"{100*a['paired_image_bootstrap_q025']:.2f}–{100*a['paired_image_bootstrap_q975']:.2f}% | "
                f"{100*o['original_mean']:.2f} ± {100*o['training_seed_sample_sd']:.2f}% | "
                f"{100*o['paired_image_bootstrap_q025']:.2f}–{100*o['paired_image_bootstrap_q975']:.2f}% |"
            )
            require(expected_line in displayed, "display row differs from audited values")
            checks["display_rows"] += 1
    require("not a confidence interval" in display and
            "same draw is shared" in display and
            "FPR95-ID is recomputed" in display,
            "bootstrap scope and operating-point limitations")
    record = {
        "state": "passed", "checked": checks,
        "source_sha256": {"summary": sha(REPORT / "summary.json"),
                          "statistics_csv": sha(REPORT / "statistics.csv"),
                          "display": sha(REPORT / "REPORT.md"),
                          "raw_audit": sha(RUN / "raw-audit.json"),
                          "audit_source": sha(Path(__file__))},
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "All 72 cells checked against 1000 paired draws, original full scores/masks and independent sklearn seed references; descriptive conditional image resampling only.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checks}))


if __name__ == "__main__":
    main()
