"""Independently check every nine-model report number and displayed row."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/cifar100-class-holdout-replication"
OLD = ROOT / "runs/cifar100-class-holdout"
REPORT = ROOT / "report/cifar100-class-holdout-replication"
PARTITIONS = (2031, 2032, 2033)
SEEDS = (0, 1, 2)
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")
TARGETS = (80, 90, 95, 99)
RATES = ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def close(actual: object, expected: float, label: str) -> None:
    value = float(actual)
    require(math.isfinite(value) and math.isfinite(expected) and
            math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {value!r} vs {expected!r}")


def independent_stats(raw: dict[int, list[float]], saved: dict, label: str) -> list[float]:
    require(set(saved) == {"partitions", "grand_mean",
                           "between_partition_means_sample_sd"}, label)
    require(set(saved["partitions"]) == {str(p) for p in PARTITIONS}, label)
    flat, means, within = [], [], []
    for partition in PARTITIONS:
        points = np.asarray(raw[partition], dtype=np.float64)
        require(points.shape == (3,) and np.isfinite(points).all(), label)
        entry = saved["partitions"][str(partition)]
        require(set(entry) == {"per_seed", "mean", "sample_sd"}, label)
        for i, value in enumerate(points):
            close(entry["per_seed"][i], float(value), f"{label} point {partition}/{i}")
        mean, sd = float(np.mean(points)), float(np.std(points, ddof=1))
        close(entry["mean"], mean, f"{label} mean {partition}")
        close(entry["sample_sd"], sd, f"{label} within SD {partition}")
        flat.extend(float(x) for x in points)
        means.append(mean)
        within.append(sd)
    grand, between = float(np.mean(means)), float(np.std(means, ddof=1))
    close(saved["grand_mean"], grand, f"{label} grand mean")
    close(saved["between_partition_means_sample_sd"], between,
          f"{label} between-partition SD")
    return [*flat, *means, *within, grand, between]


def csv_numbers(row: dict, header: list[str]) -> list[float]:
    return [float(row[name]) for name in header]


def main() -> None:
    audit_path = REPORT / "audit.json"
    snapshot = REPORT / "audit-source.py"
    require(not audit_path.exists() and not snapshot.exists(), "write-once report audit")
    summary_path = REPORT / "summary.json"
    metric_path = REPORT / "metrics.csv"
    operating_path = REPORT / "operating.csv"
    display_path = REPORT / "SUMMARY.md"
    summary = read(summary_path)
    declaration = read(BASE / "report-declaration.json")
    require(summary["state"] == "report_pending_independent_audit" and
            summary["class_seeds"] == list(PARTITIONS) and
            summary["training_seeds"] == list(SEEDS) and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            summary["metric_fields"] == list(METRICS) and
            summary["targets"] == list(TARGETS) and
            summary["operating_rates"] == list(RATES) and
            summary["n_id_test_per_partition"] == 8000 and
            summary["n_ood_test_per_partition"] == 2000 and
            summary["ood_prevalence"] == .2 and
            declaration["state"] == "frozen_before_new_partition_ood_test_scores" and
            summary["source_sha256"] == sha(
                ROOT / "tools/build_class_holdout_replication_report.py") ==
            sha(REPORT / "executed_report_source.py"),
            "frozen full-grid report header and source")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"frozen declaration input {name}")
    for name, expected in summary["source_inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"report source input {name}")
    require(summary["source_inputs_sha256"][
        "runs/cifar100-class-holdout-replication/report-declaration.json"] ==
        sha(BASE / "report-declaration.json"), "frozen report declaration hash")
    scores = {
        partition: [
            read((OLD if partition == 2031 else BASE / f"split{partition}") /
                 f"score-seed{seed}/manifest.json")
            for seed in SEEDS]
        for partition in PARTITIONS
    }
    for partition in PARTITIONS:
        for seed, manifest in enumerate(scores[partition]):
            require(manifest["seed"] == seed and manifest["n_id"] == 8000 and
                    manifest["n_ood"] == 2000 and manifest["methods"] == list(METHODS) and
                    manifest["targets"] == [.8, .9, .95, .99],
                    f"source score manifest {partition}/{seed}")
    numeric_fields = [
        *(f"class{p}_seed{s}" for p in PARTITIONS for s in SEEDS),
        *(f"class{p}_mean" for p in PARTITIONS),
        *(f"class{p}_within_sd" for p in PARTITIONS),
        "grand_mean", "between_partition_means_sd",
    ]
    def recompute(getter, saved, label):
        raw = {p: [float(getter(scores[p][s])) for s in SEEDS] for p in PARTITIONS}
        return independent_stats(raw, saved, label)
    accuracy = recompute(lambda row: row["id_test_accuracy"],
                         summary["id_test_accuracy"], "ID test accuracy")
    temperature = recompute(lambda row: row["temperature"],
                            summary["temperature"], "ID temperature")
    require(len(accuracy) == len(temperature) == len(numeric_fields) == 17,
            "nine points and eight statistics")
    with metric_path.open(newline="", encoding="utf-8") as stream:
        metrics_csv = list(csv.DictReader(stream))
    with operating_path.open(newline="", encoding="utf-8") as stream:
        operating_csv = list(csv.DictReader(stream))
    require(len(summary["cells"]) == 12 and len(metrics_csv) == 48 and
            len(operating_csv) == 144, "complete 12-method/variant grid")
    metric_position = 0
    operating_position = 0
    display = display_path.read_text(encoding="utf-8")
    def pct(entry):
        return (f"{100*entry['grand_mean']:.2f} ± "
                f"{100*entry['between_partition_means_sample_sd']:.2f}")
    require(f"Classifier ID-test accuracy (%): {pct(summary['id_test_accuracy'])}" in
            display, "displayed classifier accuracy")
    for variant in VARIANTS:
        for method in METHODS:
            cell = summary["cells"][VARIANTS.index(variant)*len(METHODS) +
                                    METHODS.index(method)]
            require(cell["variant"] == variant and cell["method"] == method and
                    set(cell["metrics"]) == set(METRICS) and
                    set(cell["operating"]) ==
                    {f"target{target}" for target in TARGETS},
                    f"cell header {variant}/{method}")
            shown = []
            for metric in METRICS:
                saved = cell["metrics"][metric]
                numbers = recompute(
                    lambda row, metric=metric: row["method_results"][variant][method]
                        ["metrics"][metric],
                    saved, f"{variant}/{method}/{metric}")
                csv_row = metrics_csv[metric_position]
                metric_position += 1
                require((csv_row["variant"], csv_row["method"], csv_row["metric"]) ==
                        (variant, method, metric), "metric CSV ordering")
                for label, found, expected in zip(
                    numeric_fields, csv_numbers(csv_row, numeric_fields), numbers):
                    close(found, expected, f"metric CSV {variant}/{method}/{metric}/{label}")
                shown.append(pct(saved))
            for target in TARGETS:
                require(set(cell["operating"][f"target{target}"]) == set(RATES),
                        f"operating grid {variant}/{method}/{target}")
                for rate in RATES:
                    saved = cell["operating"][f"target{target}"][rate]
                    numbers = recompute(
                        lambda row, target=target, rate=rate: row["method_results"][
                            variant][method]["operating_grid"][f"target{target}"][rate],
                        saved, f"{variant}/{method}/{target}/{rate}")
                    csv_row = operating_csv[operating_position]
                    operating_position += 1
                    require((csv_row["variant"], csv_row["method"],
                             int(csv_row["id_target"]), csv_row["rate"]) ==
                            (variant, method, target, rate), "operating CSV ordering")
                    for label, found, expected in zip(
                        numeric_fields, csv_numbers(csv_row, numeric_fields), numbers):
                        close(found, expected,
                              f"operating CSV {variant}/{method}/{target}/{rate}/{label}")
                    if target == 95 and rate in ("id_test_acceptance",
                                                 "ood_acceptance"):
                        shown.append(pct(saved))
            require("| " + method + " | " + " | ".join(shown) + " |" in display,
                    f"displayed all-method row {variant}/{method}")
    require(metric_position == 48 and operating_position == 144,
            "all CSV rows independently checked")
    outputs = {path.relative_to(ROOT).as_posix(): sha(path)
               for path in (summary_path, metric_path, operating_path,
                            display_path, REPORT / "executed_report_source.py")}
    result = {
        "state": "passed",
        "class_seeds": list(PARTITIONS), "training_seeds": list(SEEDS),
        "nine_score_manifests": 9,
        "metric_csv_rows": metric_position,
        "operating_csv_rows": operating_position,
        "checked_numeric_fields": (metric_position + operating_position + 2) * 17,
        "displayed_method_rows": 12,
        "report_spec_sha256": sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION_REPORT.md"),
        "report_declaration_sha256": sha(BASE / "report-declaration.json"),
        "report_source_sha256": sha(REPORT / "executed_report_source.py"),
        "audit_source_sha256": sha(Path(__file__)),
        "outputs_sha256": outputs,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Independent nine-point, partition/seed statistics, CSV and display audit. Descriptive fixed-partition sensitivity only.",
    }
    shutil.copyfile(__file__, snapshot)
    audit_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    print("passed: nine manifests, 48 metric rows, 144 operating rows, "
          "12 displayed method rows")


if __name__ == "__main__":
    main()
