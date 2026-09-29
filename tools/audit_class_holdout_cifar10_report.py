"""Independent complete numerical/display audit of three-seed CIFAR-10 extension."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
OUT = ROOT / "report/class-holdout-cifar10-extension"
BASELINE = ROOT / "report/cifar100-class-holdout/summary.json"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")
TARGETS = (80, 90, 95, 99)
OPERATING = ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(math.isfinite(float(actual)) and
            abs(float(actual) - float(expected)) < 1e-12,
            f"{message}: {actual} vs {expected}")


def average_sd(values: list[float]) -> tuple[float, float]:
    average = math.fsum(values) / 3
    sd = math.sqrt(math.fsum((value - average) ** 2 for value in values) / 2)
    return average, sd


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    output = OUT / "audit.json"
    require(not output.exists(), "write-once summary audit already exists")
    summary = read(OUT / "summary.json")
    baseline = read(BASELINE)
    require(summary["state"] == "report_pending_independent_audit" and
            summary["seeds"] == [0, 1, 2] and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            summary["metrics"] == list(METRICS) and
            summary["targets"] == list(TARGETS) and
            summary["n_id"] == 8000 and summary["n_ood"] == 10000 and
            summary["ood_prevalence"] == 10000 / 18000,
            "complete frozen extension population/schema")
    source_paths = {
        "declaration": RUN / "declaration.json",
        "test_population_audit": RUN / "test-population-audit.json",
        "baseline_summary": BASELINE,
        "report_builder": ROOT / "tools/build_class_holdout_cifar10_report.py",
    }
    records = []
    for seed in range(3):
        source_paths[f"score_seed{seed}"] = RUN / f"score-seed{seed}" / "manifest.json"
        source_paths[f"score_audit_seed{seed}"] = RUN / f"score-audit-seed{seed}.json"
        record = read(source_paths[f"score_seed{seed}"])
        audit = read(source_paths[f"score_audit_seed{seed}"])
        require(audit["state"] == "passed" and audit["seed"] == seed and
                audit["scoring_manifest_sha256"] == sha(source_paths[f"score_seed{seed}"]) and
                audit["checked"] == {"score_vectors": 12, "metric_fields": 156,
                                     "operating_fields": 528,
                                     "ood_mask_vectors": 48,
                                     "frozen_id_mask_vectors": 48},
                "full independent per-seed scoring audit")
        records.append(record)
    require(summary["inputs_sha256"] ==
            {name: sha(path) for name, path in source_paths.items()} and
            summary["csv_sha256"] == {"metrics": sha(OUT / "metrics.csv"),
                                      "operating": sha(OUT / "operating.csv")},
            "all source/report CSV hashes")
    metric_rows, operating_rows = summary["metric_rows"], summary["operating_rows"]
    metric_csv, operating_csv = csv_rows(OUT / "metrics.csv"), csv_rows(OUT / "operating.csv")
    require(len(metric_rows) == len(metric_csv) == 48 and
            len(operating_rows) == len(operating_csv) == 144,
            "48 metric and 144 operating rows required")
    checked = {"metric_seed_values": 0, "metric_statistics": 0,
               "operating_seed_values": 0, "operating_counts": 0,
               "operating_statistics": 0, "csv_rows": 0,
               "display_rows": 0}
    index = 0
    for variant in VARIANTS:
        for method in METHODS:
            for metric in METRICS:
                row, exported = metric_rows[index], metric_csv[index]
                index += 1
                require((row["variant"], row["method"], row["metric"]) ==
                        (variant, method, metric) and
                        set(row) == set(exported) and
                        [exported[k] for k in ("variant", "method", "metric")] ==
                        [variant, method, metric], "ordered metric row/CSV fields")
                values = [records[seed]["method_results"][variant][method]
                          ["metrics"][metric] for seed in range(3)]
                average, sd = average_sd(values)
                for seed, value in enumerate(values):
                    close(row[f"seed{seed}"], value, "metric seed point")
                    close(exported[f"seed{seed}"], value, "metric CSV point")
                    checked["metric_seed_values"] += 1
                for key, value in (("mean", average), ("sample_sd", sd),
                                   ("n_id", 8000), ("n_ood", 10000),
                                   ("ood_prevalence", 10000 / 18000)):
                    close(row[key], value, "metric report stat")
                    close(exported[key], value, "metric CSV stat")
                    checked["metric_statistics"] += 1
                checked["csv_rows"] += 1
    index = 0
    for variant in VARIANTS:
        for method in METHODS:
            for target in TARGETS:
                for population in OPERATING:
                    row, exported = operating_rows[index], operating_csv[index]
                    index += 1
                    require((row["variant"], row["method"], row["target"],
                             row["population"]) ==
                            (variant, method, target, population) and
                            set(row) == set(exported) and
                            [exported[k] for k in ("variant", "method", "population")] ==
                            [variant, method, population] and
                            int(exported["target"]) == target,
                            "ordered operating row/CSV fields")
                    prefix = ("calibration" if population == "calibration_acceptance"
                              else "id_test" if population == "id_test_acceptance"
                              else "ood")
                    denominator = 1200 if prefix == "calibration" else 8000 if prefix == "id_test" else 10000
                    values = []
                    for seed in range(3):
                        source = (records[seed]["method_results"][variant][method]
                                  ["operating_grid"][f"target{target}"])
                        count = source[f"{prefix}_accepted"]
                        value = source[population]
                        require(source[f"{prefix}_count"] == denominator and
                                count / denominator == value,
                                "original integer operating count/ratio")
                        require(int(row[f"accepted{seed}"]) ==
                                int(exported[f"accepted{seed}"]) == count,
                                "reported operating count")
                        close(row[f"seed{seed}"], value, "operating seed point")
                        close(exported[f"seed{seed}"], value, "operating CSV point")
                        values.append(value)
                        checked["operating_seed_values"] += 1
                        checked["operating_counts"] += 1
                    average, sd = average_sd(values)
                    for key, value in (("mean", average), ("sample_sd", sd),
                                       ("denominator", denominator)):
                        close(row[key], value, "operating report stat")
                        close(exported[key], value, "operating CSV stat")
                        checked["operating_statistics"] += 1
                    checked["csv_rows"] += 1
    original = {(cell["variant"], cell["method"]): cell for cell in baseline["cells"]}
    require(len(original) == 12, "all original same-source cells")
    display = (OUT / "REPORT.md").read_text(encoding="utf-8")
    lines = [line for line in display.splitlines() if
             line.startswith("| T1 |") or line.startswith("| calibrated |")]
    require(len(lines) == 12, "all 12 method-variant table rows")
    metric_index = {(row["variant"], row["method"], row["metric"]): row
                    for row in metric_rows}
    operating_index = {(row["variant"], row["method"], row["target"],
                        row["population"]): row for row in operating_rows}
    for variant in VARIANTS:
        for method in METHODS:
            current_a = metric_index[variant, method, "auroc"]
            current_o = operating_index[variant, method, 95, "ood_acceptance"]
            old_a = original[variant, method]["metrics"]["auroc"]
            old_o = original[variant, method]["operating"]["target95"]["ood_acceptance"]
            expected = (
                f"| {variant} | {method} | {100*current_a['mean']:.2f} ± {100*current_a['sample_sd']:.2f}% | "
                f"{100*old_a['mean']:.2f} ± {100*old_a['sample_sd']:.2f}% | "
                f"{100*current_o['mean']:.2f} ± {100*current_o['sample_sd']:.2f}% | "
                f"{100*old_o['mean']:.2f} ± {100*old_o['sample_sd']:.2f}% |"
            )
            require(expected in lines, "display row differs from full audited statistics")
            checked["display_rows"] += 1
    require("55.56%" in display and "20%" in display and
            "must not be directly compared" in display and
            "Exploratory post-hoc" in display and
            "No external confirmation" in display,
            "prevalence, post-hoc and external-confirmation limits")
    audit = {
        "state": "passed", "checked": checked,
        "source_sha256": {"summary": sha(OUT / "summary.json"),
                          "metric_csv": sha(OUT / "metrics.csv"),
                          "operating_csv": sha(OUT / "operating.csv"),
                          "report": sha(OUT / "REPORT.md"),
                          "audit_source": sha(Path(__file__))},
        "boundary": "All seed values/counts/summary statistics and all-method display audited; exploratory population with unequal AUPR prevalence.",
    }
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checked}))


if __name__ == "__main__":
    main()
