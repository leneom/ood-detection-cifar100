"""Reconstruct every three-seed class-holdout summary and displayed table row."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
REPORT = ROOT / "report" / "cifar100-class-holdout"
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, reference: float, name: str) -> None:
    require(math.isclose(float(actual), reference, rel_tol=1e-12, abs_tol=1e-12),
            name + f": {actual} versus {reference}")


def verify_statistic(observed: dict, points: list[float], name: str) -> int:
    require(set(observed) == {"per_seed", "mean", "sample_sd"} and
            len(points) == len(observed["per_seed"]) == 3 and
            all(math.isfinite(x) for x in points), name + " three-seed schema")
    for index, (actual, reference) in enumerate(zip(observed["per_seed"], points)):
        close(actual, reference, name + f" seed {index}")
    close(observed["mean"], statistics.fmean(points), name + " mean")
    close(observed["sample_sd"], statistics.stdev(points), name + " sample SD")
    return 5


def pct(points: list[float]) -> str:
    return f"{100 * statistics.fmean(points):.2f} ± {100 * statistics.stdev(points):.2f}"


def main() -> None:
    output = REPORT / "summary-audit.json"
    snapshot = REPORT / "executed_summary_audit_source.py"
    require(not output.exists() and not snapshot.exists(),
            "summary audit already exists; preserve it")
    summary = read(REPORT / "summary.json")
    require(summary["state"] == "summary_pending_independent_audit" and
            summary["seeds"] == list(SEEDS) and
            summary["methods"] == list(METHODS) and
            summary["variants"] == list(VARIANTS) and
            summary["metric_fields"] == list(METRICS) and
            summary["operating_targets"] == list(TARGETS) and
            summary["n_id_test"] == 8000 and
            summary["n_ood_test"] == 2000 and
            summary["ood_prevalence"] == .2 and
            summary["id_optimization_train"] == 36000 and
            summary["id_checkpoint_validation"] == 1600 and
            summary["id_temperature_validation"] ==
            summary["id_threshold_validation"] == 1200 and
            summary["summary_source_sha256"] ==
            sha(ROOT / "tools/build_class_holdout_summary.py") ==
            sha(REPORT / "executed_summary_source.py"),
            "fixed study, source snapshot or full-population metadata differs")
    for name, expected in summary["source_inputs_sha256"].items():
        require(sha(ROOT / name) == expected,
                "summary input changed after aggregation: " + name)
    scores = [read(RUN / f"score-seed{seed}" / "manifest.json")
              for seed in SEEDS]
    audits = [read(RUN / f"score-audit-seed{seed}" / "audit.json")
              for seed in SEEDS]
    training = [read(RUN / f"train-audit-seed{seed}" / "audit.json")
                for seed in SEEDS]
    require(all(audits[i]["state"] == "passed" and
                audits[i]["seed"] == i and
                audits[i]["scoring_manifest_sha256"] ==
                sha(RUN / f"score-seed{i}" / "manifest.json") and
                scores[i]["seed"] == i and
                scores[i]["checkpoint_sha256"] ==
                training[i]["selected_checkpoint_sha256"]
                for i in SEEDS),
            "scored seed missing independent raw/metric audit")
    require(summary["selected_epochs"] == [x["selected_epoch"] for x in training] and
            summary["selected_checkpoint_sha256"] ==
            [x["selected_checkpoint_sha256"] for x in training],
            "ID-only selected epochs or checkpoints differ")
    checks = {"statistical_numbers": 0, "metric_csv_rows": 0,
              "operating_csv_rows": 0, "display_table_rows": 0}
    for field in ("id_test_accuracy", "temperature"):
        points = [float(x[field]) for x in scores]
        checks["statistical_numbers"] += verify_statistic(summary[field], points, field)
    expected_cells = [(variant, method) for variant in VARIANTS for method in METHODS]
    cells = summary["cells"]
    require(len(cells) == len(expected_cells) == 12 and
            [(cell["variant"], cell["method"]) for cell in cells] ==
            expected_cells, "all twelve fixed variant/method cells required in order")
    with (REPORT / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metric_csv = list(csv.DictReader(stream))
    with (REPORT / "operating.csv").open(newline="", encoding="utf-8") as stream:
        operating_csv = list(csv.DictReader(stream))
    require(len(metric_csv) == 12 * 4 and len(operating_csv) == 12 * 4 * 3,
            "complete metric and operating CSV matrices required")
    markdown = (REPORT / "SUMMARY.md").read_text(encoding="utf-8")
    require("8,000 ID and 2,000 held-out-class OOD" in markdown and
            "all six methods" in markdown and
            "not untouched external confirmation" in markdown and
            f"ID-test classifier accuracy: **{pct([float(x['id_test_accuracy']) for x in scores])}%**"
            in markdown,
            "displayed denominator, scope, or ID accuracy differs")
    metric_position = 0
    operating_position = 0
    for cell in cells:
        variant, method = cell["variant"], cell["method"]
        for field in METRICS:
            points = [float(row["method_results"][variant][method]
                            ["metrics"][field]) for row in scores]
            stats = cell["metrics"][field]
            checks["statistical_numbers"] += verify_statistic(
                stats, points, variant + "/" + method + "/" + field)
            csv_row = metric_csv[metric_position]
            require((csv_row["variant"], csv_row["method"], csv_row["metric"]) ==
                    (variant, method, field), "metric CSV row identity/order differs")
            for index in SEEDS:
                close(csv_row[f"seed{index}"], points[index], "metric CSV seed")
            close(csv_row["mean"], statistics.fmean(points), "metric CSV mean")
            close(csv_row["sample_sd"], statistics.stdev(points), "metric CSV SD")
            metric_position += 1
            checks["metric_csv_rows"] += 1
        for target in TARGETS:
            require(set(cell["operating"][f"target{target}"]) == set(RATES),
                    "all three fixed operating rates required")
            for field in RATES:
                points = [float(row["method_results"][variant][method]
                                ["operating_grid"][f"target{target}"][field])
                          for row in scores]
                stats = cell["operating"][f"target{target}"][field]
                checks["statistical_numbers"] += verify_statistic(
                    stats, points,
                    variant + "/" + method + f"/target{target}/" + field)
                csv_row = operating_csv[operating_position]
                require((csv_row["variant"], csv_row["method"],
                         int(csv_row["id_target"]), csv_row["rate"]) ==
                        (variant, method, target, field),
                        "operating CSV row identity/order differs")
                for index in SEEDS:
                    close(csv_row[f"seed{index}"], points[index],
                          "operating CSV seed")
                close(csv_row["mean"], statistics.fmean(points),
                      "operating CSV mean")
                close(csv_row["sample_sd"], statistics.stdev(points),
                      "operating CSV SD")
                operating_position += 1
                checks["operating_csv_rows"] += 1
        metric_points = {
            field: [float(row["method_results"][variant][method]
                          ["metrics"][field]) for row in scores]
            for field in METRICS
        }
        id_accept = [float(row["method_results"][variant][method]
                           ["operating_grid"]["target95"]["id_test_acceptance"])
                     for row in scores]
        ood_accept = [float(row["method_results"][variant][method]
                            ["operating_grid"]["target95"]["ood_acceptance"])
                      for row in scores]
        display = ("| " + method + " | " +
                   " | ".join(pct(metric_points[field]) for field in METRICS) +
                   " | " + pct(id_accept) + " | " + pct(ood_accept) + " |")
        if variant == "T1":
            section = markdown.split("## T=1\n", 1)[1].split(
                "## ID-only fitted temperature\n", 1)[0]
        else:
            section = markdown.split("## ID-only fitted temperature\n", 1)[1]
        require(section.count(display) == 1,
                "displayed six-method table value differs: " + variant + "/" + method)
        checks["display_table_rows"] += 1
    shutil.copyfile(__file__, snapshot)
    result = {
        "state": "passed", "seeds": list(SEEDS), "checked": checks,
        "summary_sha256": sha(REPORT / "summary.json"),
        "metrics_csv_sha256": sha(REPORT / "metrics.csv"),
        "operating_csv_sha256": sha(REPORT / "operating.csv"),
        "summary_markdown_sha256": sha(REPORT / "SUMMARY.md"),
        "summary_source_sha256": sha(REPORT / "executed_summary_source.py"),
        "audit_source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Independent 3-seed aggregation and displayed all-method tables; no OOD-test-based method choice.",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checks}), flush=True)


if __name__ == "__main__":
    main()
