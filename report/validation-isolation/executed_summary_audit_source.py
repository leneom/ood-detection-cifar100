"""Independent reconstruction of the predeclared isolation summary from saved results."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "validation-isolation"
REPORT = ROOT / "report" / "validation-isolation"
NATURAL = ("cifar10", "mnist", "svhn", "textures")
POPULATIONS = NATURAL + ("noise",)
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")
OLD_DIR = {"cifar10": "eval-final", "mnist": "eval-final", "noise": "eval-final",
           "svhn": "eval-extension-svhn", "textures": "eval-extension-textures"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def numbers(observed, expected, label):
    check(set(observed) == {"per_seed", "mean", "sample_sd"}, label + " keys")
    check(len(observed["per_seed"]) == len(expected) == 3, label + " seeds")
    for actual, reference in zip(observed["per_seed"], expected):
        check(math.isclose(actual, reference, rel_tol=1e-12, abs_tol=1e-12), label + " point")
    check(math.isclose(observed["mean"], statistics.fmean(expected), rel_tol=1e-12, abs_tol=1e-12),
          label + " mean")
    check(math.isclose(observed["sample_sd"], statistics.stdev(expected), rel_tol=1e-12, abs_tol=1e-12),
          label + " sample SD")


def old_value(old, calibrated, seed, population, variant, method, field):
    if variant == "T1":
        return old[seed][population]["datasets"][population]["methods"][method][field]
    return calibrated[seed]["datasets"][population]["variants"]["calibrated"][method][field]


def main():
    report_path = REPORT / "summary.json"
    summary = read(report_path)
    check(summary["state"] == "summary_pending_independent_report_audit", "summary state")
    check(summary["seeds"] == [0, 1, 2], "training seeds")
    check(tuple(summary["methods"]) == METHODS and tuple(summary["variants"]) == VARIANTS,
          "primary methods")
    check(tuple(summary["natural_datasets"]) == NATURAL and summary["diagnostic_dataset"] == "noise",
          "natural and diagnostic populations")
    check(sha(REPORT / "executed_summary_source.py") == summary["summary_source_sha256"] ==
          sha(ROOT / "tools/build_isolated_summary.py"), "summary producer source")
    for name, expected in summary["source_inputs_sha256"].items():
        check(sha(ROOT / name) == expected, "summary source input changed: " + name)

    new = [read(PHASE / f"score-seed{s}" / "manifest.json") for s in range(3)]
    old = [{p: read(ROOT / "runs" / f"cifar100-seed{s}" / directory / "results.json")
            for p, directory in OLD_DIR.items()} for s in range(3)]
    calibrated = [read(ROOT / "runs" / "calibration-phase2" / f"seed{s}" / "results.json")
                  for s in range(3)]
    train = [read(PHASE / f"train-audit-seed{s}" / "audit.json") for s in range(3)]
    check(summary["selected_epochs_old"] == [r["old_selected_epoch"] for r in train] and
          summary["selected_epochs_new"] == [r["new_selected_epoch"] for r in train],
          "selection epochs")
    check(summary["checkpoint_validation_count_old_new"] == [5000, 2000] and
          summary["temperature_fit_count_old_new"] == [2500, 1500] and
          summary["threshold_fit_count_T1_old_new"] == [5000, 1500] and
          summary["threshold_fit_count_calibrated_old_new"] == [2500, 1500],
          "comparison sample sizes")
    for p in POPULATIONS:
        expected = {"new_id_test": [row["id_test_count"] for row in new],
                    "new_ood": [row["datasets"][p]["n_ood"] for row in new],
                    "old_id_test": [old[s][p]["datasets"][p]["methods"]["msp"]["n_id"]
                                    for s in range(3)],
                    "old_ood": [old[s][p]["datasets"][p]["methods"]["msp"]["n_ood"]
                                for s in range(3)]}
        check(summary["population_counts"][p] == expected, p + " counts")
    for field, latest, previous in (
        ("id_test_accuracy", [x["id_test_accuracy"] for x in new],
         [old[s]["cifar10"]["id_test_accuracy"] for s in range(3)]),
        ("temperature", [x["temperature"] for x in new],
         [x["temperature_fit"]["temperature"] for x in calibrated])):
        numbers(summary[field]["new"], latest, field + " new")
        numbers(summary[field]["old"], previous, field + " old")
        numbers(summary[field]["paired_new_minus_old"],
                [a - b for a, b in zip(latest, previous)], field + " paired")

    cells = {(row["variant"], row["method"], row["population"]): row
             for row in summary["cells"]}
    expected_keys = {(v, m, p) for v in VARIANTS for m in METHODS
                     for p in POPULATIONS + ("natural_macro",)}
    check(len(summary["cells"]) == len(cells) == len(expected_keys) == 72 and
          set(cells) == expected_keys, "complete unique primary grid")
    checked = 0
    for (variant, method, population), row in cells.items():
        members = NATURAL if population == "natural_macro" else (population,)
        check(row["natural_macro"] == (population == "natural_macro"), "macro label")
        for field in METRICS:
            latest = [statistics.fmean(new[s]["datasets"][p]["methods"][variant][method]
                          ["metrics"][field] for p in members) for s in range(3)]
            previous = [statistics.fmean(old_value(old, calibrated, s, p, variant, method, field)
                            for p in members) for s in range(3)]
            result = row["metrics"][field]
            numbers(result["new"], latest, "metric new")
            numbers(result["old"], previous, "metric old")
            numbers(result["paired_new_minus_old"],
                    [a-b for a, b in zip(latest, previous)], "metric paired")
            checked += 1
        check(set(row["operating"]) == {"target80", "target90", "target95", "target99"},
              "four fixed operating targets")
        for target in (80, 90, 95, 99):
            grid = row["operating"][f"target{target}"]
            for field in ("calibration_acceptance", "id_test_acceptance", "ood_acceptance"):
                expected = [statistics.fmean(
                    new[s]["datasets"][p]["methods"][variant][method]
                    ["operating_grid"][f"target{target}"][field] for p in members)
                    for s in range(3)]
                numbers(grid[field], expected, "operating rate")
                checked += 1
        latest = [statistics.fmean(
            new[s]["datasets"][p]["methods"][variant][method]
            ["operating_grid"]["target95"]["ood_acceptance"] for p in members)
            for s in range(3)]
        previous = [statistics.fmean(old_value(
            old, calibrated, s, p, variant, method,
            "ood_acceptance_at_validation_threshold") for p in members)
            for s in range(3)]
        numbers(row["operating"]["target95"]["old_ood_acceptance"], previous,
                "old operating rate")
        numbers(row["operating"]["target95"]["paired_new_minus_old_ood_acceptance"],
                [a-b for a, b in zip(latest, previous)], "paired operating rate")
        checked += 2
    output = REPORT / "summary-audit.json"
    check(not output.exists(), "summary audit already exists")
    result = {"state": "passed", "summary_sha256": sha(report_path),
              "summary_source_sha256": summary["summary_source_sha256"],
              "audit_source_sha256": sha(__file__), "cells": len(cells),
              "statistical_fields_checked": checked,
              "boundary": "Independent saved-result reconstruction, not OOD-test-based selection."}
    shutil.copyfile(__file__, REPORT / "executed_summary_audit_source.py")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

