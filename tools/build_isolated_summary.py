"""Fixed paired old/new summary after all isolation scores pass independent audits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("three finite training-seed observations required")
    return {"per_seed": values.tolist(), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_inputs():
    status_path = PHASE / "evaluation-status.json"
    require(status_path.is_file(), "evaluation has not started; no summary can be built")
    phase = read(status_path)
    require(phase["state"] == "all_scored_and_audited_pending_summary" and
            phase["completed_seeds"] == [0, 1, 2], "all evaluation stages must pass first")
    require(not REPORT.exists(), "summary report output already exists")
    inputs = {}
    new, prior, phase2 = [], [], []
    for seed in range(3):
        scored_path = PHASE / f"score-seed{seed}" / "manifest.json"
        score_audit_path = PHASE / f"audit-seed{seed}" / "audit.json"
        feature_audit_path = PHASE / f"feature-audit-seed{seed}" / "audit.json"
        training_audit_path = PHASE / f"train-audit-seed{seed}" / "audit.json"
        for path in (scored_path, score_audit_path, feature_audit_path, training_audit_path):
            inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
        score = read(scored_path)
        require(score["state"] == "scored_pending_independent_audit" and score["seed"] == seed,
                "scored seed differs")
        for path in (score_audit_path, feature_audit_path, training_audit_path):
            audit = read(path)
            require(audit["state"] == "passed" and audit["seed"] == seed,
                    "independent per-seed audit did not pass")
        require(read(score_audit_path)["scoring_manifest_sha256"] == digest(scored_path),
                "score audit does not bind scored source")
        require(read(feature_audit_path)["extraction_manifest_sha256"] ==
                score["extraction_manifest_sha256"], "feature audit does not bind extraction")
        require(read(training_audit_path)["selected_checkpoint_sha256"] == score["checkpoint_sha256"],
                "training checkpoint differs from score input")
        new.append(score)
        previous = {}
        for population, directory in OLD_DIR.items():
            path = ROOT / "runs" / f"cifar100-seed{seed}" / directory / "results.json"
            inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
            original = read(path)
            require(original["training_config"]["seed"] == seed and original["temperature"] == 1.,
                    "old T=1 baseline source differs")
            previous[population] = original
        prior.append(previous)
        path = ROOT / "runs" / "calibration-phase2" / f"seed{seed}" / "results.json"
        inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
        calibrate = read(path)
        require(calibrate["seed"] == seed and set(calibrate["datasets"]) == set(POPULATIONS),
                "old ID-only calibrated baseline differs")
        phase2.append(calibrate)
    for name in ("docs/VALIDATION_ISOLATION.md", "runs/validation-isolation/partition.npz",
                 "runs/validation-isolation/training-launch.json",
                 "runs/validation-isolation/evaluation-launch.json",
                 "runs/validation-isolation/evaluation-status.json"):
        inputs[name] = digest(ROOT / name)
    return new, prior, phase2, inputs


def old_metric(prior, phase2, seed, population, variant, method, metric):
    if variant == "T1":
        return prior[seed][population]["datasets"][population]["methods"][method][metric]
    return phase2[seed]["datasets"][population]["variants"]["calibrated"][method][metric]


def main():
    new, prior, phase2, inputs = checked_inputs()
    cells = []
    for variant in VARIANTS:
        for method in METHODS:
            for population in POPULATIONS + ("natural_macro",):
                metric_rows = {}
                for metric in METRICS:
                    if population == "natural_macro":
                        current = [np.mean([new[seed]["datasets"][p]["methods"][variant][method]
                                            ["metrics"][metric] for p in NATURAL]) for seed in range(3)]
                        old = [np.mean([old_metric(prior, phase2, seed, p, variant, method, metric)
                                        for p in NATURAL]) for seed in range(3)]
                    else:
                        current = [new[seed]["datasets"][population]["methods"][variant][method]
                                   ["metrics"][metric] for seed in range(3)]
                        old = [old_metric(prior, phase2, seed, population, variant, method, metric)
                               for seed in range(3)]
                    metric_rows[metric] = {"new": describe(current), "old": describe(old),
                                           "paired_new_minus_old": describe(np.asarray(current) - np.asarray(old))}
                operational = {}
                members = NATURAL if population == "natural_macro" else (population,)
                for target in (80, 90, 95, 99):
                    key = f"target{target}"
                    operational[key] = {
                        field: describe([
                            np.mean([new[seed]["datasets"][p]["methods"][variant][method]
                                     ["operating_grid"][key][field] for p in members])
                            for seed in range(3)])
                        for field in ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")}
                old95 = [np.mean([old_metric(prior, phase2, seed, p, variant, method,
                                             "ood_acceptance_at_validation_threshold")
                                      for p in members]) for seed in range(3)]
                new95 = [np.mean([new[seed]["datasets"][p]["methods"][variant][method]
                                   ["operating_grid"]["target95"]["ood_acceptance"]
                                    for p in members]) for seed in range(3)]
                operational["target95"]["old_ood_acceptance"] = describe(old95)
                operational["target95"]["paired_new_minus_old_ood_acceptance"] = describe(
                    np.asarray(new95) - np.asarray(old95))
                cells.append({"variant": variant, "method": method, "population": population,
                              "natural_macro": population == "natural_macro", "metrics": metric_rows,
                              "operating": operational})
    selected = [read(PHASE / f"train-audit-seed{seed}" / "audit.json") for seed in range(3)]
    old_accuracy = [prior[s]["cifar10"]["id_test_accuracy"] for s in range(3)]
    new_accuracy = [row["id_test_accuracy"] for row in new]
    old_temperature = [row["temperature_fit"]["temperature"] for row in phase2]
    new_temperature = [row["temperature"] for row in new]
    summary = {"state": "summary_pending_independent_report_audit", "seeds": [0, 1, 2],
               "methods": METHODS, "variants": VARIANTS, "natural_datasets": NATURAL,
               "diagnostic_dataset": "noise", "checkpoint_validation_count_old_new": [5000, 2000],
               "temperature_fit_count_old_new": [2500, 1500],
               "threshold_fit_count_T1_old_new": [5000, 1500],
               "threshold_fit_count_calibrated_old_new": [2500, 1500],
               "selected_epochs_old": [row["old_selected_epoch"] for row in selected],
               "selected_epochs_new": [row["new_selected_epoch"] for row in selected],
               "population_counts": {p: {"new_id_test": [row["id_test_count"] for row in new],
                                           "new_ood": [row["datasets"][p]["n_ood"] for row in new],
                                           "old_id_test": [prior[s][p]["datasets"][p]["methods"]["msp"]["n_id"]
                                                           for s in range(3)],
                                           "old_ood": [prior[s][p]["datasets"][p]["methods"]["msp"]["n_ood"]
                                                       for s in range(3)]} for p in POPULATIONS},
               "id_test_accuracy": {"new": describe(new_accuracy), "old": describe(old_accuracy),
                                    "paired_new_minus_old": describe(np.asarray(new_accuracy) - np.asarray(old_accuracy))},
               "temperature": {"new": describe(new_temperature), "old": describe(old_temperature),
                               "paired_new_minus_old": describe(np.asarray(new_temperature) - np.asarray(old_temperature))},
               "cells": cells, "source_inputs_sha256": inputs,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "boundary": "Exploratory paired three-training-seed description on previously inspected OOD tests; no winner, causal, or fresh confirmation claim."}
    REPORT.mkdir(parents=True)
    shutil.copyfile(__file__, REPORT / "executed_summary_source.py")
    summary["summary_source_sha256"] = digest(__file__)
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"state": summary["state"], "cells": len(cells),
                      "selected_epochs_new": summary["selected_epochs_new"]}), flush=True)


if __name__ == "__main__":
    main()
