"""Build the complete three-seed 80/20 CIFAR-100 class-holdout report.

All six fixed methods and both temperature variants are reported. No method
is selected using held-out-class test results.
"""
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
        raise ValueError(message)


def describe(values: list[float]) -> dict:
    require(len(values) == 3 and all(math.isfinite(x) for x in values),
            "three finite seed observations required")
    return {"per_seed": values,
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values)}


def checked_inputs() -> tuple[list[dict], list[dict], dict[str, str]]:
    phase = read(RUN / "status.json")
    require(phase["state"] == "training_complete_pending_independent_audit" and
            phase["trainer_running"] is False and
            [row["seed"] for row in phase["completed_seeds"]] == list(SEEDS),
            "all three 100-epoch schedules must finish before summary")
    require(not REPORT.exists(), "summary report exists; preserve it")
    scored, training, inputs = [], [], {}
    expected_source = {
        "train": ROOT / "tools/audit_class_holdout_training.py",
        "features": ROOT / "tools/audit_class_holdout_features.py",
        "scores": ROOT / "tools/audit_class_holdout_scores.py",
    }
    for seed in SEEDS:
        paths = {
            "train": RUN / f"train-audit-seed{seed}" / "audit.json",
            "inference": RUN / f"checkpoint-inference-audit-seed{seed}" / "audit.json",
            "inference_logits": RUN / f"checkpoint-inference-audit-seed{seed}" /
                                "checkpoint-validation.npz",
            "features": RUN / f"feature-audit-seed{seed}" / "audit.json",
            "scores": RUN / f"score-audit-seed{seed}" / "audit.json",
            "score_manifest": RUN / f"score-seed{seed}" / "manifest.json",
            "score_status": RUN / f"score-seed{seed}" / "status.json",
            "raw_manifest": RUN / f"extract-seed{seed}" / "manifest.json",
            "checkpoint": RUN / f"seed{seed}" / "final-best.pt",
        }
        if seed == 2:
            paths["inference_cuda_logits"] = (
                RUN / f"checkpoint-inference-audit-seed{seed}" /
                "checkpoint-validation-cuda.npz")
        for path in paths.values():
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        t = read(paths["train"])
        i = read(paths["inference"])
        f = read(paths["features"])
        a = read(paths["scores"])
        score = read(paths["score_manifest"])
        status = read(paths["score_status"])
        raw = read(paths["raw_manifest"])
        require(all(record["state"] == "passed" and record["seed"] == seed
                    for record in (t, i, f, a)) and
                t["training_epoch_count"] == 100 and
                t["selected_checkpoint_sha256"] ==
                i["selected_checkpoint_sha256"] ==
                score["checkpoint_sha256"] == raw["checkpoint_sha256"] ==
                sha(paths["checkpoint"]) and
                i["training_audit_sha256"] == sha(paths["train"]) and
                i["raw_id_validation_sha256"] ==
                sha(paths["inference_logits"]) and
                (seed != 2 or (
                    i["cuda_raw_id_validation_sha256"] ==
                    sha(paths["inference_cuda_logits"]) and
                    i["cpu_cuda_prediction_disagreements"] == 1 and
                    i["cpu_recomputed_correct_count"] ==
                    i["recomputed_correct_count"] + 1)) and
                f["extraction_manifest_sha256"] ==
                a["extraction_manifest_sha256"] ==
                score["extraction_manifest_sha256"] == sha(paths["raw_manifest"]) and
                a["feature_audit_sha256"] ==
                score["feature_audit_sha256"] == sha(paths["features"]) and
                a["scoring_manifest_sha256"] == sha(paths["score_manifest"]) and
                status["state"] == score["state"] ==
                "scored_pending_independent_audit" and
                status["manifest_sha256"] == sha(paths["score_manifest"]) and
                score["seed"] == seed and
                score["methods"] == list(METHODS) and
                score["targets"] == [.80, .90, .95, .99] and
                score["n_id"] == 8000 and score["n_ood"] == 2000 and
                score["ood_prevalence"] == .2 and
                score["partition_sha256"] == sha(RUN / "partition.npz") and
                score["test_population_sha256"] == sha(RUN / "test-population.npz") and
                t["audit_source_sha256"] == sha(expected_source["train"]) and
                i["source_sha256"] == sha(ROOT / (
                    "tools/audit_class_holdout_checkpoint_cross_device.py"
                    if seed == 2 else
                    "tools/audit_class_holdout_checkpoint_inference.py")) and
                f["source_sha256"] == sha(expected_source["features"]) and
                a["source_sha256"] == sha(expected_source["scores"]),
                f"seed {seed} source-bound audit or full population differs")
        scored.append(score)
        training.append(t)
    for path in (RUN / "partition.npz", RUN / "partition-audit.json",
                 RUN / "test-population.npz", RUN / "test-population-audit.json",
                 RUN / "training-launch.json",
                 ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    return scored, training, inputs


def pct(row: dict) -> str:
    return f"{100 * row['mean']:.2f} ± {100 * row['sample_sd']:.2f}"


def main() -> None:
    scored, training, inputs = checked_inputs()
    cells = []
    for variant in VARIANTS:
        for method in METHODS:
            metrics = {
                field: describe([
                    row["method_results"][variant][method]["metrics"][field]
                    for row in scored]) for field in METRICS
            }
            operating = {
                f"target{target}": {
                    field: describe([
                        row["method_results"][variant][method]["operating_grid"]
                           [f"target{target}"][field] for row in scored])
                    for field in RATES
                } for target in TARGETS
            }
            cells.append({"variant": variant, "method": method,
                          "metrics": metrics, "operating": operating})
    summary = {
        "state": "summary_pending_independent_audit", "seeds": list(SEEDS),
        "study": "CIFAR-100 80-ID/20-held-out-fine-class, source-matched",
        "methods": METHODS, "variants": VARIANTS,
        "metric_fields": METRICS, "operating_targets": TARGETS,
        "n_id_test": 8000, "n_ood_test": 2000, "ood_prevalence": .2,
        "id_optimization_train": 36000,
        "id_checkpoint_validation": 1600,
        "id_temperature_validation": 1200,
        "id_threshold_validation": 1200,
        "selected_epochs": [row["selected_epoch"] for row in training],
        "selected_checkpoint_sha256": [row["selected_checkpoint_sha256"]
                                       for row in training],
        "id_test_accuracy": describe([row["id_test_accuracy"] for row in scored]),
        "temperature": describe([row["temperature"] for row in scored]),
        "cells": cells, "source_inputs_sha256": inputs,
        "summary_source_sha256": sha(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Complete three-seed description of one predeclared class split; source-matched but CIFAR-100 was previously used elsewhere in the project. No test-selected method or cross-study causal claim.",
    }
    REPORT.mkdir(parents=True)
    shutil.copyfile(__file__, REPORT / "executed_summary_source.py")
    json_path = REPORT / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    metrics_path = REPORT / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "method", "metric", "seed0", "seed1", "seed2",
                         "mean", "sample_sd"])
        for cell in cells:
            for field in METRICS:
                row = cell["metrics"][field]
                writer.writerow([cell["variant"], cell["method"], field,
                                 *row["per_seed"], row["mean"], row["sample_sd"]])
    operating_path = REPORT / "operating.csv"
    with operating_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "method", "id_target", "rate", "seed0", "seed1",
                         "seed2", "mean", "sample_sd"])
        for cell in cells:
            for target in TARGETS:
                for field in RATES:
                    row = cell["operating"][f"target{target}"][field]
                    writer.writerow([cell["variant"], cell["method"], target, field,
                                     *row["per_seed"], row["mean"], row["sample_sd"]])
    lines = [
        "# CIFAR-100 class-holdout detection — three-seed report",
        "",
        "One fine class in each of 20 coarse groups was held out before training. Each fresh 80-class ResNet-18 used only 36,000 ID optimization images; the 1,600/1,200/1,200 ID-validation roles respectively selected the checkpoint, fitted temperature, and fixed acceptance thresholds. The official test set contributes all 8,000 ID and 2,000 held-out-class OOD images in original order. Each detector is fixed in advance. Values below are mean ± sample SD over training seeds 0, 1, 2; all six methods are shown without a test-based winner.",
        "",
        f"ID-test classifier accuracy: **{pct(summary['id_test_accuracy'])}%**. Selected epochs: {summary['selected_epochs']}.",
        "",
    ]
    for variant in VARIANTS:
        label = "T=1" if variant == "T1" else "ID-only fitted temperature"
        lines.extend([f"## {label}", "",
                      "| Method | AUROC (%) | AUPR-out (%) | AUPR-in (%) | FPR95-ID (%) | ID accepted at 95% ID-validation target (%) | OOD accepted at same threshold (%) |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for cell in cells:
            if cell["variant"] != variant:
                continue
            rates = cell["operating"]["target95"]
            lines.append("| " + cell["method"] + " | " + " | ".join(
                pct(cell["metrics"][field]) for field in METRICS) + " | " +
                pct(rates["id_test_acceptance"]) + " | " +
                pct(rates["ood_acceptance"]) + " |")
        lines.append("")
    lines.extend([
        "FPR95-ID above uses an ID **test** order statistic for the benchmark; the two acceptance columns use the separate frozen ID-validation threshold. AUPR uses the saved 8,000:2,000 population ratio. The calibrated variant changes MSP, Energy and Entropy; MaxLogit and both feature scores remain unchanged by temperature.",
        "",
        "The images share the official CIFAR-100 source and 32×32 format, but this experiment also changes the trained classifier and ID label space relative to earlier 100-class runs. Earlier CIFAR-100 images were used in this project, so this is not untouched external confirmation. These observations do not identify a causal effect of source matching or justify choosing a method from this test set.",
        "",
        "Machine-readable per-seed values: `summary.json`, `metrics.csv`, `operating.csv`; original logits, features, scores, masks, checkpoints and independent audits remain in `runs/cifar100-class-holdout/`.",
    ])
    (REPORT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"state": summary["state"], "seeds": list(SEEDS),
                      "cells": len(cells)}), flush=True)


if __name__ == "__main__":
    main()
