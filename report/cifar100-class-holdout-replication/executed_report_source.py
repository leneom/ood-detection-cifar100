"""Build the predeclared nine-model class-partition sensitivity report.

This reads scores only after every new model and its independent audits pass.
The resulting report is pending a separate numerical/display audit.
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
        raise ValueError(why)


def bind(inputs: dict[str, str], path: Path) -> str:
    digest = sha(path)
    inputs[path.relative_to(ROOT).as_posix()] = digest
    return digest


def checked_inputs() -> tuple[dict[int, list[dict]], dict[str, str]]:
    require(not REPORT.exists(), "write-once report already exists")
    declaration_path = BASE / "report-declaration.json"
    declaration = read(declaration_path)
    require(declaration["state"] == "frozen_before_new_partition_ood_test_scores" and
            declaration["class_seeds"] == list(PARTITIONS) and
            declaration["training_seeds"] == list(SEEDS) and
            declaration["variants"] == list(VARIANTS) and
            declaration["methods"] == list(METHODS) and
            declaration["metrics"] == list(METRICS) and
            declaration["id_threshold_targets"] == list(TARGETS) and
            declaration["operating_rates"] == list(RATES),
            "frozen report grid differs")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"frozen input changed: {name}")
    status = read(BASE / "status.json")
    require(status["state"] == "training_complete_pending_independent_audit" and
            status["trainer_running"] is False and
            {(r["class_seed"], r["training_seed"]) for r in status["completed"]} ==
            {(p, s) for p in (2032, 2033) for s in SEEDS} and
            all(r["epochs"] == 100 for r in status["completed"]),
            "all six full training schedules must complete")
    inputs: dict[str, str] = {}
    for path in (declaration_path, BASE / "declaration.json",
                 BASE / "training-launch.json", BASE / "status.json",
                 ROOT / "docs/CLASS_HOLDOUT_REPLICATION_REPORT.md",
                 ROOT / "docs/PROTOCOL.md"):
        bind(inputs, path)
    # Audit all six new pipelines before opening any score manifest.
    for p in (2032, 2033):
        split = BASE / f"split{p}"
        partition = split / "partition.npz"
        partition_audit = split / "partition-audit.json"
        population = split / "test-population.npz"
        population_audit = split / "test-population-audit.json"
        pa, ta = read(partition_audit), read(population_audit)
        require(pa["state"] == ta["state"] == "passed" and
                pa["class_rng_seed"] == ta["class_rng_seed"] == p and
                pa["partition_sha256"] == ta["partition_sha256"] == sha(partition) and
                ta["test_population_sha256"] == sha(population) and
                pa["source_sha256"] == sha(ROOT / "tools/audit_class_holdout_replication_partition.py") and
                ta["audit_source_sha256"] == sha(ROOT / "tools/audit_class_holdout_replication_test.py"),
                f"split {p} partition and test population audit")
        for path in (partition, partition_audit, split / "partition.json",
                     population, split / "test-population.json", population_audit):
            bind(inputs, path)
        for seed in SEEDS:
            train_path = split / f"train-audit-seed{seed}/audit.json"
            executed_train_source = split / f"train-audit-seed{seed}/executed_audit_source.py"
            infer_path = split / f"checkpoint-inference-audit-seed{seed}/audit.json"
            feature_path = split / f"feature-audit-seed{seed}/audit.json"
            score_path = split / f"score-audit-seed{seed}/audit.json"
            raw_manifest = split / f"extract-seed{seed}/manifest.json"
            scoring_manifest = split / f"score-seed{seed}/manifest.json"
            checkpoint = split / f"seed{seed}/final-best.pt"
            train, infer, feature, audit = map(read, (
                train_path, infer_path, feature_path, score_path))
            # Seed 2032/0 ran before the documented controller-resume fix.
            # Bind its immutable executed source; later audits used the amended source.
            if (p, seed) == (2032, 0):
                amendment_path = BASE / "controller-amendment.json"
                amendment = read(amendment_path)
                require(amendment["state"] == "controller_resume_key_fix" and
                        amendment["completed_before_amendment"] == [{
                            "class_seed": p,
                            "training_seed": seed,
                            "epochs": 100,
                            "selected_checkpoint_sha256": sha(checkpoint),
                            "training_audit_sha256": sha(train_path),
                        }] and
                        train["audit_source_sha256"] == sha(executed_train_source),
                        "original pre-amendment training audit provenance")
                bind(inputs, amendment_path)
            else:
                require(train["audit_source_sha256"] ==
                        sha(ROOT / "tools/audit_class_holdout_replication_training.py") and
                        train["audit_source_sha256"] == sha(executed_train_source),
                        f"split {p} seed {seed} executed training auditor")
            require(train["state"] == infer["state"] == feature["state"] ==
                    audit["state"] == "passed" and
                    train["class_seed"] == infer["class_seed"] == p and
                    train["seed"] == infer["seed"] == feature["seed"] ==
                    audit["seed"] == seed and
                    train["training_epoch_count"] == 100 and
                    train["selected_checkpoint_sha256"] ==
                    infer["selected_checkpoint_sha256"] == sha(checkpoint) and
                    infer["training_audit_sha256"] == sha(train_path) and
                    infer["recomputed_accuracy"] ==
                    train["selected_id_checkpoint_accuracy"] and
                    infer["source_sha256"] == sha(ROOT / "tools/finalize_class_holdout_replication_checkpoint.py") and
                    feature["source_sha256"] == sha(ROOT / "tools/audit_class_holdout_features.py") and
                    audit["source_sha256"] == sha(ROOT / "tools/audit_class_holdout_scores.py") and
                    feature["extraction_manifest_sha256"] ==
                    audit["extraction_manifest_sha256"] == sha(raw_manifest) and
                    audit["feature_audit_sha256"] == sha(feature_path) and
                    audit["scoring_manifest_sha256"] == sha(scoring_manifest),
                    f"split {p} seed {seed} independent training/inference/raw/score audit")
            contexts = (
                split / f"extract-seed{seed}/class-provenance.json",
                split / f"feature-audit-seed{seed}/class-provenance.json",
                split / f"score-seed{seed}/class-provenance.json",
                split / f"score-audit-seed{seed}/class-provenance.json",
            )
            extract_context, feature_context, scoring_context, audit_context = map(
                read, contexts)
            require(extract_context["class_rng_seed"] ==
                    feature_context["class_rng_seed"] ==
                    scoring_context["class_rng_seed"] ==
                    audit_context["class_rng_seed"] == p and
                    extract_context["training_seed"] ==
                    feature_context["training_seed"] ==
                    scoring_context["training_seed"] ==
                    audit_context["training_seed"] == seed and
                    extract_context["source_sha256"] == sha(ROOT / "tools/extract_class_holdout_replication.py") and
                    feature_context["source_sha256"] == sha(ROOT / "tools/audit_class_holdout_replication_features.py") and
                    scoring_context["source_sha256"] == sha(ROOT / "tools/score_class_holdout_replication.py") and
                    audit_context["source_sha256"] == sha(ROOT / "tools/audit_class_holdout_replication_scores.py") and
                    extract_context["extraction_manifest_sha256"] == sha(raw_manifest) and
                    feature_context["feature_audit_sha256"] == sha(feature_path) and
                    scoring_context["score_manifest_sha256"] == sha(scoring_manifest) and
                    audit_context["score_audit_sha256"] == sha(score_path),
                    f"split {p} seed {seed} class provenance")
            for stage, manifest_path in (("extract", raw_manifest),
                                         ("score", scoring_manifest)):
                stage_folder = split / f"{stage}-seed{seed}"
                stage_manifest = read(manifest_path)
                stage_status = read(stage_folder / "status.json")
                require(stage_status["state"] == stage_manifest["state"] and
                        stage_status["manifest_sha256"] == sha(manifest_path),
                        f"split {p} seed {seed} {stage} manifest status")
                for filename, expected in stage_manifest["outputs"].items():
                    require(sha(stage_folder / filename) == expected,
                            f"split {p} seed {seed} {stage} raw output {filename}")
                for name, expected in stage_manifest["source_hashes"].items():
                    require(sha(ROOT / name) == expected,
                            f"split {p} seed {seed} {stage} source {name}")
                bind(inputs, stage_folder / "status.json")
            for path in (executed_train_source, train_path, infer_path, feature_path, score_path,
                         raw_manifest, checkpoint, *contexts):
                bind(inputs, path)
    # This read-only original verifier checks all 160 baseline files and source hashes.
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.verify_class_holdout_completion import validate
    old_completion = read(ROOT / "report/cifar100-class-holdout/completion.json")
    require(old_completion["state"] == "complete" and
            old_completion["inputs_sha256"] == validate() and
            len(old_completion["inputs_sha256"]) == 160,
            "original 160-file completion gate")
    bind(inputs, ROOT / "report/cifar100-class-holdout/completion.json")
    scores: dict[int, list[dict]] = {}
    for p in PARTITIONS:
        run = OLD if p == 2031 else BASE / f"split{p}"
        scores[p] = []
        for seed in SEEDS:
            manifest_path = run / f"score-seed{seed}/manifest.json"
            score = read(manifest_path)
            require(score["state"] == "scored_pending_independent_audit" and
                    score["seed"] == seed and score["methods"] == list(METHODS) and
                    score["targets"] == [.8, .9, .95, .99] and
                    score["n_id"] == 8000 and score["n_ood"] == 2000 and
                    score["ood_prevalence"] == .2 and
                    score["temperature_fit_role"] == "ID_temperature_1200" and
                    score["threshold_fit_role"] == "ID_threshold_1200" and
                    score["checkpoint_sha256"] ==
                    sha(run / f"seed{seed}/final-best.pt"),
                    f"split {p} seed {seed} complete fixed score grid")
            bind(inputs, manifest_path)
            scores[p].append(score)
    return scores, inputs


def describe_by_partition(values: dict[int, list[float]]) -> dict:
    require(set(values) == set(PARTITIONS) and all(
        len(v) == 3 and all(math.isfinite(x) for x in v)
        for v in values.values()), "nine finite model observations")
    partitions = {
        str(p): {"per_seed": values[p],
                 "mean": statistics.fmean(values[p]),
                 "sample_sd": statistics.stdev(values[p])}
        for p in PARTITIONS
    }
    means = [partitions[str(p)]["mean"] for p in PARTITIONS]
    return {"partitions": partitions, "grand_mean": statistics.fmean(means),
            "between_partition_means_sample_sd": statistics.stdev(means)}


def main() -> None:
    scores, inputs = checked_inputs()
    def field(getter):
        return describe_by_partition({
            p: [float(getter(score)) for score in scores[p]] for p in PARTITIONS})
    cells = []
    for variant in VARIANTS:
        for method in METHODS:
            metrics = {key: field(lambda score, key=key: score[
                "method_results"][variant][method]["metrics"][key])
                       for key in METRICS}
            operating = {
                f"target{target}": {
                    rate: field(lambda score, target=target, rate=rate: score[
                        "method_results"][variant][method]["operating_grid"][
                            f"target{target}"][rate])
                    for rate in RATES}
                for target in TARGETS}
            cells.append({"variant": variant, "method": method,
                          "metrics": metrics, "operating": operating})
    report = {
        "state": "report_pending_independent_audit",
        "class_seeds": list(PARTITIONS), "training_seeds": list(SEEDS),
        "methods": list(METHODS), "variants": list(VARIANTS),
        "metric_fields": list(METRICS), "targets": list(TARGETS),
        "operating_rates": list(RATES),
        "n_id_test_per_partition": 8000, "n_ood_test_per_partition": 2000,
        "ood_prevalence": .2,
        "id_test_accuracy": field(lambda score: score["id_test_accuracy"]),
        "temperature": field(lambda score: score["temperature"]),
        "cells": cells, "source_inputs_sha256": inputs,
        "source_sha256": sha(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Descriptive sensitivity of three fixed 80/20 class partitions; original 2031 results were previously viewed. No OOD-based selection or independent partition CI.",
    }
    REPORT.mkdir(parents=True)
    shutil.copyfile(__file__, REPORT / "executed_report_source.py")
    (REPORT / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    points = [f"class{p}_seed{s}" for p in PARTITIONS for s in SEEDS]
    stats = [f"class{p}_mean" for p in PARTITIONS] + [
        f"class{p}_within_sd" for p in PARTITIONS] + [
        "grand_mean", "between_partition_means_sd"]
    def numeric_row(row):
        return [row["partitions"][str(p)]["per_seed"][s]
                for p in PARTITIONS for s in SEEDS] + [
                row["partitions"][str(p)]["mean"] for p in PARTITIONS] + [
                row["partitions"][str(p)]["sample_sd"] for p in PARTITIONS] + [
                row["grand_mean"], row["between_partition_means_sample_sd"]]
    with (REPORT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "method", "metric", *points, *stats])
        for cell in cells:
            for metric in METRICS:
                writer.writerow([cell["variant"], cell["method"], metric,
                                 *numeric_row(cell["metrics"][metric])])
    with (REPORT / "operating.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "method", "id_target", "rate", *points, *stats])
        for cell in cells:
            for target in TARGETS:
                for rate in RATES:
                    writer.writerow([cell["variant"], cell["method"], target, rate,
                                     *numeric_row(cell["operating"][f"target{target}"][rate])])
    def pct(row):
        return (f"{100*row['grand_mean']:.2f} ± "
                f"{100*row['between_partition_means_sample_sd']:.2f}")
    lines = [
        "# Fixed three-partition CIFAR-100 class-holdout sensitivity",
        "",
        "Descriptive only: class seed 2031 was already studied before the additional two partitions were declared. Each partition uses three independently trained 100-epoch models, all 8,000 ID and 2,000 held-out-class official test images. The displayed ± is the sample SD of three **partition means**, not a confidence interval; each partition's three training-seed values and within-partition SD are in the CSV and JSON.",
        "",
        f"Classifier ID-test accuracy (%): {pct(report['id_test_accuracy'])}. "
        f"ID-fitted temperature: {report['temperature']['grand_mean']:.4f} ± "
        f"{report['temperature']['between_partition_means_sample_sd']:.4f} "
        "across partition means.",
        "",
    ]
    for variant in VARIANTS:
        lines.extend([
            f"## {'T=1' if variant == 'T1' else 'ID-only fitted temperature'}", "",
            "| Method | AUROC % | AUPR-out % | AUPR-in % | FPR95-ID % | ID accepted at fixed95 % | OOD accepted at fixed95 % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for cell in cells:
            if cell["variant"] != variant:
                continue
            vals = [cell["metrics"][metric] for metric in METRICS]
            vals.extend(cell["operating"]["target95"][rate] for rate in (
                "id_test_acceptance", "ood_acceptance"))
            lines.append("| " + cell["method"] + " | " +
                         " | ".join(pct(value) for value in vals) + " |")
        lines.append("")
    lines.extend([
        "All four ID-validation threshold targets (80/90/95/99), achieved ID-test coverage and per-partition points are in `operating.csv` and `summary.json`. The complete raw logits, features, scores, thresholds and masks remain in the nine source runs. No method, partition, checkpoint or operating point was selected by OOD test performance.",
        "",
        "These three fixed partition retrainings do not isolate a causal class effect or estimate the distribution over all possible partitions.",
        "",
    ])
    (REPORT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"pending independent audit: {REPORT.relative_to(ROOT)} "
          f"({len(cells)*len(METRICS)} metric and "
          f"{len(cells)*len(TARGETS)*len(RATES)} operating rows)")


if __name__ == "__main__":
    main()
