"""Write once or verify the complete local CIFAR-100 class-holdout study.

This is a provenance/completeness gate after the independent numerical audits,
not an additional method-selection or score-fitting stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
REPORT = ROOT / "report" / "cifar100-class-holdout"
SEEDS = (0, 1, 2)
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def inputs() -> list[str]:
    names = [
        "docs/CIFAR100_CLASS_HOLDOUT.md",
        "docs/CLASS_HOLDOUT_RUNBOOK.md",
        "data/cifar-100-python/train",
        "data/cifar-100-python/test",
        "data/cifar-100-python/meta",
        "runs/cifar100-class-holdout/declaration.json",
        "runs/cifar100-class-holdout/partition.json",
        "runs/cifar100-class-holdout/partition.npz",
        "runs/cifar100-class-holdout/partition-audit.json",
        "runs/cifar100-class-holdout/test-population.json",
        "runs/cifar100-class-holdout/test-population.npz",
        "runs/cifar100-class-holdout/test-population-audit.json",
        "runs/cifar100-class-holdout/training-launch.json",
        "runs/cifar100-class-holdout/status.json",
        "report/cifar100-class-holdout/summary.json",
        "report/cifar100-class-holdout/metrics.csv",
        "report/cifar100-class-holdout/operating.csv",
        "report/cifar100-class-holdout/SUMMARY.md",
        "report/cifar100-class-holdout/summary-audit.json",
        "report/cifar100-class-holdout/executed_summary_source.py",
        "report/cifar100-class-holdout/executed_summary_audit_source.py",
        "report/cifar100-class-holdout/regression.log",
        "tools/audit_class_holdout_checkpoint_inference.py",
        "tools/audit_class_holdout_checkpoint_cross_device.py",
        "tools/verify_class_holdout_completion.py",
    ]
    for seed in SEEDS:
        base = f"runs/cifar100-class-holdout"
        names += [
            f"{base}/seed{seed}/final-best.pt",
            f"{base}/seed{seed}/last.pt",
            f"{base}/seed{seed}/final-selection.json",
            f"{base}/seed{seed}/history.json",
            f"{base}/seed{seed}/config.json",
            f"{base}/seed{seed}/split_indices.npz",
            f"{base}/seed{seed}/source_hashes.json",
            f"{base}/seed{seed}/status.json",
            f"{base}/train-audit-seed{seed}/audit.json",
            f"{base}/checkpoint-inference-audit-seed{seed}/audit.json",
            f"{base}/checkpoint-inference-audit-seed{seed}/checkpoint-validation.npz",
            f"{base}/extract-seed{seed}/manifest.json",
            f"{base}/extract-seed{seed}/status.json",
            f"{base}/feature-audit-seed{seed}/audit.json",
            f"{base}/score-seed{seed}/manifest.json",
            f"{base}/score-seed{seed}/status.json",
            f"{base}/score-audit-seed{seed}/audit.json",
        ]
        if seed == 2:
            names.append(f"{base}/checkpoint-inference-audit-seed2/"
                         "checkpoint-validation-cuda.npz")
        for stage in ("extract", "score"):
            manifest = read(RUN / f"{stage}-seed{seed}" / "manifest.json")
            for filename in manifest["outputs"]:
                names.append(f"{base}/{stage}-seed{seed}/{filename}")
            names += list(manifest["source_hashes"])
    return sorted(set(names))


def validate() -> dict[str, str]:
    require((REPORT / "summary.json").is_file() and
            (REPORT / "summary-audit.json").is_file() and
            (REPORT / "regression.log").is_file(),
            "three-seed summary, independent audit and regression must finish first")
    declaration = read(RUN / "declaration.json")
    partition = read(RUN / "partition-audit.json")
    population = read(RUN / "test-population-audit.json")
    launch = read(RUN / "training-launch.json")
    phase = read(RUN / "status.json")
    summary = read(REPORT / "summary.json")
    audit = read(REPORT / "summary-audit.json")
    require(declaration["state"] == "declared_before_class_partition" and
            declaration["protocol_sha256"] ==
            sha(ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md") and
            declaration["model_seeds"] == list(SEEDS) and
            declaration["planned_id_train"] == 36000 and
            declaration["planned_id_validation_roles"] == [1600, 1200, 1200] and
            declaration["planned_id_test"] == 8000 and
            declaration["planned_ood_test"] == 2000 and
            declaration["train_file_sha256"] ==
            sha(ROOT / "data/cifar-100-python/train") and
            declaration["meta_file_sha256"] ==
            sha(ROOT / "data/cifar-100-python/meta"),
            "pre-partition declaration or official train source changed")
    require(partition["state"] == population["state"] == "passed" and
            partition["declaration_sha256"] == sha(RUN / "declaration.json") and
            partition["partition_sha256"] == sha(RUN / "partition.npz") and
            population["partition_sha256"] == sha(RUN / "partition.npz") and
            population["test_population_sha256"] ==
            sha(RUN / "test-population.npz") and
            population["official_test_file_sha256"] ==
            sha(ROOT / "data/cifar-100-python/test") and
            population["id_count"] == 8000 and
            population["ood_count"] == 2000,
            "independent partition/test-population audits changed")
    require(launch["state"] == "fixed_training_schedule_launched" and
            launch["declaration_sha256"] == sha(RUN / "declaration.json") and
            launch["partition_audit_sha256"] == sha(RUN / "partition-audit.json") and
            launch["seeds"] == list(SEEDS) and
            launch["epochs_per_seed"] == 100 and
            phase["state"] == "training_complete_pending_independent_audit" and
            phase["trainer_running"] is False and
            [row["seed"] for row in phase["completed_seeds"]] == list(SEEDS),
            "fixed three-seed training schedule not complete")
    require(summary["state"] == "summary_pending_independent_audit" and
            summary["seeds"] == list(SEEDS) and
            summary["methods"] == list(METHODS) and
            summary["variants"] == ["T1", "calibrated"] and
            len(summary["cells"]) == 12 and
            summary["n_id_test"] == 8000 and
            summary["n_ood_test"] == 2000 and
            audit["state"] == "passed" and
            audit["seeds"] == list(SEEDS) and
            audit["summary_sha256"] == sha(REPORT / "summary.json") and
            audit["metrics_csv_sha256"] == sha(REPORT / "metrics.csv") and
            audit["operating_csv_sha256"] == sha(REPORT / "operating.csv") and
            audit["summary_markdown_sha256"] == sha(REPORT / "SUMMARY.md"),
            "independent three-seed report audit not complete")
    for name, digest in summary["source_inputs_sha256"].items():
        require(sha(ROOT / name) == digest, "summary input changed: " + name)
    regression = (REPORT / "regression.log").read_text(encoding="utf-8")
    require("Ran 91 tests" in regression and "OK (skipped=2)" in regression and
            "FAILED" not in regression, "full regression evidence missing")
    for seed in SEEDS:
        folder = RUN / f"seed{seed}"
        training = read(RUN / f"train-audit-seed{seed}" / "audit.json")
        inference = read(RUN / f"checkpoint-inference-audit-seed{seed}" / "audit.json")
        features = read(RUN / f"feature-audit-seed{seed}" / "audit.json")
        scores = read(RUN / f"score-audit-seed{seed}" / "audit.json")
        extracted = read(RUN / f"extract-seed{seed}" / "manifest.json")
        scored = read(RUN / f"score-seed{seed}" / "manifest.json")
        checkpoint_sha = sha(folder / "final-best.pt")
        require(all(row["state"] == "passed" and row["seed"] == seed
                    for row in (training, inference, features, scores)) and
                training["training_epoch_count"] == 100 and
                checkpoint_sha == training["selected_checkpoint_sha256"] ==
                inference["selected_checkpoint_sha256"] ==
                extracted["checkpoint_sha256"] == scored["checkpoint_sha256"] and
                inference["raw_id_validation_sha256"] == sha(
                    RUN / f"checkpoint-inference-audit-seed{seed}" /
                    "checkpoint-validation.npz") and
                inference["source_sha256"] == sha(ROOT / (
                    "tools/audit_class_holdout_checkpoint_cross_device.py"
                    if seed == 2 else
                    "tools/audit_class_holdout_checkpoint_inference.py")) and
                (seed != 2 or (
                    inference["cuda_raw_id_validation_sha256"] == sha(
                        RUN / "checkpoint-inference-audit-seed2" /
                        "checkpoint-validation-cuda.npz") and
                    inference["cpu_cuda_prediction_disagreements"] == 1 and
                    inference["cpu_recomputed_correct_count"] ==
                    inference["recomputed_correct_count"] + 1)) and
                extracted["training_audit_sha256"] ==
                sha(RUN / f"train-audit-seed{seed}" / "audit.json") and
                extracted["checkpoint_inference_audit_sha256"] ==
                sha(RUN / f"checkpoint-inference-audit-seed{seed}" / "audit.json") and
                features["extraction_manifest_sha256"] ==
                scored["extraction_manifest_sha256"] ==
                scores["extraction_manifest_sha256"] ==
                sha(RUN / f"extract-seed{seed}" / "manifest.json") and
                scored["feature_audit_sha256"] ==
                scores["feature_audit_sha256"] ==
                sha(RUN / f"feature-audit-seed{seed}" / "audit.json") and
                scores["scoring_manifest_sha256"] ==
                sha(RUN / f"score-seed{seed}" / "manifest.json") and
                scored["methods"] == list(METHODS) and
                scored["n_id"] == 8000 and scored["n_ood"] == 2000 and
                scored["temperature_fit_role"] == "ID_temperature_1200" and
                scored["threshold_fit_role"] == "ID_threshold_1200",
                f"seed {seed} frozen checkpoint, raw or ID-only audit chain")
        for stage, manifest in (("extract", extracted), ("score", scored)):
            stage_dir = RUN / f"{stage}-seed{seed}"
            status = read(stage_dir / "status.json")
            require(status["state"] == manifest["state"] and
                    status["seed"] == seed and
                    status["manifest_sha256"] == sha(stage_dir / "manifest.json"),
                    f"seed {seed} {stage} status")
            for filename, digest in manifest["outputs"].items():
                require(sha(stage_dir / filename) == digest,
                        f"seed {seed} {stage} raw file changed: {filename}")
            for name, digest in manifest["source_hashes"].items():
                require(sha(ROOT / name) == digest,
                        f"seed {seed} {stage} executed source changed: {name}")
    return {name: sha(ROOT / name) for name in inputs()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = REPORT / "completion.json"
    if args.write:
        require(not path.exists(), "completion manifest already exists")
        hashes = validate()
        record = {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(SEEDS),
            "id_test_count": 8000,
            "ood_test_count": 2000,
            "methods": list(METHODS),
            "variants": ["T1", "calibrated"],
            "inputs_sha256": hashes,
            "boundary": "One fixed CIFAR-100 80/20 class split, all three 100-epoch seeds and all methods; no OOD-test-based selection or untouched-external-confirmation claim.",
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record = read(path)
    require(record["state"] == "complete" and
            record["seeds"] == list(SEEDS) and
            record["id_test_count"] == 8000 and
            record["ood_test_count"] == 2000 and
            record["methods"] == list(METHODS), "completion header")
    actual = validate()
    require(record["inputs_sha256"] == actual,
            "completion hash-bound source/raw files changed")
    print(json.dumps({"state": "verified_complete", "files": len(actual),
                      "seeds": record["seeds"]}))


if __name__ == "__main__":
    main()
