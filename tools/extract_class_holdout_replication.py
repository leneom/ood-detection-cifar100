"""Adapt the unchanged six-score GPU extractor to audited new class splits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.runtime import file_hash
from tools import extract_class_holdout as original
from tools.train_class_holdout_replication import BASE, frozen_partition


SOURCE = (
    "tools/extract_class_holdout.py", "tools/extract_class_holdout_replication.py",
    "src/ood/class_holdout_train.py", "src/ood/data.py",
    "src/ood/evaluate.py", "src/ood/features.py", "src/ood/models.py",
    "src/ood/runtime.py", "src/ood/scores.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def preflight(class_seed: int, seed: int) -> tuple[dict, dict, dict, dict, Path]:
    split = BASE / f"split{class_seed}"
    status = read(BASE / "status.json")
    require(status["state"] == "training_complete_pending_independent_audit" and
            status["trainer_running"] is False and
            len(status["completed"]) == 6 and
            {(row["class_seed"], row["training_seed"])
             for row in status["completed"]} ==
            {(class_id, train_seed) for class_id in (2032, 2033)
             for train_seed in (0, 1, 2)},
            "all six full models must finish before any GPU test extraction")
    partition, evidence = frozen_partition(class_seed)
    test_manifest = read(split / "test-population.json")
    test_audit_path = split / "test-population-audit.json"
    test_audit = read(test_audit_path)
    require(test_audit["state"] == "passed" and
            test_audit["class_rng_seed"] == class_seed and
            test_audit["test_population_manifest_sha256"] ==
            file_hash(split / "test-population.json") and
            test_audit["test_population_sha256"] ==
            test_manifest["test_population_sha256"] ==
            file_hash(split / "test-population.npz") and
            test_audit["partition_sha256"] == evidence["partition_sha256"] and
            test_audit["official_test_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/test") and
            test_audit["audit_source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_replication_test.py") and
            test_manifest["preparation_source_sha256"] ==
            file_hash(ROOT / "tools/prepare_class_holdout_replication_test.py") and
            test_manifest["id_count"] == 8000 and test_manifest["ood_count"] == 2000,
            "independent complete official test population audit")
    launch_hash = file_hash(BASE / "training-launch.json")
    for planned_seed in (0, 1, 2):
        folder = split / f"seed{planned_seed}"
        training_path = split / f"train-audit-seed{planned_seed}/audit.json"
        inference_path = split / f"checkpoint-inference-audit-seed{planned_seed}/audit.json"
        training = read(training_path)
        inference = read(inference_path)
        checkpoint_hash = file_hash(folder / "final-best.pt")
        require(training["state"] == inference["state"] == "passed" and
                training["class_seed"] == inference["class_seed"] == class_seed and
                training["seed"] == inference["seed"] == planned_seed and
                training["training_epoch_count"] == 100 and
                training["selected_checkpoint_sha256"] ==
                inference["selected_checkpoint_sha256"] == checkpoint_hash and
                training["training_launch_sha256"] == launch_hash and
                inference["training_audit_sha256"] == file_hash(training_path) and
                inference["source_sha256"] ==
                file_hash(ROOT / "tools/finalize_class_holdout_replication_checkpoint.py") and
                inference["recomputed_accuracy"] ==
                training["selected_id_checkpoint_accuracy"] and
                inference["id_checkpoint_validation_count"] == 1600,
                f"split {class_seed} seed {planned_seed} lacks full ID-only gates")
        if inference["primary_inference_path"] == "CPU":
            raw = split / f"checkpoint-cpu-audit-seed{planned_seed}/checkpoint-validation.npz"
            require(inference["cuda_audit_sha256"] is None,
                    "CPU path must not have CUDA exception")
        else:
            raw = split / f"checkpoint-cuda-audit-seed{planned_seed}/checkpoint-validation-cuda.npz"
            require(inference["primary_inference_path"] == "CUDA" and
                    inference["cuda_audit_sha256"] == file_hash(
                        split / f"checkpoint-cuda-audit-seed{planned_seed}/audit.json"),
                    "source-bound CUDA ID-only exception")
        require(inference["raw_id_validation_sha256"] == file_hash(raw),
                "complete 1,600 ID-only checkpoint raw logits")
    folder = split / f"seed{seed}"
    training = read(split / f"train-audit-seed{seed}/audit.json")
    selection = read(folder / "final-selection.json")
    config = read(folder / "config.json")
    checkpoint = folder / "final-best.pt"
    require(training["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == file_hash(checkpoint) and
            training["partition_sha256"] == evidence["partition_sha256"] and
            config["class_partition"] == evidence and
            config["classes"] == 80 and config["seed"] == seed and
            config["id_fine_labels"] == partition["id_fine_labels"].tolist() and
            config["ood_fine_labels"] == partition["ood_fine_labels"].tolist() and
            selection["count"] == 1600 and
            selection["selection_population"] == "ID_checkpoint_validation",
            "selected frozen model belongs to this audited class split")
    return config, selection, partition, test_manifest, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    require(args.batch_size > 0, "positive extraction batch size")
    split = BASE / f"split{args.class_seed}"
    original.RUN = split
    original.SOURCE = SOURCE
    original.preflight = lambda seed: preflight(args.class_seed, seed)
    sys.argv = ["tools/extract_class_holdout.py", "--seed", str(args.seed),
                "--batch-size", str(args.batch_size)]
    original.main()
    extracted = split / f"extract-seed{args.seed}"
    supplement = extracted / "class-provenance.json"
    require(not supplement.exists(), "write-once class provenance")
    manifest = read(extracted / "manifest.json")
    record = {
        "state": "source_bound_replication_extraction", "class_rng_seed": args.class_seed,
        "training_seed": args.seed,
        "replication_declaration_sha256": file_hash(BASE / "declaration.json"),
        "partition_sha256": file_hash(split / "partition.npz"),
        "partition_audit_sha256": file_hash(split / "partition-audit.json"),
        "test_population_audit_sha256": file_hash(split / "test-population-audit.json"),
        "checkpoint_inference_audit_sha256": file_hash(
            split / f"checkpoint-inference-audit-seed{args.seed}/audit.json"),
        "extraction_manifest_sha256": file_hash(extracted / "manifest.json"),
        "source_sha256": file_hash(Path(__file__)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Reused unchanged extractor on audited class split; no temperature/threshold/OOD-based selection.",
    }
    supplement.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
