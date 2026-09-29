"""Freeze complete official CIFAR-100 test rows for one fully audited split.

This CPU-only stage reads labels/pixels but computes no model output or score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/cifar100-class-holdout-replication"
TEST = ROOT / "data/cifar-100-python/test"


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
        raise ValueError(message)


def preflight(class_seed: int) -> Path:
    split = BASE / f"split{class_seed}"
    root = read(BASE / "status.json")
    completed = {(row["class_seed"], row["training_seed"]): row
                 for row in root["completed"]}
    require(all((class_seed, seed) in completed for seed in (0, 1, 2)),
            "all three full 100-epoch models for this split must finish")
    partition_audit = read(split / "partition-audit.json")
    require(partition_audit["state"] == "passed" and
            partition_audit["class_rng_seed"] == class_seed and
            partition_audit["partition_sha256"] == sha(split / "partition.npz"),
            "source-bound split partition audit")
    for seed in (0, 1, 2):
        model = split / f"seed{seed}"
        train_audit_path = split / f"train-audit-seed{seed}/audit.json"
        train_audit = read(train_audit_path)
        cpu_path = split / f"checkpoint-cpu-audit-seed{seed}/audit.json"
        cpu = read(cpu_path)
        checkpoint_sha = sha(model / "final-best.pt")
        require(train_audit["state"] == "passed" and
                train_audit["class_seed"] == cpu["class_seed"] == class_seed and
                train_audit["seed"] == cpu["seed"] == seed and
                train_audit["training_epoch_count"] == 100 and
                train_audit["selected_checkpoint_sha256"] ==
                cpu["selected_checkpoint_sha256"] == checkpoint_sha and
                cpu["training_audit_sha256"] == sha(train_audit_path) and
                cpu["raw_id_validation_sha256"] == sha(
                    split / f"checkpoint-cpu-audit-seed{seed}/checkpoint-validation.npz"),
                "full model and CPU ID-only audits")
        if cpu["state"] == "passed":
            require(cpu["recomputed_accuracy"] ==
                    train_audit["selected_id_checkpoint_accuracy"],
                    "CPU selected checkpoint accuracy")
        else:
            cuda_path = split / f"checkpoint-cuda-audit-seed{seed}/audit.json"
            cuda = read(cuda_path)
            require(cpu["state"] == "cpu_cuda_discrepancy_requires_cuda_audit" and
                    cuda["state"] == "passed" and
                    cuda["selected_checkpoint_sha256"] == checkpoint_sha and
                    cuda["cpu_audit_sha256"] == sha(cpu_path) and
                    cuda["recomputed_accuracy"] ==
                    train_audit["selected_id_checkpoint_accuracy"],
                    "original CUDA selected checkpoint reproduction")
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    args = parser.parse_args()
    split = preflight(args.class_seed)
    output = split / "test-population.npz"
    manifest_path = split / "test-population.json"
    require(not output.exists() and not manifest_path.exists(),
            "write-once test population exists")
    with np.load(split / "partition.npz", allow_pickle=False) as partition:
        ids = partition["id_fine_labels"].astype(np.int64)
        oods = partition["ood_fine_labels"].astype(np.int64)
        mapping = partition["fine_to_coarse"].astype(np.int64)
    require(len(ids) == 80 and len(oods) == 20 and
            np.array_equal(np.sort(np.r_[ids, oods]), np.arange(100)) and
            np.array_equal(ids, np.sort(ids)), "80/20 class split")
    with TEST.open("rb") as stream:
        official = pickle.load(stream, encoding="bytes")
    pixels = np.asarray(official[b"data"])
    fine = np.asarray(official[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(official[b"coarse_labels"], dtype=np.int64)
    require(pixels.shape == (10000, 3072) and pixels.dtype == np.uint8 and
            fine.shape == coarse.shape == (10000,) and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 100)) and
            np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 500)) and
            np.array_equal(mapping[fine], coarse), "official complete test census")
    id_rows = np.where(np.isin(fine, ids))[0].astype(np.int64)
    ood_rows = np.where(np.isin(fine, oods))[0].astype(np.int64)
    id_targets = np.searchsorted(ids, fine[id_rows]).astype(np.int64)
    require(len(id_rows) == 8000 and len(ood_rows) == 2000 and
            np.array_equal(np.sort(np.r_[id_rows, ood_rows]), np.arange(10000)) and
            np.array_equal(ids[id_targets], fine[id_rows]) and
            np.array_equal(np.bincount(id_targets, minlength=80), np.full(80, 100)),
            "full original-order 8000/2000 test rows")
    temp = output.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, id_test_indices=id_rows,
                            ood_test_indices=ood_rows,
                            id_original_fine_labels=fine[id_rows],
                            ood_original_fine_labels=fine[ood_rows],
                            id_original_coarse_labels=coarse[id_rows],
                            ood_original_coarse_labels=coarse[ood_rows],
                            id_class_targets=id_targets)
    temp.replace(output)
    manifest = {
        "state": "test_population_prepared_pending_independent_audit",
        "class_rng_seed": args.class_seed,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION.md"),
        "declaration_sha256": sha(BASE / "declaration.json"),
        "partition_sha256": sha(split / "partition.npz"),
        "partition_audit_sha256": sha(split / "partition-audit.json"),
        "training_launch_sha256": sha(BASE / "training-launch.json"),
        "official_test_file_sha256": sha(TEST),
        "test_population_sha256": sha(output),
        "preparation_source_sha256": sha(Path(__file__)),
        "id_count": len(id_rows), "ood_count": len(ood_rows),
        "ood_prevalence": len(ood_rows) / len(fine),
        "id_fine_labels": ids.tolist(), "ood_fine_labels": oods.tolist(),
        "boundary": "Full original test-row population only after three audited models; no logits, score, metric or parameter selection.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": manifest["state"],
                      "class_seed": args.class_seed,
                      "id": len(id_rows), "ood": len(ood_rows)}), flush=True)


if __name__ == "__main__":
    main()
