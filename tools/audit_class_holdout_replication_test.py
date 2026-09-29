"""Independent reconstruction of each complete official test population."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil

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
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    args = parser.parse_args()
    split = BASE / f"split{args.class_seed}"
    destination = split / "test-population-audit.json"
    snapshot = split / "test-population-audit-source.py"
    require(not destination.exists() and not snapshot.exists(),
            "write-once test population audit exists")
    manifest = read(split / "test-population.json")
    declaration = read(BASE / "declaration.json")
    partition_manifest = read(split / "partition.json")
    partition_audit = read(split / "partition-audit.json")
    require(manifest["state"] == "test_population_prepared_pending_independent_audit" and
            manifest["class_rng_seed"] == args.class_seed and
            declaration["state"] == "frozen_before_new_class_partition_derivation" and
            manifest["protocol_sha256"] == sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION.md") and
            manifest["declaration_sha256"] == sha(BASE / "declaration.json") ==
            partition_manifest["declaration_sha256"] and
            manifest["partition_sha256"] == sha(split / "partition.npz") ==
            partition_manifest["partition_sha256"] == partition_audit["partition_sha256"] and
            manifest["partition_audit_sha256"] == sha(split / "partition-audit.json") and
            partition_audit["state"] == "passed" and
            manifest["training_launch_sha256"] == sha(BASE / "training-launch.json") and
            manifest["official_test_file_sha256"] == sha(TEST) and
            manifest["test_population_sha256"] == sha(split / "test-population.npz") and
            manifest["preparation_source_sha256"] ==
            sha(ROOT / "tools/prepare_class_holdout_replication_test.py"),
            "frozen protocol/training partition and test source binding")
    with np.load(split / "partition.npz", allow_pickle=False) as partition, \
         np.load(split / "test-population.npz", allow_pickle=False) as stored:
        with TEST.open("rb") as stream:
            source = pickle.load(stream, encoding="bytes")
        pixels = np.asarray(source[b"data"])
        fine = np.asarray(source[b"fine_labels"], dtype=np.int64)
        coarse = np.asarray(source[b"coarse_labels"], dtype=np.int64)
        ids = set(int(label) for label in partition["id_fine_labels"])
        oods = set(int(label) for label in partition["ood_fine_labels"])
        require(ids.isdisjoint(oods) and len(ids) == 80 and len(oods) == 20 and
                ids | oods == set(range(100)) and
                manifest["id_fine_labels"] == sorted(ids) and
                manifest["ood_fine_labels"] == sorted(oods),
                "complete original fine-class split")
        require(pixels.shape == (10000, 3072) and pixels.dtype == np.uint8 and
                fine.shape == coarse.shape == (10000,) and
                np.array_equal(np.bincount(fine, minlength=100), np.full(100, 100)) and
                np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 500)) and
                np.array_equal(partition["fine_to_coarse"][fine], coarse),
                "independent complete official CIFAR-100 test census")
        id_rows = np.asarray([i for i, label in enumerate(fine)
                              if int(label) in ids], dtype=np.int64)
        ood_rows = np.asarray([i for i, label in enumerate(fine)
                               if int(label) in oods], dtype=np.int64)
        remap = {label: position for position, label in enumerate(sorted(ids))}
        targets = np.asarray([remap[int(fine[i])] for i in id_rows], dtype=np.int64)
        expected = {
            "id_test_indices": id_rows,
            "ood_test_indices": ood_rows,
            "id_original_fine_labels": fine[id_rows],
            "ood_original_fine_labels": fine[ood_rows],
            "id_original_coarse_labels": coarse[id_rows],
            "ood_original_coarse_labels": coarse[ood_rows],
            "id_class_targets": targets,
        }
        require(set(stored.files) == set(expected) and
                all(np.array_equal(stored[name], value)
                    for name, value in expected.items()),
                "seven raw test arrays independently reconstructed")
        require(len(id_rows) == manifest["id_count"] == 8000 and
                len(ood_rows) == manifest["ood_count"] == 2000 and
                manifest["ood_prevalence"] == .2 and
                np.array_equal(np.sort(np.r_[id_rows, ood_rows]), np.arange(10000)) and
                np.array_equal(np.bincount(targets, minlength=80), np.full(80, 100)),
                "all 8k ID/2k OOD official rows exactly once")
    record = {
        "state": "passed", "class_rng_seed": args.class_seed,
        "id_count": 8000, "ood_count": 2000,
        "original_test_rows_reconstructed": 10000,
        "class_count_checks": 100, "coarse_count_checks": 20,
        "seven_arrays_exact": True,
        "official_test_file_sha256": sha(TEST),
        "partition_sha256": sha(split / "partition.npz"),
        "test_population_sha256": sha(split / "test-population.npz"),
        "test_population_manifest_sha256": sha(split / "test-population.json"),
        "preparation_source_sha256": sha(
            ROOT / "tools/prepare_class_holdout_replication_test.py"),
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Full test-population/row/source audit only; no checkpoint, logits, score or metric read.",
    }
    shutil.copyfile(__file__, snapshot)
    destination.write_text(json.dumps(record, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "class_seed": args.class_seed,
                      "id": 8000, "ood": 2000}), flush=True)


if __name__ == "__main__":
    main()
