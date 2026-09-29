"""Independent CPU audit of the frozen same-source CIFAR-100 test populations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
TEST = ROOT / "data" / "cifar-100-python" / "test"


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


def audit() -> dict:
    manifest = read(RUN / "test-population.json")
    declaration = read(RUN / "declaration.json")
    partition_manifest = read(RUN / "partition.json")
    partition_audit = read(RUN / "partition-audit.json")
    require(manifest["state"] == "test_population_prepared_pending_independent_audit" and
            manifest["protocol_sha256"] == declaration["protocol_sha256"] ==
            sha(ROOT / "docs" / "CIFAR100_CLASS_HOLDOUT.md") and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") ==
            partition_manifest["declaration_sha256"] and
            manifest["partition_sha256"] == sha(RUN / "partition.npz") ==
            partition_manifest["partition_sha256"] == partition_audit["partition_sha256"] and
            manifest["partition_audit_sha256"] == sha(RUN / "partition-audit.json") and
            partition_audit["state"] == "passed" and
            manifest["official_test_file_sha256"] == sha(TEST) and
            manifest["test_population_sha256"] == sha(RUN / "test-population.npz") and
            manifest["preparation_source_sha256"] ==
            sha(ROOT / "tools" / "prepare_class_holdout_test.py"),
            "source, declaration or audited training partition binding differs")
    with np.load(RUN / "partition.npz", allow_pickle=False) as train, \
         np.load(RUN / "test-population.npz", allow_pickle=False) as saved:
        with TEST.open("rb") as stream:
            source = pickle.load(stream, encoding="bytes")
        pixel_data = np.asarray(source[b"data"])
        fine = np.asarray(source[b"fine_labels"], dtype=np.int64)
        coarse = np.asarray(source[b"coarse_labels"], dtype=np.int64)
        id_classes = set(int(label) for label in train["id_fine_labels"])
        ood_classes = set(int(label) for label in train["ood_fine_labels"])
        require(id_classes.isdisjoint(ood_classes) and len(id_classes) == 80 and
                len(ood_classes) == 20 and id_classes | ood_classes == set(range(100)) and
                manifest["id_fine_labels"] == sorted(id_classes) and
                manifest["ood_fine_labels"] == sorted(ood_classes),
                "80/20 class list differs")
        require(pixel_data.shape == (10000, 3072) and pixel_data.dtype == np.uint8 and
                len(fine) == len(coarse) == 10000 and
                np.array_equal(np.bincount(fine, minlength=100), np.full(100, 100)) and
                np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 500)) and
                np.array_equal(train["fine_to_coarse"][fine], coarse),
                "official test rows/counts or training-derived coarse mapping differ")
        independent_id_rows = np.asarray([i for i, label in enumerate(fine)
                                          if int(label) in id_classes], dtype=np.int64)
        independent_ood_rows = np.asarray([i for i, label in enumerate(fine)
                                           if int(label) in ood_classes], dtype=np.int64)
        label_map = {label: position for position, label in enumerate(sorted(id_classes))}
        targets = np.asarray([label_map[int(fine[i])] for i in independent_id_rows],
                             dtype=np.int64)
        expected = {
            "id_test_indices": independent_id_rows,
            "ood_test_indices": independent_ood_rows,
            "id_original_fine_labels": fine[independent_id_rows],
            "ood_original_fine_labels": fine[independent_ood_rows],
            "id_original_coarse_labels": coarse[independent_id_rows],
            "ood_original_coarse_labels": coarse[independent_ood_rows],
            "id_class_targets": targets,
        }
        require(set(saved.files) == set(expected) and
                all(np.array_equal(saved[name], value) for name, value in expected.items()),
                "saved test row indices or original labels differ from independent reconstruction")
        require(len(independent_id_rows) == manifest["id_count"] == 8000 and
                len(independent_ood_rows) == manifest["ood_count"] == 2000 and
                manifest["ood_prevalence"] == .2 and
                np.array_equal(np.sort(np.r_[independent_id_rows, independent_ood_rows]),
                               np.arange(10000)) and
                np.array_equal(np.bincount(targets, minlength=80), np.full(80, 100)),
                "full 8000/2000 population coverage differs")
        require(not np.intersect1d(train["train_indices"],
                                   train["excluded_ood_train_indices"]).size and
                len(train["excluded_ood_train_indices"]) == 10000,
                "training exclusion invariant differs")
    return {
        "state": "passed", "id_count": 8000, "ood_count": 2000,
        "original_test_rows_reconstructed": 10000,
        "class_count_checks": 100, "coarse_count_checks": 20,
        "seven_arrays_exact": True,
        "official_test_file_sha256": sha(TEST),
        "partition_sha256": sha(RUN / "partition.npz"),
        "test_population_sha256": sha(RUN / "test-population.npz"),
        "test_population_manifest_sha256": sha(RUN / "test-population.json"),
        "preparation_source_sha256": sha(ROOT / "tools" / "prepare_class_holdout_test.py"),
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Test population and source audit only; no checkpoint, logits, OOD scores or metrics.",
    }


def main() -> None:
    destination = RUN / "test-population-audit.json"
    snapshot = RUN / "test-population-audit-source.py"
    require(not destination.exists() and not snapshot.exists(),
            "test population audit already exists; preserve it")
    result = audit()
    shutil.copyfile(__file__, snapshot)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"state": result["state"], "id": result["id_count"],
                      "ood": result["ood_count"]}), flush=True)


if __name__ == "__main__":
    main()
