"""Independently verify every held-out class and ID train/validation role row."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
TRAIN = ROOT / "data" / "cifar-100-python" / "train"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def main() -> None:
    destination = RUN / "partition-audit.json"
    require(not destination.exists(), "partition already audited")
    declaration = read(RUN / "declaration.json")
    manifest = read(RUN / "partition.json")
    require(manifest["state"] == "partition_prepared_pending_independent_audit" and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            manifest["source_sha256"] == sha(ROOT / "tools/prepare_class_holdout.py") and
            manifest["protocol_sha256"] == sha(ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md") and
            declaration["train_file_sha256"] == manifest["train_file_sha256"] == sha(TRAIN) and
            declaration["meta_file_sha256"] == manifest["meta_file_sha256"] ==
            sha(ROOT / "data/cifar-100-python/meta"),
            "frozen declaration, source or training file changed")
    partition = RUN / "partition.npz"
    require(manifest["partition_sha256"] == sha(partition), "partition hash changed")
    with np.load(partition, allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    expected_keys = {"id_fine_labels", "ood_fine_labels", "fine_to_coarse",
                     "train_indices", "validation_indices", "excluded_ood_train_indices",
                     "checkpoint_indices", "temperature_indices", "threshold_indices",
                     "train_original_fine_labels", "validation_original_fine_labels"}
    require(set(arrays) == expected_keys, "partition array schema differs")
    with TRAIN.open("rb") as stream:
        source = pickle.load(stream, encoding="bytes")
    fine = np.asarray(source[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(source[b"coarse_labels"], dtype=np.int64)
    require(fine.shape == coarse.shape == (50000,) and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500)),
            "official training labels/counts differ")
    mapping = np.asarray([np.unique(coarse[fine == label]).item() for label in range(100)])
    require(np.array_equal(mapping, arrays["fine_to_coarse"]) and
            np.array_equal(mapping, manifest["fine_to_coarse"]),
            "fine-to-coarse mapping differs")
    generator = np.random.default_rng(2031)
    holdout = []
    for coarse_id in range(20):
        members = np.flatnonzero(mapping == coarse_id)
        require(len(members) == 5, "coarse membership differs")
        holdout.append(int(generator.permutation(members)[0]))
    holdout = np.sort(np.asarray(holdout, dtype=np.int64))
    known = np.sort(np.asarray([label for label in range(100) if label not in set(holdout)], dtype=np.int64))
    require(np.array_equal(holdout, arrays["ood_fine_labels"]) and
            np.array_equal(known, arrays["id_fine_labels"]) and
            manifest["ood_fine_labels"] == holdout.tolist() and
            manifest["id_fine_labels"] == known.tolist(),
            "predeclared one-class-per-coarse draw differs")
    training, validation, excluded = (arrays[key] for key in
        ("train_indices", "validation_indices", "excluded_ood_train_indices"))
    require(all(row.dtype == np.int64 and np.array_equal(row, np.unique(row))
                for row in (training, validation, excluded)) and
            [len(row) for row in (training, validation, excluded)] == [36000, 4000, 10000] and
            np.array_equal(np.sort(np.concatenate((training, validation, excluded))),
                           np.arange(50000)) and
            np.array_equal(excluded, np.flatnonzero(np.isin(fine, holdout))),
            "all official training rows must be partitioned exactly once")
    require(not np.isin(fine[training], holdout).any() and
            not np.isin(fine[validation], holdout).any() and
            np.array_equal(arrays["train_original_fine_labels"], fine[training]) and
            np.array_equal(arrays["validation_original_fine_labels"], fine[validation]) and
            np.array_equal(np.bincount(fine[training], minlength=100)[known], np.full(80, 450)) and
            np.array_equal(np.bincount(fine[validation], minlength=100)[known], np.full(80, 50)),
            "held-out leakage or per-ID-class train/validation counts")
    split_rng = np.random.default_rng(2026)
    reproduced_val = []
    for label in known:
        candidates = np.flatnonzero(fine == label)
        reproduced_val.extend(split_rng.permutation(candidates)[:50].tolist())
    require(np.array_equal(np.sort(reproduced_val), validation),
            "ID validation draw cannot be independently regenerated")
    role_rng = np.random.default_rng(2029)
    reconstructed: dict[str, list[int]] = {key: [] for key in
        ("checkpoint", "temperature", "threshold")}
    for label in known:
        rows = validation[fine[validation] == label]
        draw = role_rng.permutation(rows)
        reconstructed["checkpoint"].extend(draw[:20].tolist())
        reconstructed["temperature"].extend(draw[20:35].tolist())
        reconstructed["threshold"].extend(draw[35:].tolist())
    for role, per_class, total in (("checkpoint", 20, 1600),
                                   ("temperature", 15, 1200),
                                   ("threshold", 15, 1200)):
        actual = arrays[role + "_indices"]
        require(len(actual) == total and
                np.array_equal(actual, np.sort(reconstructed[role])) and
                np.array_equal(np.bincount(fine[actual], minlength=100)[known],
                               np.full(80, per_class)) and
                not np.isin(fine[actual], holdout).any(),
                role + " independently regenerated ID role differs")
    require(np.array_equal(np.sort(np.concatenate([arrays[role + "_indices"]
                    for role in reconstructed])), validation),
            "ID validation roles overlap or omit rows")
    record = {"state": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
              "declaration_sha256": sha(RUN / "declaration.json"),
              "partition_sha256": sha(partition),
              "partition_manifest_sha256": sha(RUN / "partition.json"),
              "source_sha256": sha(Path(__file__)),
              "counts": {"id_classes": 80, "held_out_classes": 20,
                         "id_train": 36000, "id_validation": 4000,
                         "excluded_ood_train": 10000,
                         "checkpoint": 1600, "temperature": 1200, "threshold": 1200},
              "boundary": "Independent official-train-only class and ID role reconstruction; no official test file or OOD score read."}
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "counts": record["counts"]}))


if __name__ == "__main__":
    main()
