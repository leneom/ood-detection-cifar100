"""Derive the predeclared 80-ID/20-OOD CIFAR-100 class and ID-role split."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
DATA = ROOT / "data" / "cifar-100-python"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    output = RUN / "partition.npz"
    manifest_path = RUN / "partition.json"
    require(not output.exists() and not manifest_path.exists(), "class partition already exists")
    declaration_path = RUN / "declaration.json"
    declaration = read(declaration_path)
    require(declaration["state"] == "declared_before_class_partition" and
            declaration["protocol_sha256"] == sha(ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md") and
            declaration["train_file_sha256"] == sha(DATA / "train") and
            declaration["meta_file_sha256"] == sha(DATA / "meta") and
            declaration["class_rng_seed"] == 2031 and
            declaration["validation_split_seed"] == 2026 and
            declaration["validation_role_seed"] == 2029,
            "pre-partition frozen declaration/source differs")
    with (DATA / "train").open("rb") as stream:
        training = pickle.load(stream, encoding="bytes")
    with (DATA / "meta").open("rb") as stream:
        metadata = pickle.load(stream, encoding="bytes")
    fine = np.asarray(training[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(training[b"coarse_labels"], dtype=np.int64)
    pixels = training[b"data"]
    require(fine.shape == coarse.shape == (50000,) and
            pixels.shape == (50000, 3072) and pixels.dtype == np.uint8 and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500)) and
            np.array_equal(np.unique(coarse), np.arange(20)) and
            len(metadata[b"fine_label_names"]) == 100 and
            len(metadata[b"coarse_label_names"]) == 20,
            "official 50k CIFAR-100 training schema/counts differ")
    fine_to_coarse = np.full(100, -1, dtype=np.int64)
    groups: list[np.ndarray] = []
    for group in range(20):
        labels = np.unique(fine[coarse == group])
        require(len(labels) == 5, f"coarse group {group} does not contain five fine labels")
        for label in labels:
            require(fine_to_coarse[label] == -1 and
                    np.all(coarse[fine == label] == group),
                    f"fine label {label} has nonunique coarse parent")
            fine_to_coarse[label] = group
        groups.append(labels)
    require(np.all(fine_to_coarse >= 0), "missing fine-to-coarse mapping")
    class_rng = np.random.default_rng(2031)
    ood_fine = np.sort(np.asarray([class_rng.permutation(labels)[0] for labels in groups], dtype=np.int64))
    id_fine = np.setdiff1d(np.arange(100), ood_fine)
    require(len(id_fine) == 80 and len(ood_fine) == 20 and
            np.array_equal(np.bincount(fine_to_coarse[ood_fine], minlength=20), np.ones(20, dtype=np.int64)),
            "one held-out fine label per coarse group")
    split_rng = np.random.default_rng(2026)
    train_rows: list[int] = []
    val_rows: list[int] = []
    for label in id_fine:
        belonging = np.flatnonzero(fine == label)
        shuffled = split_rng.permutation(belonging)
        val_rows.extend(shuffled[:50].tolist())
        train_rows.extend(shuffled[50:].tolist())
    train_rows = np.sort(np.asarray(train_rows, dtype=np.int64))
    val_rows = np.sort(np.asarray(val_rows, dtype=np.int64))
    excluded = np.flatnonzero(np.isin(fine, ood_fine)).astype(np.int64)
    role_rng = np.random.default_rng(2029)
    roles: list[list[int]] = [[], [], []]
    for label in id_fine:
        belonging = val_rows[fine[val_rows] == label]
        require(len(belonging) == 50, f"fine label {label} validation count differs")
        shuffled = role_rng.permutation(belonging)
        for destination, selected in zip(roles, (shuffled[:20], shuffled[20:35], shuffled[35:])):
            destination.extend(selected.tolist())
    checkpoint, temperature, threshold = [np.sort(np.asarray(rows, dtype=np.int64)) for rows in roles]
    require([len(x) for x in (train_rows, val_rows, excluded, checkpoint, temperature, threshold)] ==
            [36000, 4000, 10000, 1600, 1200, 1200] and
            np.array_equal(np.sort(np.concatenate((train_rows, val_rows, excluded))), np.arange(50000)) and
            np.array_equal(np.sort(np.concatenate((checkpoint, temperature, threshold))), val_rows),
            "complete disjoint train/validation/held-out and ID-role partitions")
    arrays = {"id_fine_labels": id_fine, "ood_fine_labels": ood_fine,
              "fine_to_coarse": fine_to_coarse, "train_indices": train_rows,
              "validation_indices": val_rows, "excluded_ood_train_indices": excluded,
              "checkpoint_indices": checkpoint, "temperature_indices": temperature,
              "threshold_indices": threshold, "train_original_fine_labels": fine[train_rows],
              "validation_original_fine_labels": fine[val_rows]}
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)
    fine_names = [value.decode("utf-8") for value in metadata[b"fine_label_names"]]
    coarse_names = [value.decode("utf-8") for value in metadata[b"coarse_label_names"]]
    record = {"state": "partition_prepared_pending_independent_audit",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "declaration_sha256": sha(declaration_path),
              "protocol_sha256": declaration["protocol_sha256"],
              "source_sha256": sha(Path(__file__)),
              "partition_sha256": sha(output),
              "train_file_sha256": declaration["train_file_sha256"],
              "meta_file_sha256": declaration["meta_file_sha256"],
              "class_rng_seed": 2031, "validation_split_seed": 2026,
              "validation_role_seed": 2029,
              "fine_label_names": fine_names, "coarse_label_names": coarse_names,
              "id_fine_labels": id_fine.tolist(), "ood_fine_labels": ood_fine.tolist(),
              "fine_to_coarse": fine_to_coarse.tolist(),
              "counts": {"id_train": 36000, "id_validation": 4000,
                         "held_out_train_excluded": 10000,
                         "checkpoint": 1600, "temperature": 1200, "threshold": 1200},
              "boundary": "Training-source class/ID role partition only; official test file and OOD scores not read."}
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": record["state"], "id_classes": 80,
                      "held_out_classes": 20, "id_train": 36000,
                      "id_validation_roles": [1600, 1200, 1200]}))


if __name__ == "__main__":
    main()
