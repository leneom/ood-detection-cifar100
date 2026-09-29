"""Derive both predeclared CIFAR-100 held-out-class replication partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-replication"
DATA = ROOT / "data/cifar-100-python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def derive(class_seed: int, fine: np.ndarray, coarse: np.ndarray,
           fine_to_coarse: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(class_seed)
    ood_fine = np.sort(np.array([
        rng.permutation(np.flatnonzero(fine_to_coarse == group))[0]
        for group in range(20)], dtype=np.int64))
    id_fine = np.setdiff1d(np.arange(100), ood_fine)
    split_rng = np.random.default_rng(2026)
    train, validation = [], []
    for label in id_fine:
        rows = np.flatnonzero(fine == label)
        shuffled = split_rng.permutation(rows)
        validation.extend(shuffled[:50])
        train.extend(shuffled[50:])
    train = np.sort(np.asarray(train, dtype=np.int64))
    validation = np.sort(np.asarray(validation, dtype=np.int64))
    excluded = np.flatnonzero(np.isin(fine, ood_fine)).astype(np.int64)
    role_rng = np.random.default_rng(2029)
    roles = ([], [], [])
    for label in id_fine:
        rows = validation[fine[validation] == label]
        shuffled = role_rng.permutation(rows)
        for bucket, subrows in zip(roles,
                                   (shuffled[:20], shuffled[20:35], shuffled[35:])):
            bucket.extend(subrows)
    checkpoint, temperature, threshold = (
        np.sort(np.asarray(rows, dtype=np.int64)) for rows in roles)
    require(len(id_fine) == 80 and len(ood_fine) == 20 and
            np.array_equal(np.bincount(fine_to_coarse[ood_fine], minlength=20),
                           np.ones(20, dtype=np.int64)) and
            [len(x) for x in (train, validation, excluded, checkpoint,
                              temperature, threshold)] ==
            [36000, 4000, 10000, 1600, 1200, 1200] and
            np.array_equal(np.sort(np.r_[train, validation, excluded]),
                           np.arange(50000)) and
            np.array_equal(np.sort(np.r_[checkpoint, temperature, threshold]),
                           validation),
            "complete one-per-coarse class/ID-role partition")
    return {
        "id_fine_labels": id_fine, "ood_fine_labels": ood_fine,
        "fine_to_coarse": fine_to_coarse,
        "train_indices": train, "validation_indices": validation,
        "excluded_ood_train_indices": excluded,
        "checkpoint_indices": checkpoint,
        "temperature_indices": temperature,
        "threshold_indices": threshold,
        "train_original_fine_labels": fine[train],
        "validation_original_fine_labels": fine[validation],
    }


def main() -> None:
    declaration = read(RUN / "declaration.json")
    require(declaration["state"] == "frozen_before_new_class_partition_derivation" and
            declaration["new_class_rng_seeds"] == [2032, 2033] and
            declaration["validation_split_seed"] == 2026 and
            declaration["validation_role_seed"] == 2029,
            "fixed prospective replication declaration")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, "declared source changed: " + name)
    for seed in declaration["new_class_rng_seeds"]:
        require(not (RUN / f"split{seed}").exists(),
                "write-once split directory already exists")
    with (DATA / "train").open("rb") as stream:
        source = pickle.load(stream, encoding="bytes")
    with (DATA / "meta").open("rb") as stream:
        metadata = pickle.load(stream, encoding="bytes")
    fine = np.asarray(source[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(source[b"coarse_labels"], dtype=np.int64)
    require(fine.shape == coarse.shape == (50000,) and
            source[b"data"].shape == (50000, 3072) and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500)) and
            np.array_equal(np.unique(coarse), np.arange(20)) and
            len(metadata[b"fine_label_names"]) == 100 and
            len(metadata[b"coarse_label_names"]) == 20,
            "official CIFAR-100 train/meta census")
    fine_to_coarse = np.full(100, -1, dtype=np.int64)
    for label in range(100):
        parents = np.unique(coarse[fine == label])
        require(len(parents) == 1, "unique fine-to-coarse parent")
        fine_to_coarse[label] = parents[0]
    require(np.array_equal(np.bincount(fine_to_coarse, minlength=20),
                           np.full(20, 5)), "five fine classes per coarse group")
    partitions = {seed: derive(seed, fine, coarse, fine_to_coarse)
                  for seed in declaration["new_class_rng_seeds"]}
    with np.load(ROOT / "runs/cifar100-class-holdout/partition.npz",
                 allow_pickle=False) as old:
        original_ood = old["ood_fine_labels"]
    require(not np.array_equal(partitions[2032]["ood_fine_labels"],
                               original_ood) and
            not np.array_equal(partitions[2033]["ood_fine_labels"],
                               original_ood) and
            not np.array_equal(partitions[2032]["ood_fine_labels"],
                               partitions[2033]["ood_fine_labels"]),
            "fixed RNG partitions collided; preserve and amend before training")
    fine_names = [value.decode("utf-8") for value in metadata[b"fine_label_names"]]
    coarse_names = [value.decode("utf-8") for value in metadata[b"coarse_label_names"]]
    for seed, partition in partitions.items():
        folder = RUN / f"split{seed}"
        folder.mkdir()
        archive = folder / "partition.npz"
        with archive.open("wb") as stream:
            np.savez_compressed(stream, **partition)
        record = {
            "state": "partition_prepared_pending_independent_audit",
            "class_rng_seed": seed,
            "validation_split_seed": 2026,
            "validation_role_seed": 2029,
            "declaration_sha256": sha(RUN / "declaration.json"),
            "protocol_sha256": sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION.md"),
            "source_sha256": sha(Path(__file__)),
            "partition_sha256": sha(archive),
            "train_file_sha256": sha(DATA / "train"),
            "meta_file_sha256": sha(DATA / "meta"),
            "id_fine_labels": partition["id_fine_labels"].tolist(),
            "ood_fine_labels": partition["ood_fine_labels"].tolist(),
            "fine_to_coarse": fine_to_coarse.tolist(),
            "fine_label_names": fine_names,
            "coarse_label_names": coarse_names,
            "counts": {"id_train": 36000, "id_validation": 4000,
                       "held_out_train_excluded": 10000,
                       "checkpoint": 1600, "temperature": 1200,
                       "threshold": 1200},
            "boundary": "Training-file-only partition; official test pixels/labels not read.",
        }
        (folder / "partition.json").write_text(json.dumps(record, indent=2) + "\n",
                                               encoding="utf-8")
        print(json.dumps({"state": record["state"], "class_rng_seed": seed,
                          "ood_fine_labels": record["ood_fine_labels"]}), flush=True)


if __name__ == "__main__":
    main()
