"""Independently reconstruct the two frozen, train-file-only class partitions."""
from __future__ import annotations

from datetime import datetime, timezone
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
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def audit(seed: int, fine: np.ndarray, coarse: np.ndarray, mapping: np.ndarray,
          declaration_hash: str, old_ood: np.ndarray) -> tuple[dict, np.ndarray]:
    folder = RUN / f"split{seed}"
    destination = folder / "partition-audit.json"
    require(not destination.exists(), "write-once partition audit already exists")
    manifest = read(folder / "partition.json")
    archive = folder / "partition.npz"
    require(manifest["state"] == "partition_prepared_pending_independent_audit" and
            manifest["class_rng_seed"] == seed and
            manifest["validation_split_seed"] == 2026 and
            manifest["validation_role_seed"] == 2029 and
            manifest["declaration_sha256"] == declaration_hash and
            manifest["protocol_sha256"] == sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION.md") and
            manifest["source_sha256"] == sha(ROOT / "tools/prepare_class_holdout_replication.py") and
            manifest["partition_sha256"] == sha(archive) and
            manifest["train_file_sha256"] == sha(DATA / "train") and
            manifest["meta_file_sha256"] == sha(DATA / "meta"),
            "manifest/source/data hash binding")
    with np.load(archive, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    require(set(values) == {"id_fine_labels", "ood_fine_labels", "fine_to_coarse",
                            "train_indices", "validation_indices",
                            "excluded_ood_train_indices", "checkpoint_indices",
                            "temperature_indices", "threshold_indices",
                            "train_original_fine_labels", "validation_original_fine_labels"},
            "exact partition array schema")
    generator = np.random.default_rng(seed)
    heldout = []
    for parent in range(20):
        children = np.where(mapping == parent)[0]
        require(children.size == 5, "five children per parent")
        heldout.append(int(generator.permutation(children)[0]))
    heldout = np.sort(np.asarray(heldout, dtype=np.int64))
    included = np.asarray(sorted(set(range(100)) - set(heldout)), dtype=np.int64)
    require(not np.array_equal(heldout, old_ood) and
            np.array_equal(heldout, values["ood_fine_labels"]) and
            np.array_equal(included, values["id_fine_labels"]) and
            np.array_equal(mapping, values["fine_to_coarse"]) and
            manifest["ood_fine_labels"] == heldout.tolist() and
            manifest["id_fine_labels"] == included.tolist() and
            manifest["fine_to_coarse"] == mapping.tolist(),
            "class RNG draw and labels independently reconstructed")
    train = values["train_indices"]
    validation = values["validation_indices"]
    excluded = values["excluded_ood_train_indices"]
    require(all(rows.dtype == np.int64 and np.array_equal(rows, np.unique(rows))
                for rows in (train, validation, excluded)) and
            [len(rows) for rows in (train, validation, excluded)] == [36000, 4000, 10000] and
            np.array_equal(np.sort(np.r_[train, validation, excluded]), np.arange(50000)) and
            np.array_equal(excluded, np.where(np.isin(fine, heldout))[0]) and
            np.array_equal(values["train_original_fine_labels"], fine[train]) and
            np.array_equal(values["validation_original_fine_labels"], fine[validation]) and
            np.array_equal(np.bincount(fine[train], minlength=100)[included], np.full(80, 450)) and
            np.array_equal(np.bincount(fine[validation], minlength=100)[included], np.full(80, 50)) and
            not np.isin(fine[train], heldout).any() and
            not np.isin(fine[validation], heldout).any(),
            "all 50k train rows isolated with exact per-class counts")
    split_rng = np.random.default_rng(2026)
    expected_val = []
    for label in included:
        class_rows = np.where(fine == label)[0]
        expected_val.extend(split_rng.permutation(class_rows)[:50])
    require(np.array_equal(validation, np.sort(expected_val)),
            "independent stratified validation regeneration")
    roles = {name: [] for name in ("checkpoint", "temperature", "threshold")}
    role_rng = np.random.default_rng(2029)
    for label in included:
        draw = role_rng.permutation(validation[fine[validation] == label])
        roles["checkpoint"].extend(draw[:20])
        roles["temperature"].extend(draw[20:35])
        roles["threshold"].extend(draw[35:])
    for name, count in (("checkpoint", 20), ("temperature", 15), ("threshold", 15)):
        rows = values[name + "_indices"]
        require(rows.dtype == np.int64 and
                np.array_equal(rows, np.sort(roles[name])) and
                len(rows) == 80 * count and
                np.array_equal(np.bincount(fine[rows], minlength=100)[included],
                               np.full(80, count)),
                name + " ID role regeneration")
    require(np.array_equal(np.sort(np.concatenate([values[name + "_indices"]
                       for name in roles])), validation), "roles disjoint and exhaustive")
    record = {
        "state": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
        "class_rng_seed": seed, "declaration_sha256": declaration_hash,
        "partition_sha256": sha(archive),
        "partition_manifest_sha256": sha(folder / "partition.json"),
        "producer_source_sha256": sha(ROOT / "tools/prepare_class_holdout_replication.py"),
        "source_sha256": sha(Path(__file__)),
        "train_file_sha256": sha(DATA / "train"), "meta_file_sha256": sha(DATA / "meta"),
        "counts": {"id_classes": 80, "held_out_classes": 20,
                   "id_train": 36000, "id_validation": 4000,
                   "excluded_ood_train": 10000, "checkpoint": 1600,
                   "temperature": 1200, "threshold": 1200},
        "boundary": "Independent class, row and role regeneration from official training file only; no official test file read.",
    }
    return record, heldout


def main() -> None:
    declaration = read(RUN / "declaration.json")
    require(declaration["state"] == "frozen_before_new_class_partition_derivation" and
            declaration["new_class_rng_seeds"] == [2032, 2033] and
            declaration["validation_split_seed"] == 2026 and
            declaration["validation_role_seed"] == 2029,
            "frozen replication design")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, "declared input differs: " + name)
    with (DATA / "train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    fine = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(raw[b"coarse_labels"], dtype=np.int64)
    require(fine.shape == coarse.shape == (50000,) and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500)) and
            np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 2500)),
            "official train census")
    mapping = np.asarray([np.unique(coarse[fine == label]).item() for label in range(100)],
                         dtype=np.int64)
    require(np.array_equal(np.bincount(mapping, minlength=20), np.full(20, 5)),
            "fine-to-coarse mapping")
    with np.load(ROOT / "runs/cifar100-class-holdout/partition.npz",
                 allow_pickle=False) as original:
        old_ood = original["ood_fine_labels"]
    staged = {}
    for seed in (2032, 2033):
        staged[seed] = audit(seed, fine, coarse, mapping,
                             sha(RUN / "declaration.json"), old_ood)
    require(not np.array_equal(staged[2032][1], staged[2033][1]),
            "new class draws distinct")
    for seed, (record, _) in staged.items():
        destination = RUN / f"split{seed}/partition-audit.json"
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"state": "passed", "class_rng_seed": seed,
                          "partition_sha256": record["partition_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
