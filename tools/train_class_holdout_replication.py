"""Source-bound adapter for the unchanged 100-epoch 80-class trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood import class_holdout_train as original
from src.ood.runtime import file_hash

BASE = ROOT / "runs/cifar100-class-holdout-replication"
SOURCES = (
    "src/ood/class_holdout_train.py", "src/ood/data.py", "src/ood/models.py",
    "src/ood/runtime.py", "tools/train_class_holdout_replication.py",
    "tools/prepare_class_holdout_replication.py",
    "tools/audit_class_holdout_replication_partition.py",
    "docs/CLASS_HOLDOUT_REPLICATION.md",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def frozen_partition(class_seed: int) -> tuple[dict[str, np.ndarray], dict]:
    require(class_seed in (2032, 2033), "predeclared class seed only")
    declaration_path = BASE / "declaration.json"
    declaration = read(declaration_path)
    require(declaration["state"] == "frozen_before_new_class_partition_derivation" and
            declaration["new_class_rng_seeds"] == [2032, 2033] and
            declaration["training_seeds"] == [0, 1, 2] and
            declaration["epochs_per_seed"] == 100,
            "frozen replication declaration")
    for name, expected in declaration["inputs_sha256"].items():
        require(file_hash(ROOT / name) == expected, "declared input changed: " + name)
    folder = BASE / f"split{class_seed}"
    manifest_path = folder / "partition.json"
    audit_path = folder / "partition-audit.json"
    archive = folder / "partition.npz"
    manifest = read(manifest_path)
    audit = read(audit_path)
    require(manifest["state"] == "partition_prepared_pending_independent_audit" and
            manifest["class_rng_seed"] == class_seed and
            manifest["declaration_sha256"] == audit["declaration_sha256"] ==
            file_hash(declaration_path) and
            manifest["partition_sha256"] == audit["partition_sha256"] ==
            file_hash(archive) and
            audit["partition_manifest_sha256"] == file_hash(manifest_path) and
            manifest["protocol_sha256"] == file_hash(ROOT / "docs/CLASS_HOLDOUT_REPLICATION.md") and
            manifest["source_sha256"] == audit["producer_source_sha256"] ==
            file_hash(ROOT / "tools/prepare_class_holdout_replication.py") and
            audit["source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_replication_partition.py") and
            manifest["train_file_sha256"] == audit["train_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/train") and
            manifest["meta_file_sha256"] == audit["meta_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/meta") and
            audit["state"] == "passed" and audit["class_rng_seed"] == class_seed,
            "independently audited partition/hash binding")
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    require([len(arrays[name]) for name in
             ("train_indices", "validation_indices", "checkpoint_indices",
              "temperature_indices", "threshold_indices", "excluded_ood_train_indices",
              "id_fine_labels", "ood_fine_labels")] ==
            [36000, 4000, 1600, 1200, 1200, 10000, 80, 20] and
            np.array_equal(np.sort(np.r_[arrays["train_indices"],
                                        arrays["validation_indices"],
                                        arrays["excluded_ood_train_indices"]]),
                           np.arange(50000)) and
            np.array_equal(np.sort(np.r_[arrays["checkpoint_indices"],
                                        arrays["temperature_indices"],
                                        arrays["threshold_indices"]]),
                           arrays["validation_indices"]),
            "partition population and ID roles")
    evidence = {
        "partition_sha256": file_hash(archive),
        "partition_manifest_sha256": file_hash(manifest_path),
        "partition_audit_sha256": file_hash(audit_path),
        "declaration_sha256": file_hash(declaration_path),
        "checkpoint_population": "ID_checkpoint_validation_1600",
    }
    return arrays, evidence


def configure(class_seed: int) -> None:
    # The imported trainer resolves these module globals when train() runs.
    # Its model, optimization, 100-epoch loop, ID selection and resume code stay intact.
    original.RUN = BASE / f"split{class_seed}"
    original.PARTITION = original.RUN / "partition.npz"
    original.SOURCE_NAMES = SOURCES
    original.frozen_partition = lambda: frozen_partition(class_seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    configure(args.class_seed)
    arrays, evidence = frozen_partition(args.class_seed)
    if args.preflight:
        from src.ood.data import load_cifar
        _, fine = load_cifar(ROOT / "data", "cifar100", True, False)
        fine = fine.numpy()
        require(np.array_equal(fine[arrays["train_indices"]],
                               arrays["train_original_fine_labels"]) and
                np.array_equal(fine[arrays["validation_indices"]],
                               arrays["validation_original_fine_labels"]) and
                not np.isin(fine[arrays["train_indices"]],
                              arrays["ood_fine_labels"]).any(),
                "torchvision train-file row/label preflight")
        print(json.dumps({"state": "preflight_passed", "class_seed": args.class_seed,
                          "training_seed": args.seed, "partition": evidence,
                          "source_hashes": {name: file_hash(ROOT / name) for name in SOURCES}}),
              flush=True)
        return
    original.train(args.seed, 100, args.resume)


if __name__ == "__main__":
    main()
