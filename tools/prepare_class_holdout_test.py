"""Freeze official CIFAR-100 test row populations for the audited class holdout.

This CPU-only step reads labels/pixels after the training-side partition audit.
It computes no model output, score, metric, or selection criterion.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
TEST = ROOT / "data" / "cifar-100-python" / "test"
PROTOCOL = ROOT / "docs" / "CIFAR100_CLASS_HOLDOUT.md"


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
        raise ValueError(message)


def main() -> None:
    output = RUN / "test-population.npz"
    manifest_path = RUN / "test-population.json"
    require(not output.exists() and not manifest_path.exists(),
            "test population already exists; preserve and audit it")
    declaration = read(RUN / "declaration.json")
    partition_manifest = read(RUN / "partition.json")
    partition_audit = read(RUN / "partition-audit.json")
    require(declaration["state"] == "declared_before_class_partition" and
            declaration["protocol_sha256"] == sha(PROTOCOL) and
            partition_manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            partition_manifest["partition_sha256"] == partition_audit["partition_sha256"] ==
            sha(RUN / "partition.npz") and
            partition_audit["partition_manifest_sha256"] == sha(RUN / "partition.json") and
            partition_audit["state"] == "passed", "training-side partition is not frozen and audited")
    with np.load(RUN / "partition.npz", allow_pickle=False) as stored:
        id_fine = stored["id_fine_labels"].astype(np.int64)
        ood_fine = stored["ood_fine_labels"].astype(np.int64)
        fine_to_coarse = stored["fine_to_coarse"].astype(np.int64)
    require(len(id_fine) == 80 and len(ood_fine) == 20 and
            np.array_equal(np.sort(np.r_[id_fine, ood_fine]), np.arange(100)) and
            np.array_equal(id_fine, np.sort(id_fine)), "80/20 class partition differs")
    with TEST.open("rb") as stream:
        test = pickle.load(stream, encoding="bytes")
    pixels = np.asarray(test[b"data"])
    fine = np.asarray(test[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(test[b"coarse_labels"], dtype=np.int64)
    require(pixels.shape == (10000, 3072) and pixels.dtype == np.uint8 and
            fine.shape == coarse.shape == (10000,) and
            np.all((0 <= fine) & (fine < 100)) and
            np.all((0 <= coarse) & (coarse < 20)) and
            np.array_equal(np.bincount(fine, minlength=100), np.full(100, 100)) and
            np.array_equal(np.bincount(coarse, minlength=20), np.full(20, 500)) and
            np.array_equal(fine_to_coarse[fine], coarse),
            "official test class counts or train-derived fine/coarse mapping differ")
    id_rows = np.flatnonzero(np.isin(fine, id_fine)).astype(np.int64)
    ood_rows = np.flatnonzero(np.isin(fine, ood_fine)).astype(np.int64)
    id_targets = np.searchsorted(id_fine, fine[id_rows]).astype(np.int64)
    require(len(id_rows) == 8000 and len(ood_rows) == 2000 and
            np.array_equal(np.sort(np.r_[id_rows, ood_rows]), np.arange(10000)) and
            np.array_equal(id_fine[id_targets], fine[id_rows]) and
            np.array_equal(np.bincount(id_targets, minlength=80), np.full(80, 100)),
            "fixed 8000/2000 original-order test rows differ")
    temp = output.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, id_test_indices=id_rows, ood_test_indices=ood_rows,
                            id_original_fine_labels=fine[id_rows],
                            ood_original_fine_labels=fine[ood_rows],
                            id_original_coarse_labels=coarse[id_rows],
                            ood_original_coarse_labels=coarse[ood_rows],
                            id_class_targets=id_targets)
    temp.replace(output)
    manifest = {
        "state": "test_population_prepared_pending_independent_audit",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(PROTOCOL),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "partition_sha256": sha(RUN / "partition.npz"),
        "partition_audit_sha256": sha(RUN / "partition-audit.json"),
        "official_test_file_sha256": sha(TEST),
        "test_population_sha256": sha(output),
        "preparation_source_sha256": sha(Path(__file__)),
        "id_count": len(id_rows), "ood_count": len(ood_rows),
        "ood_prevalence": len(ood_rows) / len(fine),
        "id_fine_labels": id_fine.tolist(), "ood_fine_labels": ood_fine.tolist(),
        "boundary": "Original test row populations only; no model output, OOD score, metric or parameter selection.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({"state": manifest["state"], "id": len(id_rows),
                      "ood": len(ood_rows)}), flush=True)


if __name__ == "__main__":
    main()
