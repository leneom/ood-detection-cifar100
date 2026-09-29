"""Record the complete official CIFAR-10 test rows before 80-class inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.data import load_cifar

RUN = ROOT / "runs/class-holdout-cifar10-extension"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def main() -> None:
    archive = RUN / "test-population.npz"
    manifest_path = RUN / "test-population.json"
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("write-once CIFAR-10 population already exists")
    declaration = json.loads((RUN / "declaration.json").read_text(encoding="utf-8"))
    if declaration["state"] != "frozen_before_80class_cifar10_inference":
        raise ValueError("frozen population declaration required")
    for name, digest in declaration["inputs_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("declared input changed: " + name)
    images, labels = load_cifar(ROOT / "data", "cifar10", False, False)
    pixels = images.numpy()
    original_labels = labels.numpy()
    if (pixels.shape != (10000, 32, 32, 3) or pixels.dtype != np.uint8 or
            original_labels.shape != (10000,) or
            not np.array_equal(np.bincount(original_labels, minlength=10),
                               np.full(10, 1000))):
        raise AssertionError("complete official test population expected")
    original_rows = np.arange(10000, dtype=np.int32)
    with archive.open("wb") as stream:
        np.savez_compressed(stream, original_rows=original_rows,
                            original_labels=original_labels.astype(np.int16))
    record = {
        "state": "population_pending_independent_audit",
        "count": 10000, "class_counts": [1000] * 10,
        "original_row_order": "official CIFAR-10 test rows 0..9999",
        "pixel_shape": list(pixels.shape), "pixel_dtype": str(pixels.dtype),
        "pixel_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
        "source_test_file_sha256": sha(ROOT / "data/cifar-10-batches-py/test_batch"),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "population_sha256": sha(archive),
        "producer_source_sha256": sha(Path(__file__)),
    }
    manifest_path.write_text(json.dumps(record, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"count": 10000, "pixel_sha256": record["pixel_sha256"]}))


if __name__ == "__main__":
    main()
