"""Independently reconstruct CIFAR-10 test rows directly from its batch file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def main() -> None:
    output = RUN / "test-population-audit.json"
    if output.exists():
        raise FileExistsError("write-once independent test audit exists")
    declaration = json.loads((RUN / "declaration.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "test-population.json").read_text(encoding="utf-8"))
    batch_path = ROOT / "data/cifar-10-batches-py/test_batch"
    with batch_path.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    pixels = np.asarray(batch[b"data"], dtype=np.uint8).reshape(10000, 3, 32, 32)
    pixels = pixels.transpose(0, 2, 3, 1).copy()
    labels = np.asarray(batch[b"labels"], dtype=np.int16)
    with np.load(RUN / "test-population.npz", allow_pickle=False) as source:
        stored = {name: source[name] for name in source.files}
    assert set(stored) == {"original_rows", "original_labels"}
    assert pixels.shape == (10000, 32, 32, 3) and pixels.dtype == np.uint8
    assert np.array_equal(stored["original_rows"], np.arange(10000))
    assert np.array_equal(stored["original_labels"], labels)
    assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 1000))
    assert manifest["state"] == "population_pending_independent_audit"
    assert manifest["count"] == 10000 and manifest["class_counts"] == [1000] * 10
    assert manifest["pixel_shape"] == list(pixels.shape)
    assert manifest["pixel_dtype"] == "uint8"
    assert manifest["pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert manifest["source_test_file_sha256"] == sha(batch_path)
    assert manifest["declaration_sha256"] == sha(RUN / "declaration.json")
    assert manifest["population_sha256"] == sha(RUN / "test-population.npz")
    assert manifest["producer_source_sha256"] == sha(ROOT / "tools/prepare_class_holdout_cifar10.py")
    assert declaration["inputs_sha256"]["data/cifar-10-batches-py/test_batch"] == sha(batch_path)
    for name, expected in declaration["inputs_sha256"].items():
        assert sha(ROOT / name) == expected, "declared input changed: " + name
    record = {
        "state": "passed", "count": 10000, "class_counts": [1000] * 10,
        "pixel_sha256": manifest["pixel_sha256"],
        "test_population_manifest_sha256": sha(RUN / "test-population.json"),
        "test_population_sha256": sha(RUN / "test-population.npz"),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "boundary": "Direct official test-batch reconstruction; no model inference or OOD score.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "count": 10000}))


if __name__ == "__main__":
    main()
