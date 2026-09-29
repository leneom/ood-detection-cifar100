"""Independent CPU audit of 80-class raw extraction and ID-only feature fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
POPULATIONS = ("checkpoint_id", "temperature_id", "threshold_id",
               "id_test", "ood_test")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pixels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        source = pickle.load(stream, encoding="bytes")
    data = np.asarray(source[b"data"])
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    fine = np.asarray(source[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(source[b"coarse_labels"], dtype=np.int64)
    return images, fine, coarse


def input_sha(images: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()


def fit_id(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    require(x.shape == (36000, 512) and y.shape == (36000,) and
            np.isfinite(x).all() and
            np.array_equal(np.bincount(y, minlength=80), np.full(80, 450)),
            "36k optimization-ID bank must contain 450 rows for each of 80 classes")
    lengths = np.linalg.norm(x, axis=1)
    require(np.all(lengths > 1e-12), "zero training feature")
    bank = x / lengths[:, None]
    means = np.stack([x[y == label].mean(axis=0) for label in range(80)])
    residual = x - means[y]
    covariance = residual.T @ residual / len(x)
    scale = float(np.diag(covariance).mean())
    require(np.isfinite(scale) and scale > 0, "invalid pooled covariance scale")
    covariance = .99 * covariance + .01 * scale * np.eye(512)
    factor, lower = cho_factor(covariance, lower=True, check_finite=True)
    precision = cho_solve((factor, lower), np.eye(512), check_finite=True)
    return bank, means, precision, scale


def probe_indices(count: int) -> np.ndarray:
    rng = np.random.default_rng(2041)
    return np.unique(np.r_[0, count // 2, count - 1,
                           rng.choice(count, min(12, count), replace=False)])


def probe_scores(query: np.ndarray, bank: np.ndarray,
                 means: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float64)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    distance = np.maximum(0.,
                          np.sum(qn * qn, axis=1, keepdims=True) +
                          np.sum(bank * bank, axis=1)[None] - 2 * qn @ bank.T)
    knn = np.partition(distance, 49, axis=1)[:, 49]
    delta = q[:, None, :] - means[None, :, :]
    maha = np.einsum("ncd,df,ncf->nc", delta, precision, delta,
                     optimize=True).min(axis=1)
    return knn, np.maximum(maha, 0.)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    extraction = RUN / f"extract-seed{seed}"
    destination = RUN / f"feature-audit-seed{seed}"
    require(not destination.exists(), "feature audit exists; preserve it")
    manifest, status = read(extraction / "manifest.json"), read(extraction / "status.json")
    require(status["state"] == manifest["state"] ==
            "extracted_pending_CPU_calibration_and_audit" and
            status["manifest_sha256"] == sha(extraction / "manifest.json") and
            manifest["seed"] == seed and
            manifest["checkpoint_sha256"] ==
            sha(RUN / f"seed{seed}" / "final-best.pt") and
            manifest["training_audit_sha256"] ==
            sha(RUN / f"train-audit-seed{seed}" / "audit.json") and
            manifest["checkpoint_inference_audit_sha256"] ==
            sha(RUN / f"checkpoint-inference-audit-seed{seed}" / "audit.json") and
            manifest["partition_sha256"] == sha(RUN / "partition.npz") and
            manifest["test_population_sha256"] == sha(RUN / "test-population.npz") and
            manifest["test_population_audit_sha256"] ==
            sha(RUN / "test-population-audit.json"),
            "extraction/selected model/partition provenance differs")
    guard = manifest["gpu_guard"]
    require(guard["max_used_mib"] == 4096 and
            guard["max_util_percent"] == 55 and
            guard["max_memory_util_percent"] == 10 and
            guard["max_power_w"] == 55 and
            guard["sample_gap_seconds"] == 2 and
            len(guard["samples"]) == 3 and
            all(row["used_mib"] <= 4096 and row["util_percent"] <= 55 and
                row["memory_util_percent"] <= 10 and row["power_w"] <= 55
                for row in guard["samples"]),
            "three-sample pre-extraction GPU contention guard differs")
    for name, expected in manifest["source_hashes"].items():
        require(sha(extraction / "source_snapshot" / name) == expected ==
                sha(ROOT / name), "extraction source snapshot differs: " + name)
    for name, expected in manifest["outputs"].items():
        require(sha(extraction / name) == expected,
                "raw or feature-score output changed: " + name)
    train_source = ROOT / "data/cifar-100-python/train"
    test_source = ROOT / "data/cifar-100-python/test"
    declaration = read(RUN / "declaration.json")
    test_manifest = read(RUN / "test-population.json")
    require(declaration["train_file_sha256"] == sha(train_source) and
            test_manifest["official_test_file_sha256"] == sha(test_source),
            "official CIFAR-100 source changed")
    train_pixels, train_fine, train_coarse = pixels(train_source)
    test_pixels, test_fine, test_coarse = pixels(test_source)
    partition = arrays(RUN / "partition.npz")
    test = arrays(RUN / "test-population.npz")
    lookup = np.full(100, -1, dtype=np.int64)
    lookup[partition["id_fine_labels"]] = np.arange(80)
    expected_rows = {
        "id_train": (train_pixels, train_fine, train_coarse,
                     partition["train_indices"]),
        "checkpoint_id": (train_pixels, train_fine, train_coarse,
                          partition["checkpoint_indices"]),
        "temperature_id": (train_pixels, train_fine, train_coarse,
                           partition["temperature_indices"]),
        "threshold_id": (train_pixels, train_fine, train_coarse,
                         partition["threshold_indices"]),
        "id_test": (test_pixels, test_fine, test_coarse,
                    test["id_test_indices"]),
        "ood_test": (test_pixels, test_fine, test_coarse,
                     test["ood_test_indices"]),
    }
    raw_data = {}
    for name, (images, fine, coarse, rows) in expected_rows.items():
        raw = arrays(extraction / f"{name}_raw.npz")
        count = len(rows)
        require(raw["logits"].shape == (count, 80) and
                raw["features"].shape == (count, 512) and
                np.isfinite(raw["logits"]).all() and
                np.isfinite(raw["features"]).all() and
                np.array_equal(raw["original_indices"], rows) and
                np.array_equal(raw["original_fine_labels"], fine[rows]) and
                np.array_equal(raw["original_coarse_labels"], coarse[rows]) and
                np.array_equal(raw["predictions"], raw["logits"].argmax(axis=1)) and
                manifest["input_pixels"][name]["sha256"] == input_sha(images[rows]) and
                manifest["input_pixels"][name]["shape"] == [count, 32, 32, 3],
                "raw pixel, row, label, prediction or shape differs: " + name)
        if name == "ood_test":
            require("id_class_targets" not in raw and
                    np.all(lookup[fine[rows]] == -1),
                    "held-out class mapped into ID label space")
        else:
            require(np.array_equal(raw["id_class_targets"], lookup[fine[rows]]) and
                    np.all(raw["id_class_targets"] >= 0),
                    "ID target map or held-out leakage differs: " + name)
        raw_data[name] = raw
    require(np.array_equal(raw_data["id_train"]["original_indices"],
                           partition["train_indices"]) and
            manifest["feature_fit"] == {
                "population": "ID optimization training only", "count": 36000,
                "knn_k": 50, "mahalanobis_shrinkage": .01},
            "feature-fit population or parameters differ")
    bank, means, precision, scale = fit_id(
        raw_data["id_train"]["features"],
        raw_data["id_train"]["id_class_targets"])
    results = {}
    for name in POPULATIONS:
        raw = raw_data[name]
        saved = arrays(extraction / f"{name}_feature_scores.npz")
        rows = probe_indices(len(raw["features"]))
        knn, maha = probe_scores(raw["features"][rows], bank, means, precision)
        require(set(saved) == {"knn", "mahalanobis"} and
                all(values.shape == (len(raw["features"]),) and
                    np.isfinite(values).all() for values in saved.values()) and
                np.allclose(knn, saved["knn"][rows], atol=1e-4, rtol=1e-5) and
                np.allclose(maha, saved["mahalanobis"][rows],
                            atol=1e-3, rtol=1e-5),
                "independent ID-bank feature-score probes differ: " + name)
        results[name] = {
            "population_count": len(raw["features"]),
            "probe_rows": rows.tolist(), "probe_count": len(rows),
            "knn_max_abs_error": float(np.max(np.abs(knn - saved["knn"][rows]))),
            "mahalanobis_max_abs_error": float(
                np.max(np.abs(maha - saved["mahalanobis"][rows]))),
        }
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    record = {
        "state": "passed", "seed": seed,
        "fit_population": "36,000 optimization-ID features only",
        "class_count": 80, "class_count_each": 450, "feature_dimension": 512,
        "normalized_knn_k": 50, "pooled_covariance_shrinkage": .01,
        "pooled_covariance_mean_diagonal": scale,
        "populations": results,
        "extraction_manifest_sha256": sha(extraction / "manifest.json"),
        "test_population_sha256": sha(RUN / "test-population.npz"),
        "source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Exact raw pixel/row and full ID-bank refit with fixed score probes; full score vectors and metrics need separate audit.",
    }
    (destination / "audit.json").write_text(json.dumps(record, indent=2,
                                                       allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "probes": sum(item["probe_count"]
                                    for item in results.values())}), flush=True)


if __name__ == "__main__":
    main()
