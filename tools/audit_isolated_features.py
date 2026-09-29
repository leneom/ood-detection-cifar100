"""Independently refit isolated KNN/Mahalanobis from saved training-ID features.

This CPU audit checks the complete 45k fit population and fixed query probes
per saved ID/OOD population. It does not use OOD rows for either fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.linalg import cho_factor, cho_solve


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "validation-isolation"
POPULATIONS = ("id_validation", "id_test", "cifar10", "mnist", "svhn", "textures", "noise")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def fit_training_id(features, labels):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape != (45000, 512) or y.shape != (45000,) or not np.isfinite(x).all():
        raise ValueError("fixed 45,000-row pooled training-ID feature bank required")
    if not np.array_equal(np.bincount(y, minlength=100), np.full(100, 450)):
        raise ValueError("training-ID class counts differ")
    lengths = np.linalg.norm(x, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("zero training feature")
    knn_bank = x / lengths[:, None]
    means = np.stack([x[y == label].mean(axis=0) for label in range(100)])
    residual = x - means[y]
    covariance = residual.T @ residual / len(x)
    scale = float(np.mean(np.diag(covariance)))
    covariance *= .99
    covariance.flat[::513] += .01 * scale
    factor, lower = cho_factor(covariance, lower=True, check_finite=True)
    precision = cho_solve((factor, lower), np.eye(x.shape[1]))
    return knn_bank, means, precision, scale


def probe_rows(n):
    rng = np.random.default_rng(2041)
    return np.unique(np.r_[0, n // 2, n - 1, rng.choice(n, min(12, n), replace=False)])


def score_probes(query, bank, means, precision):
    q = np.asarray(query, dtype=np.float64)
    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
    distance = np.maximum(0., np.sum(q_norm ** 2, axis=1, keepdims=True) +
                          np.sum(bank ** 2, axis=1)[None] - 2 * q_norm @ bank.T)
    kth = np.partition(distance, 49, axis=1)[:, 49]
    delta = q[:, None, :] - means[None, :, :]
    maha = np.einsum("ncd,df,ncf->nc", delta, precision, delta, optimize=True)
    return kth, np.maximum(0., maha.min(axis=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = p.parse_args().seed
    extraction = PHASE / f"extract-seed{seed}"
    manifest = read(extraction / "manifest.json")
    status = read(extraction / "status.json")
    if (status["state"] != "extracted_pending_CPU_calibration_and_audit" or
            status["manifest_sha256"] != digest(extraction / "manifest.json") or manifest["seed"] != seed):
        raise ValueError("complete frozen extraction required")
    for name, expected in manifest["outputs"].items():
        if digest(extraction / name) != expected:
            raise ValueError("extracted output changed: " + name)
    destination = PHASE / f"feature-audit-seed{seed}"
    if destination.exists():
        raise FileExistsError("feature audit exists; inspect before another attempt")
    train = arrays(extraction / "id_train_raw.npz")
    original = arrays(ROOT / f"runs/cifar100-seed{seed}/split_indices.npz")
    if not np.array_equal(train["original_indices"], original["train"]):
        raise ValueError("feature fit did not use exactly the original 45k ID training indices")
    bank, means, precision, scale = fit_training_id(train["features"], train["labels"])
    results = {}
    for population in POPULATIONS:
        raw = arrays(extraction / f"{population}_raw.npz")
        saved = arrays(extraction / f"{population}_feature_scores.npz")
        rows = probe_rows(len(raw["features"]))
        knn, maha = score_probes(raw["features"][rows], bank, means, precision)
        if len(saved["knn"]) != len(raw["features"]) or len(saved["mahalanobis"]) != len(raw["features"]):
            raise ValueError("feature-score population length mismatch")
        knn_error = np.abs(knn - saved["knn"][rows])
        maha_error = np.abs(maha - saved["mahalanobis"][rows])
        if not np.allclose(knn, saved["knn"][rows], atol=1e-4, rtol=1e-5):
            raise ValueError("independent normalized 50-NN probe scores differ: " + population)
        if not np.allclose(maha, saved["mahalanobis"][rows], atol=1e-3, rtol=1e-5):
            raise ValueError("independent pooled Mahalanobis probe scores differ: " + population)
        results[population] = {"population_count": len(raw["features"]),
                               "probe_rows": rows.tolist(), "probe_count": len(rows),
                               "knn_max_abs_error": float(knn_error.max()),
                               "mahalanobis_max_abs_error": float(maha_error.max())}
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    record = {"state": "passed", "seed": seed,
              "fit_population": "45,000 training ID features only", "classes": 100,
              "class_count_each": 450, "feature_dimension": 512,
              "normalized_knn_k": 50, "pooled_covariance_shrinkage": .01,
              "pooled_covariance_mean_diagonal": scale,
              "populations": results, "source_sha256": digest(__file__),
              "extraction_manifest_sha256": digest(extraction / "manifest.json"),
              "audited_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "Full training-ID refit with fixed independent query probes; complete score vectors are checked by the separate scoring audit."}
    (destination / "audit.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "probes": sum(v["probe_count"] for v in results.values())}), flush=True)


if __name__ == "__main__":
    main()
