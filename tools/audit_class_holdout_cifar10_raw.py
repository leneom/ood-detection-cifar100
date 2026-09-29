"""Independent CPU audit of frozen 80-class CIFAR-10 logits/features/scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
BASE = ROOT / "runs/cifar100-class-holdout"
sys.path.insert(0, str(ROOT / "tools"))
from audit_class_holdout_features import fit_id, probe_indices, probe_scores
from audit_isolated_scoring import manual_scores


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    output = RUN / f"raw-audit-seed{seed}.json"
    require(not output.exists(), "write-once independent raw audit exists")
    folder = RUN / f"extract-seed{seed}"
    manifest, status = read(folder / "manifest.json"), read(folder / "status.json")
    declaration, population = read(RUN / "declaration.json"), read(RUN / "test-population.json")
    require(manifest["state"] == status["state"] ==
            "extracted_pending_independent_audit" and
            manifest["seed"] == status["seed"] == seed and
            status["manifest_sha256"] == sha(folder / "manifest.json") and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            manifest["test_population_sha256"] == sha(RUN / "test-population.npz") and
            manifest["test_population_audit_sha256"] ==
            sha(RUN / "test-population-audit.json") and
            manifest["pixel_sha256"] == population["pixel_sha256"] and
            manifest["feature_fit"] == {
                "population": "original 36000 optimization-ID only",
                "knn_k": 50, "mahalanobis_shrinkage": .01},
            "complete frozen extraction provenance")
    for name, digest in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == digest, "declared input changed: " + name)
    for name, digest in manifest["source_sha256"].items():
        require(sha(ROOT / name) == digest == sha(folder / "source_snapshot" / name),
                "executed source snapshot changed: " + name)
    require(len(manifest["gpu_guard_samples"]) == 3 and
            all(item["used_mib"] <= 4096 and item["util_percent"] <= 55 and
                item["memory_util_percent"] <= 10 and item["power_w"] <= 55
                for item in manifest["gpu_guard_samples"]),
            "three-sample pre-inference GPU guard")
    for name, digest in manifest["output_sha256"].items():
        require(sha(folder / name) == digest, "raw extraction changed: " + name)
    require(manifest["checkpoint_sha256"] ==
            sha(BASE / f"seed{seed}" / "final-best.pt") and
            manifest["original_extract_manifest_sha256"] ==
            sha(BASE / f"extract-seed{seed}" / "manifest.json") and
            manifest["original_score_manifest_sha256"] ==
            sha(BASE / f"score-seed{seed}" / "manifest.json") and
            manifest["original_score_audit_sha256"] ==
            sha(BASE / f"score-audit-seed{seed}" / "audit.json"),
            "same selected checkpoint and complete original ID score audit")
    with (ROOT / "data/cifar-10-batches-py/test_batch").open("rb") as stream:
        source = pickle.load(stream, encoding="bytes")
    pixels = np.asarray(source[b"data"], dtype=np.uint8).reshape(10000, 3, 32, 32)
    pixels = pixels.transpose(0, 2, 3, 1).copy()
    original_labels = np.asarray(source[b"labels"])
    raw = arrays(folder / "cifar10_raw.npz")
    t1 = arrays(folder / "cifar10_T1_scores.npz")
    feature = arrays(folder / "cifar10_feature_scores.npz")
    require(set(raw) == {"logits", "features", "predictions", "original_rows",
                         "original_labels"} and
            raw["logits"].shape == (10000, 80) and
            raw["features"].shape == (10000, 512) and
            np.isfinite(raw["logits"]).all() and
            np.isfinite(raw["features"]).all() and
            np.array_equal(raw["predictions"], raw["logits"].argmax(axis=1)) and
            np.array_equal(raw["original_rows"], np.arange(10000)) and
            np.array_equal(raw["original_labels"], original_labels) and
            hashlib.sha256(pixels.tobytes()).hexdigest() ==
            manifest["pixel_sha256"] == population["pixel_sha256"],
            "full official CIFAR-10 raw row/pixel/logit/feature audit")
    reference_t1 = manual_scores(raw["logits"], 1.)
    require(set(t1) == set(reference_t1) and
            all(t1[name].shape == (10000,) and np.isfinite(t1[name]).all() and
                np.allclose(t1[name], reference_t1[name], rtol=1e-5,
                            atol=1e-5 if name != "energy" else 1e-4)
                for name in reference_t1),
            "all original T=1 score vectors independently recalculated")
    bank_path = BASE / f"extract-seed{seed}" / "id_train_raw.npz"
    require(sha(bank_path) == manifest["id_train_bank_sha256"],
            "original optimization-ID bank changed")
    bank_data = arrays(bank_path)
    bank, means, precision, scale = fit_id(bank_data["features"],
                                            bank_data["id_class_targets"])
    probe = probe_indices(10000)
    knn, maha = probe_scores(raw["features"][probe], bank, means, precision)
    require(set(feature) == {"knn", "mahalanobis"} and
            all(values.shape == (10000,) and np.isfinite(values).all()
                for values in feature.values()) and
            np.allclose(feature["knn"][probe], knn, rtol=1e-5, atol=1e-4) and
            np.allclose(feature["mahalanobis"][probe], maha,
                        rtol=1e-5, atol=1e-3),
            "ID-bank independent KNN/Mahalanobis probes differ")
    record = {
        "state": "passed", "seed": seed, "count": 10000,
        "pixel_sha256": population["pixel_sha256"],
        "class_counts": np.bincount(original_labels, minlength=10).tolist(),
        "checked_t1_score_vectors": 4,
        "feature_probe_rows": probe.tolist(),
        "knn_max_abs_error": float(np.max(np.abs(feature["knn"][probe] - knn))),
        "mahalanobis_max_abs_error": float(np.max(np.abs(feature["mahalanobis"][probe] - maha))),
        "id_bank_covariance_mean_diagonal": scale,
        "manifest_sha256": sha(folder / "manifest.json"),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "reference_sources_sha256": {
            "fit_and_probes": sha(ROOT / "tools/audit_class_holdout_features.py"),
            "logit_scores": sha(ROOT / "tools/audit_isolated_scoring.py")},
        "boundary": "Complete raw CIFAR-10 pixels/rows/logits and four T1 vectors; ID-training-only feature refit with fixed probes. Full calibration/metric audit remains separate.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "t1_vectors": 4, "feature_probes": len(probe)}), flush=True)


if __name__ == "__main__":
    main()
