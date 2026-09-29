"""Independent NumPy/sklearn audit of CIFAR-10 scores and frozen ID operating points."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
BASE = ROOT / "runs/cifar100-class-holdout"
sys.path.insert(0, str(ROOT / "tools"))
from audit_isolated_scoring import independent_metrics, manual_scores

VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)


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


def close(actual: float, expected: float, message: str, tolerance: float = 1e-10) -> None:
    require(abs(float(actual) - float(expected)) < tolerance,
            f"{message}: {actual} vs {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    output = RUN / f"score-audit-seed{seed}.json"
    require(not output.exists(), "write-once score audit already exists")
    folder = RUN / f"score-seed{seed}"
    raw_folder = RUN / f"extract-seed{seed}"
    old_folder = BASE / f"score-seed{seed}"
    score, status = read(folder / "manifest.json"), read(folder / "status.json")
    old = read(old_folder / "manifest.json")
    raw = read(raw_folder / "manifest.json")
    raw_audit_path = RUN / f"raw-audit-seed{seed}.json"
    raw_audit = read(raw_audit_path)
    require(score["state"] == status["state"] == "scored_pending_independent_audit" and
            score["seed"] == status["seed"] == seed and
            status["manifest_sha256"] == sha(folder / "manifest.json") and
            raw_audit["state"] == "passed" and
            raw_audit["manifest_sha256"] == score["extraction_manifest_sha256"] ==
            sha(raw_folder / "manifest.json") and
            score["raw_audit_sha256"] == sha(raw_audit_path) and
            score["original_score_manifest_sha256"] == sha(old_folder / "manifest.json") and
            score["original_score_audit_sha256"] ==
            sha(BASE / f"score-audit-seed{seed}" / "audit.json") and
            score["original_id_test_scores_sha256"] == sha(old_folder / "id_test_scores.npz") and
            score["original_threshold_id_scores_sha256"] ==
            sha(old_folder / "threshold_id_scores.npz") and
            score["checkpoint_sha256"] == raw["checkpoint_sha256"] ==
            old["checkpoint_sha256"] and
            score["temperature"] == old["temperature"] and
            score["temperature_fit_role"] == "ID_temperature_1200" and
            score["threshold_fit_role"] == "ID_threshold_1200" and
            score["n_id"] == 8000 and score["n_ood"] == 10000 and
            score["ood_prevalence"] == 10000 / 18000 and
            score["methods"] == list(METHODS) and
            score["variants"] == list(VARIANTS) and
            score["targets"] == list(TARGETS),
            "complete original ID-only control and new score provenance")
    for name, digest in score["source_sha256"].items():
        require(sha(ROOT / name) == digest == sha(folder / "source_snapshot" / name),
                "executed scoring source changed: " + name)
    for name, digest in score["output_sha256"].items():
        require(sha(folder / name) == digest, "scored array changed: " + name)
    raw_arrays = arrays(raw_folder / "cifar10_raw.npz")
    t1 = arrays(raw_folder / "cifar10_T1_scores.npz")
    features = arrays(raw_folder / "cifar10_feature_scores.npz")
    observed = arrays(folder / "cifar10_scores.npz")
    masks = arrays(folder / "acceptance_masks.npz")
    inside = arrays(old_folder / "id_test_scores.npz")
    validation = arrays(old_folder / "threshold_id_scores.npz")
    old_masks = arrays(old_folder / "acceptance_masks.npz")
    require(set(observed) == {f"{v}_{m}" for v in VARIANTS for m in METHODS} and
            len(masks) == 48, "twelve complete score vectors and 48 acceptance masks")
    checks = {"score_vectors": 0, "metric_fields": 0,
              "operating_fields": 0, "ood_mask_vectors": 0,
              "frozen_id_mask_vectors": 0}
    for variant in VARIANTS:
        reference_logits = manual_scores(raw_arrays["logits"],
                                         1. if variant == "T1" else old["temperature"])
        for method in METHODS:
            key = f"{variant}_{method}"
            actual = observed[key]
            if method in ("knn", "mahalanobis"):
                require(np.array_equal(actual, features[method]),
                        "training-ID feature vector changed")
            elif method == "maxlogit":
                require(np.array_equal(actual, t1[method]),
                        "temperature-invariant MaxLogit changed")
            else:
                require(np.allclose(actual, reference_logits[method],
                                    rtol=1e-5,
                                    atol=1e-4 if method == "energy" else 1e-5),
                        "independent calibrated logit score vector differs")
            require(actual.shape == (10000,) and np.isfinite(actual).all(),
                    "complete finite OOD score vector")
            checks["score_vectors"] += 1
            original_id = inside[key]
            original_validation = validation[key]
            direct = independent_metrics(original_id, actual, original_validation)
            saved_metrics = score["method_results"][variant][method]["metrics"]
            require(set(saved_metrics) == set(direct), "complete benchmark metric fields")
            for field, value in direct.items():
                close(saved_metrics[field], value, "independent ranking/ID threshold " + field)
                checks["metric_fields"] += 1
            old_metrics = old["method_results"][variant][method]["metrics"]
            for field in ("benchmark_threshold", "validation_threshold",
                          "id_tpr_at_fpr95",
                          "id_test_acceptance_at_validation_threshold",
                          "calibration_id_acceptance"):
                close(saved_metrics[field], old_metrics[field],
                      "unchanged original ID field " + field)
            for target in TARGETS:
                name = f"target{target}"
                old_grid = old["method_results"][variant][method]["operating_grid"][name]
                new_grid = score["method_results"][variant][method]["operating_grid"][name]
                require(set(new_grid) == {"target", "threshold", "calibration_accepted",
                                         "calibration_count", "calibration_acceptance",
                                         "id_test_accepted", "id_test_count",
                                         "id_test_acceptance", "ood_accepted",
                                         "ood_count", "ood_acceptance"},
                        "complete fixed target fields")
                for field in ("target", "threshold", "calibration_accepted",
                              "calibration_count", "calibration_acceptance",
                              "id_test_accepted", "id_test_count",
                              "id_test_acceptance"):
                    close(new_grid[field], old_grid[field],
                          "unchanged original ID operating field " + field)
                    checks["operating_fields"] += 1
                expected_id_mask = np.asarray(original_id <= old_grid["threshold"],
                                              dtype=np.uint8)
                require(np.array_equal(expected_id_mask,
                                       old_masks[f"{key}_{name}_test_ID"]),
                        "original ID acceptance mask and threshold")
                checks["frozen_id_mask_vectors"] += 1
                expected = np.asarray(actual <= old_grid["threshold"],
                                      dtype=np.uint8)
                require(np.array_equal(masks[f"{key}_{name}_OOD"], expected),
                        "complete frozen-threshold CIFAR-10 acceptance mask")
                require(new_grid["ood_accepted"] == int(expected.sum()) and
                        new_grid["ood_count"] == 10000,
                        "complete OOD accepted/count")
                close(new_grid["ood_acceptance"], expected.mean(),
                      "complete OOD acceptance")
                checks["operating_fields"] += 3
                checks["ood_mask_vectors"] += 1
    record = {
        "state": "passed", "seed": seed, "checked": checks,
        "scoring_manifest_sha256": sha(folder / "manifest.json"),
        "raw_audit_sha256": sha(raw_audit_path),
        "audit_source_sha256": sha(Path(__file__)),
        "reference_source_sha256": sha(ROOT / "tools/audit_isolated_scoring.py"),
        "boundary": "All 12 vectors, sklearn ranking metrics and four original ID-validation operating thresholds independently checked; no OOD fitting.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed, "checked": checks}),
          flush=True)


if __name__ == "__main__":
    main()
