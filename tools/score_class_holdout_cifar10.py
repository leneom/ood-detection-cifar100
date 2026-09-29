"""Score all CIFAR-10 rows with original 80-class ID-only calibrations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
BASE = ROOT / "runs/cifar100-class-holdout"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from score_class_holdout import score_population
from src.ood.metrics import detection_metrics


VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)
SOURCES = ("tools/score_class_holdout_cifar10.py",
           "tools/score_class_holdout.py", "tools/calibration_core.py",
           "src/ood/metrics.py")


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


def save(path: Path, **data: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def run(seed: int) -> None:
    folder = RUN / f"score-seed{seed}"
    require(not folder.exists(), "write-once CIFAR-10 score output exists")
    raw_dir = RUN / f"extract-seed{seed}"
    raw_audit_path = RUN / f"raw-audit-seed{seed}.json"
    raw_audit = read(raw_audit_path)
    raw_manifest = read(raw_dir / "manifest.json")
    baseline_dir = BASE / f"score-seed{seed}"
    baseline = read(baseline_dir / "manifest.json")
    baseline_audit = read(BASE / f"score-audit-seed{seed}" / "audit.json")
    require(raw_audit["state"] == "passed" and raw_audit["seed"] == seed and
            raw_audit["manifest_sha256"] == sha(raw_dir / "manifest.json") and
            raw_audit["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_cifar10_raw.py") and
            baseline_audit["state"] == "passed" and
            baseline_audit["scoring_manifest_sha256"] == sha(baseline_dir / "manifest.json") and
            raw_manifest["original_score_manifest_sha256"] ==
            sha(baseline_dir / "manifest.json") and
            raw_manifest["checkpoint_sha256"] == baseline["checkpoint_sha256"] and
            baseline["temperature_fit_role"] == "ID_temperature_1200" and
            baseline["threshold_fit_role"] == "ID_threshold_1200" and
            baseline["methods"] == list(METHODS) and
            baseline["targets"] == [.8, .9, .95, .99] and
            baseline["n_id"] == 8000 and baseline["n_ood"] == 2000,
            "frozen ID-only model, calibration and raw audit required")
    raw = arrays(raw_dir / "cifar10_raw.npz")
    t1 = arrays(raw_dir / "cifar10_T1_scores.npz")
    features = arrays(raw_dir / "cifar10_feature_scores.npz")
    original_id = arrays(baseline_dir / "id_test_scores.npz")
    original_threshold = arrays(baseline_dir / "threshold_id_scores.npz")
    require(raw["logits"].shape == (10000, 80) and
            set(original_id) == set(original_threshold) ==
            {f"{variant}_{method}" for variant in VARIANTS for method in METHODS},
            "complete original ID and new CIFAR-10 score shapes")
    populations = score_population(raw["logits"], t1, features,
                                   baseline["temperature"])
    outputs = {f"{variant}_{method}": populations[variant][method]
               for variant in VARIANTS for method in METHODS}
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    folder.mkdir()
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    score_path = folder / "cifar10_scores.npz"
    mask_path = folder / "acceptance_masks.npz"
    save(score_path, **outputs)
    masks = {}
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for method in METHODS:
            key = f"{variant}_{method}"
            inside, outside, threshold_id = (original_id[key], outputs[key],
                                              original_threshold[key])
            require(inside.shape == (8000,) and outside.shape == (10000,) and
                    threshold_id.shape == (1200,) and
                    np.isfinite(inside).all() and np.isfinite(outside).all(),
                    "frozen original ID and full OOD vectors")
            metrics = detection_metrics(inside, outside, threshold_id)
            old = baseline["method_results"][variant][method]
            require(metrics["benchmark_threshold"] ==
                    old["metrics"]["benchmark_threshold"] and
                    metrics["validation_threshold"] ==
                    old["metrics"]["validation_threshold"] and
                    metrics["id_test_acceptance_at_validation_threshold"] ==
                    old["metrics"]["id_test_acceptance_at_validation_threshold"],
                    "original ID benchmark/validation threshold changed")
            operating = {}
            for target in TARGETS:
                name = f"target{target}"
                old_grid = old["operating_grid"][name]
                threshold = old_grid["threshold"]
                accepted = np.asarray(outside <= threshold, dtype=np.uint8)
                masks[f"{key}_{name}_OOD"] = accepted
                operating[name] = {
                    "target": target / 100, "threshold": threshold,
                    "calibration_accepted": old_grid["calibration_accepted"],
                    "calibration_count": 1200,
                    "calibration_acceptance": old_grid["calibration_acceptance"],
                    "id_test_accepted": old_grid["id_test_accepted"],
                    "id_test_count": 8000,
                    "id_test_acceptance": old_grid["id_test_acceptance"],
                    "ood_accepted": int(accepted.sum()), "ood_count": 10000,
                    "ood_acceptance": float(accepted.mean()),
                }
            results[variant][method] = {"metrics": metrics,
                                        "operating_grid": operating}
    save(mask_path, **masks)
    require(all(sha(ROOT / name) == digest for name, digest in source_hashes.items()),
            "score source changed during computation")
    record = {
        "state": "scored_pending_independent_audit", "seed": seed,
        "checkpoint_sha256": raw_manifest["checkpoint_sha256"],
        "extraction_manifest_sha256": sha(raw_dir / "manifest.json"),
        "raw_audit_sha256": sha(raw_audit_path),
        "original_score_manifest_sha256": sha(baseline_dir / "manifest.json"),
        "original_score_audit_sha256": sha(BASE / f"score-audit-seed{seed}" / "audit.json"),
        "original_id_test_scores_sha256": sha(baseline_dir / "id_test_scores.npz"),
        "original_threshold_id_scores_sha256": sha(baseline_dir / "threshold_id_scores.npz"),
        "temperature": baseline["temperature"],
        "temperature_fit_role": baseline["temperature_fit_role"],
        "threshold_fit_role": baseline["threshold_fit_role"],
        "n_id": 8000, "n_ood": 10000, "ood_prevalence": 10000 / 18000,
        "methods": list(METHODS), "variants": list(VARIANTS),
        "targets": list(TARGETS), "method_results": results,
        "output_sha256": {score_path.name: sha(score_path),
                          mask_path.name: sha(mask_path)},
        "source_sha256": source_hashes,
        "boundary": "Original ID scores, temperature and four ID-validation thresholds reused unchanged; full official CIFAR-10 test only evaluated.",
    }
    (folder / "manifest.json").write_text(json.dumps(record, indent=2,
                                                    allow_nan=False) + "\n",
                                          encoding="utf-8")
    (folder / "status.json").write_text(json.dumps({
        "state": record["state"], "seed": seed,
        "manifest_sha256": sha(folder / "manifest.json")}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"state": record["state"], "seed": seed,
                      "methods": 12, "ood_count": 10000}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run(parser.parse_args().seed)


if __name__ == "__main__":
    main()
