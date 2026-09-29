"""CPU-only fixed six-method scoring of audited 80-class extraction outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from calibration_core import fit_temperature, logit_scores, calibration_metrics
from src.ood.metrics import detection_metrics, id_threshold


RUN = ROOT / "runs" / "cifar100-class-holdout"
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (.80, .90, .95, .99)
SOURCE = ("tools/score_class_holdout.py", "tools/calibration_core.py",
          "src/ood/metrics.py")


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


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def save_arrays(path: Path, **data: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temp.replace(path)


def score_population(logits: np.ndarray, t1: dict[str, np.ndarray],
                     feature: dict[str, np.ndarray], temperature: float) -> dict:
    require(logits.ndim == 2 and logits.shape[1] == 80 and
            set(t1) == {"msp", "energy", "maxlogit", "entropy"} and
            set(feature) == {"knn", "mahalanobis"},
            "80-class logits and six fixed scores required")
    baseline = {**t1, **feature}
    changed = logit_scores(logits, temperature)
    calibrated = {**baseline, **{name: changed[name]
                                   for name in ("msp", "energy", "entropy")}}
    for variant in (baseline, calibrated):
        require(set(variant) == set(METHODS) and
                all(np.asarray(values).shape == (len(logits),) and
                    np.isfinite(values).all() for values in variant.values()),
                "score length, method or finite guard failed")
    for name in ("maxlogit", "knn", "mahalanobis"):
        require(np.array_equal(baseline[name], calibrated[name]),
                "positive temperature changed invariant score: " + name)
    return {"T1": baseline, "calibrated": calibrated}


def operating_grid(id_scores: np.ndarray, ood_scores: np.ndarray,
                   threshold_scores: np.ndarray) -> tuple[dict, dict]:
    """Fixed empirical thresholds from the 1,200-row ID threshold role only."""
    records, masks = {}, {}
    for target in TARGETS:
        threshold = id_threshold(threshold_scores, target)
        key = f"target{int(round(100 * target))}"
        validation_mask = np.asarray(threshold_scores <= threshold, dtype=np.uint8)
        id_mask = np.asarray(id_scores <= threshold, dtype=np.uint8)
        ood_mask = np.asarray(ood_scores <= threshold, dtype=np.uint8)
        masks[key + "_threshold_ID"] = validation_mask
        masks[key + "_test_ID"] = id_mask
        masks[key + "_OOD"] = ood_mask
        records[key] = {
            "target": target, "threshold": threshold,
            "calibration_accepted": int(validation_mask.sum()),
            "calibration_count": len(validation_mask),
            "calibration_acceptance": float(validation_mask.mean()),
            "id_test_accepted": int(id_mask.sum()), "id_test_count": len(id_mask),
            "id_test_acceptance": float(id_mask.mean()),
            "ood_accepted": int(ood_mask.sum()), "ood_count": len(ood_mask),
            "ood_acceptance": float(ood_mask.mean()),
            "calibration_ties_at_threshold": int(np.count_nonzero(threshold_scores == threshold)),
        }
    return records, masks


def preflight(seed: int) -> tuple[Path, dict, dict]:
    raw_dir = RUN / f"extract-seed{seed}"
    status, manifest = read(raw_dir / "status.json"), read(raw_dir / "manifest.json")
    require(status["state"] == manifest["state"] ==
            "extracted_pending_CPU_calibration_and_audit" and
            status["seed"] == manifest["seed"] == seed and
            status["manifest_sha256"] == sha(raw_dir / "manifest.json") and
            manifest["checkpoint_sha256"] ==
            sha(RUN / f"seed{seed}" / "final-best.pt") and
            manifest["training_audit_sha256"] ==
            sha(RUN / f"train-audit-seed{seed}" / "audit.json") and
            manifest["partition_sha256"] == sha(RUN / "partition.npz") and
            manifest["test_population_sha256"] == sha(RUN / "test-population.npz") and
            manifest["test_population_audit_sha256"] ==
            sha(RUN / "test-population-audit.json") and
            manifest["feature_fit"] == {
                "population": "ID optimization training only", "count": 36000,
                "knn_k": 50, "mahalanobis_shrinkage": .01},
            "frozen extraction provenance differs")
    require(all(sha(raw_dir / "source_snapshot" / name) == digest
                for name, digest in manifest["source_hashes"].items()) and
            all(sha(raw_dir / name) == digest
                for name, digest in manifest["outputs"].items()),
            "extraction source snapshot or raw arrays changed")
    feature_audit_path = RUN / f"feature-audit-seed{seed}" / "audit.json"
    feature_audit = read(feature_audit_path)
    require(feature_audit["state"] == "passed" and
            feature_audit["seed"] == seed and
            feature_audit["extraction_manifest_sha256"] ==
            sha(raw_dir / "manifest.json") and
            feature_audit["source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_features.py"),
            "independent ID-only feature fit audit missing or changed")
    return raw_dir, manifest, arrays(RUN / "partition.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    raw_dir, raw_manifest, partition = preflight(seed)
    output = RUN / f"score-seed{seed}"
    require(not output.exists(), "scoring output exists; preserve and inspect it")
    temperature_raw = arrays(raw_dir / "temperature_id_raw.npz")
    threshold_raw = arrays(raw_dir / "threshold_id_raw.npz")
    id_raw = arrays(raw_dir / "id_test_raw.npz")
    ood_raw = arrays(raw_dir / "ood_test_raw.npz")
    test = arrays(RUN / "test-population.npz")
    require(np.array_equal(temperature_raw["original_indices"],
                           partition["temperature_indices"]) and
            np.array_equal(threshold_raw["original_indices"],
                           partition["threshold_indices"]) and
            np.array_equal(id_raw["original_indices"], test["id_test_indices"]) and
            np.array_equal(ood_raw["original_indices"], test["ood_test_indices"]) and
            np.array_equal(id_raw["id_class_targets"], test["id_class_targets"]) and
            "id_class_targets" not in ood_raw and
            not np.intersect1d(temperature_raw["original_indices"],
                               threshold_raw["original_indices"]).size and
            len(temperature_raw["logits"]) == len(threshold_raw["logits"]) == 1200 and
            len(id_raw["logits"]) == 8000 and len(ood_raw["logits"]) == 2000,
            "ID fit/threshold or full test populations differ")
    fit = fit_temperature(temperature_raw["logits"],
                          temperature_raw["id_class_targets"])
    fit["population"] = "1,200 ID temperature-role rows only"
    fit["fitting_role"] = "temperature"
    fit["fitting_count"] = 1200
    temperature = fit["temperature"]
    outputs = {}
    sources = {name: sha(ROOT / name) for name in SOURCE}
    output.mkdir()
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCE:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    fit_path = output / "temperature_fit.json"
    fit_path.write_text(json.dumps(fit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs[fit_path.name] = sha(fit_path)

    def saved(path: Path, **data: np.ndarray) -> None:
        save_arrays(path, **data)
        outputs[path.name] = sha(path)

    populations = {}
    for name, raw in (("temperature_id", temperature_raw),
                      ("threshold_id", threshold_raw),
                      ("id_test", id_raw), ("ood_test", ood_raw)):
        values = score_population(
            raw["logits"], arrays(raw_dir / f"{name}_T1_scores.npz"),
            arrays(raw_dir / f"{name}_feature_scores.npz"), temperature)
        saved(output / f"{name}_scores.npz",
              **{f"{variant}_{method}": values[variant][method]
                 for variant in values for method in METHODS})
        populations[name] = values
    correct = np.asarray(id_raw["predictions"] == id_raw["id_class_targets"],
                         dtype=np.uint8)
    saved(output / "id_test_correctness.npz", correct=correct,
          original_indices=id_raw["original_indices"],
          original_fine_labels=id_raw["original_fine_labels"],
          id_class_targets=id_raw["id_class_targets"],
          predictions=id_raw["predictions"])
    probability = {variant: calibration_metrics(id_raw["logits"],
                                                id_raw["id_class_targets"], value)
                   for variant, value in (("T1", 1.), ("calibrated", temperature))}
    require(probability["T1"]["accuracy"] == probability["calibrated"]["accuracy"],
            "positive temperature changed ID classification accuracy")
    methods, all_masks = {}, {}
    for variant in ("T1", "calibrated"):
        methods[variant] = {}
        for method in METHODS:
            inside = populations["id_test"][variant][method]
            outside = populations["ood_test"][variant][method]
            threshold = populations["threshold_id"][variant][method]
            metrics = detection_metrics(inside, outside, threshold)
            grid, masks = operating_grid(inside, outside, threshold)
            methods[variant][method] = {"metrics": metrics, "operating_grid": grid}
            all_masks.update({f"{variant}_{method}_{name}": mask
                              for name, mask in masks.items()})
    saved(output / "acceptance_masks.npz", **all_masks)
    require(all(sha(ROOT / name) == digest for name, digest in sources.items()),
            "CPU scoring source changed mid-run")
    manifest = {
        "state": "scored_pending_independent_audit", "seed": seed,
        "checkpoint_sha256": raw_manifest["checkpoint_sha256"],
        "extraction_manifest_sha256": sha(raw_dir / "manifest.json"),
        "feature_audit_sha256": sha(RUN / f"feature-audit-seed{seed}" / "audit.json"),
        "partition_sha256": sha(RUN / "partition.npz"),
        "test_population_sha256": sha(RUN / "test-population.npz"),
        "temperature": temperature,
        "temperature_fit_role": "ID_temperature_1200",
        "threshold_fit_role": "ID_threshold_1200",
        "targets": TARGETS, "methods": METHODS,
        "n_id": 8000, "n_ood": 2000, "ood_prevalence": .2,
        "method_results": methods, "probability": probability,
        "id_test_accuracy": float(correct.mean()),
        "source_hashes": sources, "outputs": outputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Frozen class-holdout method parameters; no OOD validation or test-based method selection.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    status_path = output / "status.json"
    status_path.write_text(json.dumps({"state": manifest["state"], "seed": seed,
                                       "manifest_sha256": sha(manifest_path)}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"seed": seed, "state": manifest["state"],
                      "id_accuracy": manifest["id_test_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
