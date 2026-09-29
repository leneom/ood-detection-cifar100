"""Fit ID-only temperature/thresholds and score saved isolation outputs on CPU."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.ood.metrics import detection_metrics, id_threshold
from calibration_core import fit_temperature, logit_scores, calibration_metrics


PHASE = ROOT / "runs" / "validation-isolation"
NATURAL = ("cifar10", "mnist", "svhn", "textures")
OOD = NATURAL + ("noise",)
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (.80, .90, .95, .99)
SOURCES = ("tools/score_isolated_eval.py", "tools/calibration_core.py", "src/ood/metrics.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def save_arrays(path, **contents):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **contents)
    temp.replace(path)


def score_population(raw_logits, t1, feature, temperature):
    """Exact extractor T=1 controls; only MSP/Energy/Entropy receive ID-fitted T."""
    if set(t1) != {"msp", "energy", "maxlogit", "entropy"} or set(feature) != {"knn", "mahalanobis"}:
        raise ValueError("six fixed score columns required")
    baseline = {**t1, **feature}
    calibrated_logits = logit_scores(raw_logits, temperature)
    calibrated = {**baseline, **{m: calibrated_logits[m] for m in ("msp", "energy", "entropy")}}
    for variant in (baseline, calibrated):
        if set(variant) != set(METHODS) or any(len(v) != len(raw_logits) or not np.isfinite(v).all() for v in variant.values()):
            raise ValueError("score length/finite guard failed")
    for method in ("maxlogit", "knn", "mahalanobis"):
        if not np.array_equal(baseline[method], calibrated[method]):
            raise ValueError("temperature changed an invariant control")
    return {"T1": baseline, "calibrated": calibrated}


def operating_grid(id_scores, ood_scores, threshold_scores):
    """The only threshold source is the 1,500-row ID threshold role."""
    records, masks = {}, {}
    for target in TARGETS:
        threshold = id_threshold(threshold_scores, target)
        key = f"target{int(round(target * 100))}"
        cal_mask = np.asarray(threshold_scores <= threshold, dtype=np.uint8)
        id_mask = np.asarray(id_scores <= threshold, dtype=np.uint8)
        ood_mask = np.asarray(ood_scores <= threshold, dtype=np.uint8)
        masks[key + "_threshold_ID"] = cal_mask
        masks[key + "_test_ID"] = id_mask
        masks[key + "_OOD"] = ood_mask
        records[key] = {"target": target, "threshold": threshold,
                        "calibration_accepted": int(cal_mask.sum()), "calibration_count": len(cal_mask),
                        "calibration_acceptance": float(cal_mask.mean()),
                        "id_test_accepted": int(id_mask.sum()), "id_test_count": len(id_mask),
                        "id_test_acceptance": float(id_mask.mean()),
                        "ood_accepted": int(ood_mask.sum()), "ood_count": len(ood_mask),
                        "ood_acceptance": float(ood_mask.mean()),
                        "calibration_ties_at_threshold": int(np.count_nonzero(threshold_scores == threshold))}
    return records, masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = p.parse_args()
    raw_dir = PHASE / f"extract-seed{args.seed}"
    raw_status = read(raw_dir / "status.json")
    raw = read(raw_dir / "manifest.json")
    if (raw_status["state"] != "extracted_pending_CPU_calibration_and_audit" or
            raw_status["manifest_sha256"] != file_hash(raw_dir / "manifest.json") or raw["seed"] != args.seed):
        raise ValueError("frozen extraction incomplete or changed")
    if file_hash(Path(raw["checkpoint"])) != raw["checkpoint_sha256"]:
        raise ValueError("frozen classifier checkpoint changed since extraction")
    if file_hash(PHASE / "partition.npz") != raw["validation_roles_partition_sha256"]:
        raise ValueError("frozen ID-validation partition changed since extraction")
    for name, expected in raw["source_hashes"].items():
        if file_hash(raw_dir / "source_snapshot" / name) != expected:
            raise ValueError("executed extraction source snapshot differs: " + name)
    for name, expected in raw["outputs"].items():
        if file_hash(raw_dir / name) != expected:
            raise ValueError("extracted raw output changed: " + name)
    output = PHASE / f"score-seed{args.seed}"
    if output.exists():
        raise FileExistsError("scoring output exists; preserve and inspect before another attempt")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    partition = arrays(PHASE / "partition.npz")
    if raw["validation_roles"] != {name: partition[name + "_rows"].tolist()
                                   for name in ("checkpoint", "temperature", "threshold")}:
        raise ValueError("extraction validation roles differ from frozen partition")
    val_raw = arrays(raw_dir / "id_validation_raw.npz")
    if (not np.array_equal(val_raw["original_indices"], partition["original_validation_indices"]) or
            not np.array_equal(val_raw["labels"], partition["validation_labels"])):
        raise ValueError("validation logits/labels do not align to disjoint roles")
    fit_rows, threshold_rows = partition["temperature_rows"], partition["threshold_rows"]
    if len(fit_rows) != 1500 or len(threshold_rows) != 1500 or np.intersect1d(fit_rows, threshold_rows).size:
        raise ValueError("temperature and threshold roles are not disjoint 1500-row groups")
    test_raw = arrays(raw_dir / "id_test_raw.npz")
    fit = fit_temperature(val_raw["logits"][fit_rows], val_raw["labels"][fit_rows])
    fit["population"] = "1500 ID temperature-fit rows only"
    fit["fitting_role"] = "temperature"
    fit["fitting_count"] = 1500
    temperature = fit["temperature"]
    output.mkdir()
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    write(output / "temperature_fit.json", fit)
    id_scores = score_population(test_raw["logits"], arrays(raw_dir / "id_test_T1_scores.npz"),
                                 arrays(raw_dir / "id_test_feature_scores.npz"), temperature)
    val_scores = score_population(val_raw["logits"], arrays(raw_dir / "id_validation_T1_scores.npz"),
                                  arrays(raw_dir / "id_validation_feature_scores.npz"), temperature)
    outputs = {"temperature_fit.json": file_hash(output / "temperature_fit.json")}

    def saved(path, **content):
        save_arrays(path, **content)
        outputs[path.name] = file_hash(path)

    for name, values in (("id_test", id_scores), ("id_validation", val_scores)):
        saved(output / f"{name}_scores.npz",
              **{f"{variant}_{method}": values[variant][method]
                 for variant in values for method in METHODS})
    correct = np.asarray(test_raw["logits"].argmax(axis=1) == test_raw["labels"], dtype=np.uint8)
    saved(output / "id_test_correctness.npz", correct=correct, labels=test_raw["labels"],
          predictions=test_raw["logits"].argmax(axis=1))
    probability = {variant: calibration_metrics(test_raw["logits"], test_raw["labels"], t)
                   for variant, t in (("T1", 1.), ("calibrated", temperature))}
    if probability["T1"]["accuracy"] != probability["calibrated"]["accuracy"]:
        raise ValueError("positive temperature changed classifier accuracy")
    datasets = {}
    for population in OOD:
        raw_ood = arrays(raw_dir / f"{population}_raw.npz")
        outside = score_population(raw_ood["logits"], arrays(raw_dir / f"{population}_T1_scores.npz"),
                                   arrays(raw_dir / f"{population}_feature_scores.npz"), temperature)
        saved(output / f"{population}_scores.npz",
              **{f"{variant}_{method}": outside[variant][method]
                 for variant in outside for method in METHODS})
        methods, all_masks = {}, {}
        for variant in ("T1", "calibrated"):
            methods[variant] = {}
            for method in METHODS:
                inside_score = id_scores[variant][method]
                ood_score = outside[variant][method]
                threshold_score = val_scores[variant][method][threshold_rows]
                metrics = detection_metrics(inside_score, ood_score, threshold_score)
                grid, masks = operating_grid(inside_score, ood_score, threshold_score)
                methods[variant][method] = {"metrics": metrics, "operating_grid": grid}
                all_masks.update({f"{variant}_{method}_{name}": mask for name, mask in masks.items()})
        saved(output / f"{population}_acceptance_masks.npz", **all_masks)
        datasets[population] = {"kind": raw["population_descriptions"][population],
                                "n_ood": len(raw_ood["logits"]), "methods": methods}
        print(json.dumps({"seed": args.seed, "population": population,
                          "n_ood": len(raw_ood["logits"])}), flush=True)
    if any(file_hash(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError("scoring source changed during execution")
    manifest = {"state": "scored_pending_independent_audit", "seed": args.seed,
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "extraction_manifest_sha256": file_hash(raw_dir / "manifest.json"),
                "partition_sha256": file_hash(PHASE / "partition.npz"),
                "temperature": temperature, "temperature_fit_role": "ID_temperature_1500",
                "threshold_fit_role": "ID_threshold_1500", "targets": TARGETS,
                "methods": METHODS, "datasets": datasets, "probability": probability,
                "id_test_accuracy": float(correct.mean()), "id_test_count": len(correct),
                "source_hashes": source_hashes, "outputs": outputs,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "boundary": "Posthoc existing OOD test populations remain exploratory; noise is diagnostic."}
    write(output / "manifest.json", manifest)
    write(output / "status.json", {"state": "scored_pending_independent_audit",
          "seed": args.seed, "manifest_sha256": file_hash(output / "manifest.json")})
    print(json.dumps({"seed": args.seed, "state": "scored_pending_independent_audit"}), flush=True)


if __name__ == "__main__":
    main()
