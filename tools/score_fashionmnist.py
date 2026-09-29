"""Score Fashion-MNIST OOD only with frozen ID-calibrated inputs."""
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
RUN = ROOT / "runs" / "fashionmnist-extension"
ISOLATED = ROOT / "runs" / "validation-isolation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from calibration_core import logit_scores
from src.ood.metrics import detection_metrics, id_threshold


METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
TARGETS = (80, 90, 95, 99)
PRIOR_POPULATIONS = ("cifar10", "mnist", "svhn", "textures", "noise")
SOURCES = ("tools/score_fashionmnist.py", "tools/calibration_core.py",
           "src/ood/metrics.py", "docs/FASHIONMNIST_EXTENSION.md",
           "tools/prepare_fashionmnist.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def save(path, **content):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **content)
    temp.replace(path)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    raw_folder = RUN / f"extract-seed{seed}"
    raw_manifest, raw_status = read(raw_folder / "manifest.json"), read(raw_folder / "status.json")
    feature_audit = read(RUN / f"feature-audit-seed{seed}" / "audit.json")
    if (raw_status["state"] != "extracted_pending_independent_audit_and_scoring" or
            raw_status["manifest_sha256"] != sha(raw_folder / "manifest.json") or
            raw_manifest["seed"] != seed or feature_audit["state"] != "passed" or
            feature_audit["extraction_manifest_sha256"] != sha(raw_folder / "manifest.json")):
        raise ValueError("independently audited frozen extraction required")
    for name, expected in raw_manifest["outputs"].items():
        if sha(raw_folder / name) != expected:
            raise ValueError("extracted source output changed: " + name)
    old_folder = ISOLATED / f"score-seed{seed}"
    old_manifest, old_status = read(old_folder / "manifest.json"), read(old_folder / "status.json")
    old_audit = read(ISOLATED / f"audit-seed{seed}" / "audit.json")
    if (old_status["manifest_sha256"] != sha(old_folder / "manifest.json") or
            old_audit["state"] != "passed" or
            old_audit["scoring_manifest_sha256"] != sha(old_folder / "manifest.json") or
            old_manifest["checkpoint_sha256"] != raw_manifest["checkpoint_sha256"] or
            old_manifest["temperature_fit_role"] != "ID_temperature_1500" or
            old_manifest["threshold_fit_role"] != "ID_threshold_1500"):
        raise ValueError("previously frozen ID-only calibration differs")
    for name in ("id_test_scores.npz", "id_validation_scores.npz"):
        if old_manifest["outputs"][name] != sha(old_folder / name):
            raise ValueError("frozen ID score file changed")
    partition = arrays(ISOLATED / "partition.npz")
    threshold_rows = partition["threshold_rows"]
    if len(threshold_rows) != 1500:
        raise ValueError("frozen 1500-row threshold population differs")
    raw = arrays(raw_folder / "raw.npz")
    t1 = arrays(raw_folder / "T1_scores.npz")
    feature = arrays(raw_folder / "feature_scores.npz")
    id_test = arrays(old_folder / "id_test_scores.npz")
    id_validation = arrays(old_folder / "id_validation_scores.npz")
    if (raw["logits"].shape != (10000, 100) or
            set(t1) != {"msp", "energy", "maxlogit", "entropy"} or
            set(feature) != {"knn", "mahalanobis"}):
        raise ValueError("fixed OOD raw-score dimensions differ")
    manual_t1 = logit_scores(raw["logits"], 1.)
    for name, value in t1.items():
        if not np.allclose(value, manual_t1[name], atol=2e-5, rtol=2e-5):
            raise ValueError("T=1 logits score differs: " + name)
    temperature = old_manifest["temperature"]
    calibrated_logits = logit_scores(raw["logits"], temperature)
    scores = {"T1": {**t1, **feature},
              "calibrated": {**t1, **feature,
                             **{m: calibrated_logits[m] for m in ("msp", "energy", "entropy")}}}
    for method in ("maxlogit", "knn", "mahalanobis"):
        if not np.array_equal(scores["T1"][method], scores["calibrated"][method]):
            raise ValueError("ID temperature changed an invariant detector")
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    destination = RUN / f"score-seed{seed}"
    if destination.exists():
        raise FileExistsError("Fashion-MNIST scores already exist")
    destination.mkdir()
    snapshots = destination / "source_snapshot"
    snapshots.mkdir()
    for name in SOURCES:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(destination / "status.json", {"state": "scoring", "seed": seed,
                                        "started_at": datetime.now(timezone.utc).isoformat()})
    score_arrays = {f"{variant}_{method}": scores[variant][method]
                    for variant in VARIANTS for method in METHODS}
    save(destination / "scores.npz", **score_arrays)
    mask_arrays = {}
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for method in METHODS:
            key = f"{variant}_{method}"
            inside = id_test[key]
            outside = scores[variant][method]
            calibration = id_validation[key][threshold_rows]
            if (len(inside) != 10000 or len(outside) != 10000 or
                    len(calibration) != 1500 or not np.isfinite(outside).all()):
                raise ValueError("fixed ID/OOD denominator or score finite guard")
            frozen = old_manifest["datasets"]["cifar10"]["methods"][variant][method]["operating_grid"]
            for population in PRIOR_POPULATIONS[1:]:
                other = old_manifest["datasets"][population]["methods"][variant][method]["operating_grid"]
                if any(other[f"target{t}"]["threshold"] != frozen[f"target{t}"]["threshold"]
                       for t in TARGETS):
                    raise ValueError("prior OOD population changed a frozen ID threshold")
            metrics = detection_metrics(inside, outside)
            operating = {}
            for target in TARGETS:
                target_key = f"target{target}"
                previous = frozen[target_key]
                threshold = previous["threshold"]
                if id_threshold(calibration, target / 100) != threshold:
                    raise ValueError("saved ID threshold no longer matches original 1500 ID rows")
                cal_mask = np.asarray(calibration <= threshold, dtype=np.uint8)
                id_mask = np.asarray(inside <= threshold, dtype=np.uint8)
                ood_mask = np.asarray(outside <= threshold, dtype=np.uint8)
                if (int(cal_mask.sum()) != previous["calibration_accepted"] or
                        int(id_mask.sum()) != previous["id_test_accepted"] or
                        int(np.count_nonzero(calibration == threshold)) !=
                        previous["calibration_ties_at_threshold"]):
                    raise ValueError("frozen ID threshold mask/count changed")
                for suffix, mask in (("threshold_ID", cal_mask), ("test_ID", id_mask),
                                     ("OOD", ood_mask)):
                    mask_arrays[f"{key}_{target_key}_{suffix}"] = mask
                operating[target_key] = {field: previous[field] for field in
                    ("target", "threshold", "calibration_accepted", "calibration_count",
                     "calibration_acceptance", "id_test_accepted", "id_test_count",
                     "id_test_acceptance", "calibration_ties_at_threshold")}
                operating[target_key].update({"ood_accepted": int(ood_mask.sum()),
                                              "ood_count": len(outside),
                                              "ood_acceptance": float(ood_mask.mean())})
            current95 = operating["target95"]
            metrics.update({"validation_threshold": current95["threshold"],
                            "id_test_acceptance_at_validation_threshold":
                            current95["id_test_acceptance"],
                            "ood_acceptance_at_validation_threshold":
                            current95["ood_acceptance"],
                            "calibration_id_acceptance":
                            current95["calibration_acceptance"]})
            results[variant][method] = {"metrics": metrics, "operating_grid": operating}
    save(destination / "acceptance_masks.npz", **mask_arrays)
    outputs = {name: sha(destination / name) for name in ("scores.npz", "acceptance_masks.npz")}
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("scoring source changed during execution")
    manifest = {"state": "scored_pending_independent_audit", "seed": seed,
                "dataset": "fashionmnist_official_test", "n_ood": 10000,
                "n_id_test": 10000, "temperature": temperature,
                "checkpoint_sha256": raw_manifest["checkpoint_sha256"],
                "extraction_manifest_sha256": sha(raw_folder / "manifest.json"),
                "feature_audit_sha256": sha(RUN / f"feature-audit-seed{seed}" / "audit.json"),
                "prior_id_score_manifest_sha256": sha(old_folder / "manifest.json"),
                "partition_sha256": sha(ISOLATED / "partition.npz"),
                "methods": METHODS, "variants": VARIANTS, "targets": TARGETS,
                "results": results, "source_hashes": source_hashes, "outputs": outputs,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "boundary": "Frozen ID temperature/thresholds reused; no Fashion-MNIST-based fitting."}
    write(destination / "manifest.json", manifest)
    write(destination / "status.json", {"state": manifest["state"], "seed": seed,
                                        "manifest_sha256": sha(destination / "manifest.json")})
    print(json.dumps({"state": manifest["state"], "seed": seed,
                      "methods": len(METHODS), "variants": len(VARIANTS)}), flush=True)


if __name__ == "__main__":
    main()

