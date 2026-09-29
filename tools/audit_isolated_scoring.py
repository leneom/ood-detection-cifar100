"""Independent NumPy/sklearn audit of the isolated raw-score and metric stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import brentq
from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "validation-isolation"
POPULATIONS = ("cifar10", "mnist", "svhn", "textures", "noise")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def close(actual, expected, label, atol=1e-11):
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"{label}: {actual} differs from independent {expected}")


def manual_scores(logits, temperature):
    z = np.asarray(logits, dtype=np.float64)
    scaled = z / temperature
    high = scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled - high)
    logsum = high[:, 0] + np.log(exp.sum(axis=1))
    p = exp / exp.sum(axis=1, keepdims=True)
    entropy = -(p * (scaled - logsum[:, None])).sum(axis=1)
    return {"msp": 1 - p.max(axis=1), "energy": -temperature * logsum,
            "maxlogit": -z.max(axis=1), "entropy": entropy}


def independent_temperature(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    def derivative(log_t):
        beta = np.exp(-log_t)
        scaled = z * beta
        shifted = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        p = shifted / shifted.sum(axis=1, keepdims=True)
        return float(beta * np.mean(z[np.arange(len(y)), y] - (p * z).sum(axis=1)))

    lo, hi = np.log(.05), np.log(20.)
    dlo, dhi = derivative(lo), derivative(hi)
    if dlo >= 0:
        return .05
    if dhi <= 0:
        return 20.
    return float(np.exp(brentq(derivative, lo, hi, xtol=1e-11)))


def independent_metrics(inside, outside, threshold_role):
    x, y, c = map(np.asarray, (inside, outside, threshold_role))
    labels = np.r_[np.zeros(len(x), dtype=np.int8), np.ones(len(y), dtype=np.int8)]
    scores = np.r_[x, y]
    benchmark_threshold = sorted(x.tolist())[int(np.ceil(.95 * len(x))) - 1]
    operational_threshold = sorted(c.tolist())[int(np.ceil(.95 * len(c))) - 1]
    return {"auroc": roc_auc_score(labels, scores),
            "aupr_out": average_precision_score(labels, scores),
            "aupr_in": average_precision_score(1 - labels, -scores),
            "fpr95_id": float(np.mean(y <= benchmark_threshold)),
            "id_tpr_at_fpr95": float(np.mean(x <= benchmark_threshold)),
            "benchmark_threshold": benchmark_threshold,
            "validation_threshold": operational_threshold,
            "id_test_acceptance_at_validation_threshold": float(np.mean(x <= operational_threshold)),
            "ood_acceptance_at_validation_threshold": float(np.mean(y <= operational_threshold)),
            "calibration_id_acceptance": float(np.mean(c <= operational_threshold)),
            "n_id": len(x), "n_ood": len(y),
            "ood_prevalence": len(y) / (len(x) + len(y))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = p.parse_args().seed
    raw_dir = PHASE / f"extract-seed{seed}"
    score_dir = PHASE / f"score-seed{seed}"
    raw, scored = read(raw_dir / "manifest.json"), read(score_dir / "manifest.json")
    if (read(score_dir / "status.json")["manifest_sha256"] != digest(score_dir / "manifest.json") or
            scored["extraction_manifest_sha256"] != digest(raw_dir / "manifest.json") or
            raw["checkpoint_sha256"] != scored["checkpoint_sha256"] or
            scored["partition_sha256"] != digest(PHASE / "partition.npz")):
        raise ValueError("producer/checkpoint/partition binding differs")
    for manifest, directory in ((raw, raw_dir), (scored, score_dir)):
        for name, expected in manifest["outputs"].items():
            if digest(directory / name) != expected:
                raise ValueError("producer output changed: " + name)
        for name, expected in manifest["source_hashes"].items():
            if digest(directory / "source_snapshot" / name) != expected:
                raise ValueError("producer executed source snapshot changed: " + name)
    audit_dir = PHASE / f"audit-seed{seed}"
    if audit_dir.exists():
        raise FileExistsError("audit output exists; preserve and inspect before another attempt")
    partition = arrays(PHASE / "partition.npz")
    fit_rows, threshold_rows = partition["temperature_rows"], partition["threshold_rows"]
    val_raw = arrays(raw_dir / "id_validation_raw.npz")
    fit = read(score_dir / "temperature_fit.json")
    if len(fit_rows) != 1500 or len(threshold_rows) != 1500 or np.intersect1d(fit_rows, threshold_rows).size:
        raise ValueError("validation role count differs")
    if not np.array_equal(val_raw["labels"], partition["validation_labels"]):
        raise ValueError("temperature labels differ from frozen partition")
    ref_t = independent_temperature(val_raw["logits"][fit_rows], val_raw["labels"][fit_rows])
    if fit["fallback_reason"] is None:
        close(scored["temperature"], ref_t, "temperature analytic-root fit", atol=2e-5)
    elif scored["temperature"] != 1.:
        raise ValueError("declared failed-fit fallback did not use T=1")
    checked = {"temperature_independent_root": ref_t, "score_vectors": 0,
               "exact_control_vectors": 0, "metric_fields": 0,
               "mask_vectors": 0, "operating_fields": 0}

    def check_exact_controls(population, saved):
        raw_t1 = arrays(raw_dir / f"{population}_T1_scores.npz")
        raw_feature = arrays(raw_dir / f"{population}_feature_scores.npz")
        for method, source in list(raw_t1.items()) + list(raw_feature.items()):
            if not np.array_equal(saved[f"T1_{method}"], source):
                raise ValueError(f"T=1 control differs from extraction: {population}/{method}")
            checked["exact_control_vectors"] += 1
        for method in ("maxlogit", "knn", "mahalanobis"):
            if not np.array_equal(saved[f"calibrated_{method}"], saved[f"T1_{method}"]):
                raise ValueError(f"temperature-invariant control differs: {population}/{method}")
            checked["exact_control_vectors"] += 1
    test_raw = arrays(raw_dir / "id_test_raw.npz")
    val_saved = arrays(score_dir / "id_validation_scores.npz")
    id_saved = arrays(score_dir / "id_test_scores.npz")
    for population, logits, saved in (("id_validation", val_raw["logits"], val_saved),
                                      ("id_test", test_raw["logits"], id_saved)):
        check_exact_controls(population, saved)
        for variant, t in (("T1", 1.), ("calibrated", scored["temperature"])):
            manual = manual_scores(logits, t)
            for method in ("msp", "energy", "maxlogit", "entropy"):
                if variant == "calibrated" and method == "maxlogit":
                    continue
                tolerance = 2e-5 if variant == "T1" else 2e-10
                if not np.allclose(saved[f"{variant}_{method}"], manual[method], atol=tolerance, rtol=0):
                    raise ValueError(f"{population} {variant} {method} raw score differs")
                checked["score_vectors"] += 1
    for population in POPULATIONS:
        outside_raw = arrays(raw_dir / f"{population}_raw.npz")
        outside = arrays(score_dir / f"{population}_scores.npz")
        check_exact_controls(population, outside)
        masks = arrays(score_dir / f"{population}_acceptance_masks.npz")
        for variant, t in (("T1", 1.), ("calibrated", scored["temperature"])):
            manual = manual_scores(outside_raw["logits"], t)
            for method in ("msp", "energy", "maxlogit", "entropy"):
                if variant == "calibrated" and method == "maxlogit":
                    continue
                tolerance = 2e-5 if variant == "T1" else 2e-10
                if not np.allclose(outside[f"{variant}_{method}"], manual[method], atol=tolerance, rtol=0):
                    raise ValueError(f"{population} {variant} {method} raw score differs")
                checked["score_vectors"] += 1
            for method in METHODS:
                key = f"{variant}_{method}"
                x, y, c = id_saved[key], outside[key], val_saved[key][threshold_rows]
                if not all(np.isfinite(v).all() and v.ndim == 1 for v in (x, y, c)):
                    raise ValueError("nonfinite or nonvector score")
                if method in ("maxlogit", "knn", "mahalanobis") and not np.array_equal(
                        outside[f"T1_{method}"], outside[f"calibrated_{method}"]):
                    raise ValueError("temperature-invariant control changed")
                observed = scored["datasets"][population]["methods"][variant][method]
                expected = independent_metrics(x, y, c)
                for field, value in expected.items():
                    close(observed["metrics"][field], value, f"{population}/{key}/{field}")
                    checked["metric_fields"] += 1
                for target in TARGETS:
                    label = f"target{target}"
                    threshold = sorted(c.tolist())[int(np.ceil(target / 100 * len(c))) - 1]
                    row = observed["operating_grid"][label]
                    close(row["threshold"], threshold, f"{population}/{key}/{label}/threshold")
                    for suffix, values in (("threshold_ID", c), ("test_ID", x), ("OOD", y)):
                        actual = masks[f"{key}_{label}_{suffix}"]
                        if not np.array_equal(actual, np.asarray(values <= threshold, dtype=np.uint8)):
                            raise ValueError(f"acceptance mask differs: {population}/{key}/{label}/{suffix}")
                        checked["mask_vectors"] += 1
                    if row["calibration_ties_at_threshold"] != int(np.count_nonzero(c == threshold)):
                        raise ValueError("threshold tie count differs")
                    for field, values in (("calibration", c), ("id_test", x), ("ood", y)):
                        accepted = int(np.count_nonzero(values <= threshold))
                        if row[field + "_accepted"] != accepted or row[field + "_count"] != len(values):
                            raise ValueError("operating count differs")
                        close(row[field + "_acceptance"], accepted / len(values), "operating rate")
                        checked["operating_fields"] += 3
    audit_dir.mkdir()
    shutil.copyfile(__file__, audit_dir / "executed_audit_source.py")
    result = {"state": "passed", "seed": seed, "checked": checked,
              "source_sha256": digest(__file__), "scoring_manifest_sha256": digest(score_dir / "manifest.json"),
              "extraction_manifest_sha256": digest(raw_dir / "manifest.json"),
              "sklearn_version": sklearn_version, "audited_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "Raw score and operating-metric audit; full training-feature baseline refit and displayed summary audits remain separate."}
    (audit_dir / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"seed": seed, "state": "passed", "checked": checked}), flush=True)


if __name__ == "__main__":
    main()
