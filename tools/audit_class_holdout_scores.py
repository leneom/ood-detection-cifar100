"""Independent NumPy/sklearn audit of fixed 80-class scores and metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn import __version__ as sklearn_version

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
sys.path.insert(0, str(ROOT / "tools"))
from audit_isolated_scoring import (independent_temperature,
                                    independent_metrics, manual_scores)


METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
TARGETS = (80, 90, 95, 99)
POPULATIONS = ("temperature_id", "threshold_id", "id_test", "ood_test")


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


def close(actual: float, expected: float, message: str,
          atol: float = 1e-10) -> None:
    require(np.isclose(actual, expected, rtol=0, atol=atol),
            message + f": {actual} versus {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    raw_dir = RUN / f"extract-seed{seed}"
    score_dir = RUN / f"score-seed{seed}"
    audit_dir = RUN / f"score-audit-seed{seed}"
    require(not audit_dir.exists(), "score audit exists; preserve it")
    raw, scored = read(raw_dir / "manifest.json"), read(score_dir / "manifest.json")
    raw_status, score_status = read(raw_dir / "status.json"), read(score_dir / "status.json")
    feature_audit = read(RUN / f"feature-audit-seed{seed}" / "audit.json")
    require(raw_status["state"] == "extracted_pending_CPU_calibration_and_audit" and
            raw_status["manifest_sha256"] == sha(raw_dir / "manifest.json") and
            score_status["state"] == "scored_pending_independent_audit" and
            score_status["manifest_sha256"] == sha(score_dir / "manifest.json") and
            scored["state"] == "scored_pending_independent_audit" and
            raw["seed"] == scored["seed"] == seed and
            scored["extraction_manifest_sha256"] ==
            feature_audit["extraction_manifest_sha256"] ==
            sha(raw_dir / "manifest.json") and
            feature_audit["state"] == "passed" and
            scored["feature_audit_sha256"] ==
            sha(RUN / f"feature-audit-seed{seed}" / "audit.json") and
            scored["checkpoint_sha256"] == raw["checkpoint_sha256"] ==
            sha(RUN / f"seed{seed}" / "final-best.pt") and
            scored["partition_sha256"] == raw["partition_sha256"] ==
            sha(RUN / "partition.npz") and
            scored["test_population_sha256"] == raw["test_population_sha256"] ==
            sha(RUN / "test-population.npz") and
            scored["temperature_fit_role"] == "ID_temperature_1200" and
            scored["threshold_fit_role"] == "ID_threshold_1200" and
            scored["methods"] == list(METHODS) and
            scored["targets"] == [.80, .90, .95, .99] and
            scored["n_id"] == 8000 and scored["n_ood"] == 2000 and
            scored["ood_prevalence"] == .2,
            "producer, full-population or independent feature-audit binding differs")
    for manifest, folder in ((raw, raw_dir), (scored, score_dir)):
        for name, expected in manifest["outputs"].items():
            require(sha(folder / name) == expected,
                    "producer output differs: " + name)
        for name, expected in manifest["source_hashes"].items():
            require(sha(folder / "source_snapshot" / name) == expected ==
                    sha(ROOT / name), "executed producer source differs: " + name)
    partition = arrays(RUN / "partition.npz")
    test = arrays(RUN / "test-population.npz")
    temperature_raw = arrays(raw_dir / "temperature_id_raw.npz")
    threshold_raw = arrays(raw_dir / "threshold_id_raw.npz")
    id_raw = arrays(raw_dir / "id_test_raw.npz")
    ood_raw = arrays(raw_dir / "ood_test_raw.npz")
    require(np.array_equal(temperature_raw["original_indices"],
                           partition["temperature_indices"]) and
            np.array_equal(threshold_raw["original_indices"],
                           partition["threshold_indices"]) and
            np.array_equal(id_raw["original_indices"], test["id_test_indices"]) and
            np.array_equal(ood_raw["original_indices"], test["ood_test_indices"]) and
            np.array_equal(id_raw["id_class_targets"], test["id_class_targets"]) and
            not np.intersect1d(temperature_raw["original_indices"],
                               threshold_raw["original_indices"]).size and
            len(temperature_raw["logits"]) == len(threshold_raw["logits"]) == 1200 and
            len(id_raw["logits"]) == 8000 and len(ood_raw["logits"]) == 2000,
            "ID-only roles or original test rows differ")
    fit = read(score_dir / "temperature_fit.json")
    require(fit["population"] == "1,200 ID temperature-role rows only" and
            fit["fitting_role"] == "temperature" and
            fit["fitting_count"] == 1200 and
            fit["temperature"] == scored["temperature"],
            "ID-only temperature fit metadata differs")
    ref_temperature = independent_temperature(
        temperature_raw["logits"], temperature_raw["id_class_targets"])
    if fit["fallback_reason"] is None:
        close(scored["temperature"], ref_temperature,
              "independent analytic-root temperature", atol=2e-5)
    else:
        require(scored["temperature"] == 1., "failed-fit fallback did not use T=1")
    counts = {"raw_logit_score_vectors": 0, "exact_control_vectors": 0,
              "metric_fields": 0, "operating_fields": 0, "mask_vectors": 0}
    saved_scores = {}
    for name in POPULATIONS:
        raw_population = arrays(raw_dir / f"{name}_raw.npz")
        saved = arrays(score_dir / f"{name}_scores.npz")
        t1 = arrays(raw_dir / f"{name}_T1_scores.npz")
        feature = arrays(raw_dir / f"{name}_feature_scores.npz")
        require(set(saved) == {f"{variant}_{method}"
                               for variant in ("T1", "calibrated")
                               for method in METHODS},
                "twelve fixed score vectors missing: " + name)
        for method, source in list(t1.items()) + list(feature.items()):
            require(np.array_equal(saved[f"T1_{method}"], source),
                    "T1 score changed from extraction: " + name + "/" + method)
            counts["exact_control_vectors"] += 1
        for method in ("maxlogit", "knn", "mahalanobis"):
            require(np.array_equal(saved[f"calibrated_{method}"],
                                   saved[f"T1_{method}"]),
                    "temperature changed invariant score: " + name + "/" + method)
            counts["exact_control_vectors"] += 1
        for variant, value in (("T1", 1.), ("calibrated", scored["temperature"])):
            manual = manual_scores(raw_population["logits"], value)
            for method in ("msp", "energy", "maxlogit", "entropy"):
                if variant == "calibrated" and method == "maxlogit":
                    continue
                tolerance = 2e-5 if variant == "T1" else 2e-10
                require(np.allclose(saved[f"{variant}_{method}"], manual[method],
                                    rtol=0, atol=tolerance),
                        "independent raw-logit score differs: " +
                        name + "/" + variant + "/" + method)
                counts["raw_logit_score_vectors"] += 1
        saved_scores[name] = saved
    correctness = arrays(score_dir / "id_test_correctness.npz")
    ref_correct = np.asarray(id_raw["logits"].argmax(axis=1) ==
                             id_raw["id_class_targets"], dtype=np.uint8)
    require(np.array_equal(correctness["correct"], ref_correct) and
            np.array_equal(correctness["original_indices"],
                           test["id_test_indices"]) and
            np.array_equal(correctness["original_fine_labels"],
                           test["id_original_fine_labels"]) and
            np.array_equal(correctness["predictions"],
                           id_raw["logits"].argmax(axis=1)),
            "ID-test correctness or label alignment differs")
    close(scored["id_test_accuracy"], float(ref_correct.mean()),
          "8k ID-test accuracy")
    masks = arrays(score_dir / "acceptance_masks.npz")
    require(len(masks) == 2 * 6 * 4 * 3, "144 fixed acceptance masks required")
    for variant in ("T1", "calibrated"):
        for method in METHODS:
            key = f"{variant}_{method}"
            inside = saved_scores["id_test"][key]
            outside = saved_scores["ood_test"][key]
            threshold_role = saved_scores["threshold_id"][key]
            observed = scored["method_results"][variant][method]
            reference = independent_metrics(inside, outside, threshold_role)
            for field, value in reference.items():
                close(observed["metrics"][field], value,
                      key + "/" + field)
                counts["metric_fields"] += 1
            for target in TARGETS:
                label = f"target{target}"
                cutoff = sorted(threshold_role.tolist())[
                    int(np.ceil(target / 100 * len(threshold_role))) - 1]
                row = observed["operating_grid"][label]
                close(row["threshold"], cutoff, key + "/" + label + "/threshold")
                require(row["target"] == target / 100 and
                        row["calibration_ties_at_threshold"] ==
                        int(np.count_nonzero(threshold_role == cutoff)),
                        "threshold target or ties differ: " + key + "/" + label)
                for suffix, values, prefix in (
                    ("threshold_ID", threshold_role, "calibration"),
                    ("test_ID", inside, "id_test"),
                    ("OOD", outside, "ood"),
                ):
                    expected_mask = np.asarray(values <= cutoff, dtype=np.uint8)
                    require(np.array_equal(masks[f"{key}_{label}_{suffix}"],
                                           expected_mask),
                            "frozen-threshold acceptance mask differs: " +
                            key + "/" + label + "/" + suffix)
                    counts["mask_vectors"] += 1
                    accepted = int(expected_mask.sum())
                    require(row[prefix + "_accepted"] == accepted and
                            row[prefix + "_count"] == len(values),
                            "operating count differs: " + key + "/" + label)
                    close(row[prefix + "_acceptance"], accepted / len(values),
                          "operating acceptance: " + key + "/" + label)
                    counts["operating_fields"] += 3
    audit_dir.mkdir()
    shutil.copyfile(__file__, audit_dir / "executed_audit_source.py")
    result = {
        "state": "passed", "seed": seed, "checked": counts,
        "independent_temperature": ref_temperature,
        "extraction_manifest_sha256": sha(raw_dir / "manifest.json"),
        "scoring_manifest_sha256": sha(score_dir / "manifest.json"),
        "feature_audit_sha256": sha(RUN / f"feature-audit-seed{seed}" / "audit.json"),
        "sklearn_version": sklearn_version,
        "math_reference_sha256": sha(ROOT / "tools/audit_isolated_scoring.py"),
        "source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Independent full score, sklearn metric and fixed ID-threshold audit; multiseed summary remains separate.",
    }
    (audit_dir / "audit.json").write_text(json.dumps(result, indent=2,
                                                    allow_nan=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "checked": counts}), flush=True)


if __name__ == "__main__":
    main()
