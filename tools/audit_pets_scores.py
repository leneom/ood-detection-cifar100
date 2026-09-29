"""Independent raw-score, sklearn metric, and frozen-threshold mask audit."""
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
RUN = ROOT / "runs" / "oxford-pets-extension"
ISOLATED = ROOT / "runs" / "validation-isolation"
sys.path.insert(0, str(ROOT / "tools"))
from audit_isolated_scoring import manual_scores, independent_metrics, close

METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
TARGETS = (80, 90, 95, 99)


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


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    extracted = RUN / f"extract-seed{seed}"
    scored = RUN / f"score-seed{seed}"
    extraction, result = read(extracted / "manifest.json"), read(scored / "manifest.json")
    status = read(scored / "status.json")
    require(status["state"] == result["state"] == "scored_pending_independent_audit" and
            status["manifest_sha256"] == sha(scored / "manifest.json") and
            result["extraction_manifest_sha256"] == sha(extracted / "manifest.json") and
            result["feature_audit_sha256"] == sha(RUN / f"feature-audit-seed{seed}" / "audit.json") and
            result["checkpoint_sha256"] == extraction["checkpoint_sha256"] and
            result["seed"] == seed, "frozen score source binding")
    for name, expected in result["source_hashes"].items():
        require(sha(scored / "source_snapshot" / name) == expected == sha(ROOT / name),
                "scoring source snapshot changed: " + name)
    for name, expected in result["outputs"].items():
        require(sha(scored / name) == expected, "scored output hash changed: " + name)
    old_folder = ISOLATED / f"score-seed{seed}"
    old = read(old_folder / "manifest.json")
    require(result["prior_id_score_manifest_sha256"] == sha(old_folder / "manifest.json") and
            result["temperature"] == old["temperature"] and
            result["partition_sha256"] == sha(ISOLATED / "partition.npz"), "ID fit differs")
    x = arrays(old_folder / "id_test_scores.npz")
    c = arrays(old_folder / "id_validation_scores.npz")
    rows = arrays(ISOLATED / "partition.npz")["threshold_rows"]
    raw = arrays(extracted / "raw.npz")
    control = arrays(extracted / "T1_scores.npz")
    feature = arrays(extracted / "feature_scores.npz")
    produced = arrays(scored / "scores.npz")
    masks = arrays(scored / "acceptance_masks.npz")
    require(set(produced) == {f"{v}_{m}" for v in VARIANTS for m in METHODS}, "12 scores")
    require(len(masks) == 12 * 4 * 3, "all 144 acceptance masks")
    manual_t1 = manual_scores(raw["logits"], 1.)
    manual_calibrated = manual_scores(raw["logits"], old["temperature"])
    checks = {"score_vectors": 0, "metric_fields": 0, "operating_fields": 0,
              "mask_vectors": 0, "exact_invariant_vectors": 0}
    for variant in VARIANTS:
        for method in METHODS:
            key = f"{variant}_{method}"
            actual = produced[key]
            if method in ("knn", "mahalanobis"):
                require(np.array_equal(actual, feature[method]), key + " ID-fit feature copy")
                checks["exact_invariant_vectors"] += 1
            elif variant == "T1" or method == "maxlogit":
                require(np.array_equal(actual, control[method]), key + " T1 control copy")
                require(np.allclose(actual, manual_t1[method], atol=2e-5, rtol=2e-5),
                        key + " independent T1 math")
                checks["exact_invariant_vectors"] += 1
            else:
                require(np.allclose(actual, manual_calibrated[method], rtol=0, atol=1e-11),
                        key + " independent calibrated math")
            checks["score_vectors"] += 1
            inside, calibration = x[key], c[key][rows]
            expected = independent_metrics(inside, actual, calibration)
            observed = result["results"][variant][method]
            for field, value in expected.items():
                if field in ("n_id", "n_ood"):
                    require(observed["metrics"][field] == value, key + "/" + field)
                else:
                    close(observed["metrics"][field], value, key + "/" + field)
                checks["metric_fields"] += 1
            for target in TARGETS:
                target_key = f"target{target}"
                threshold = sorted(calibration.tolist())[int(np.ceil(target/100*len(calibration))) - 1]
                frozen = old["datasets"]["cifar10"]["methods"][variant][method]["operating_grid"][target_key]
                row = observed["operating_grid"][target_key]
                require(threshold == frozen["threshold"] == row["threshold"],
                        key + "/" + target_key + " frozen ID threshold")
                for suffix, values in (("threshold_ID", calibration), ("test_ID", inside),
                                       ("OOD", actual)):
                    expected_mask = np.asarray(values <= threshold, dtype=np.uint8)
                    require(np.array_equal(masks[f"{key}_{target_key}_{suffix}"], expected_mask),
                            key + "/" + target_key + "/" + suffix)
                    checks["mask_vectors"] += 1
                for prefix, values in (("calibration", calibration), ("id_test", inside),
                                       ("ood", actual)):
                    accepted = int(np.count_nonzero(values <= threshold))
                    require(row[prefix + "_accepted"] == accepted and
                            row[prefix + "_count"] == len(values), "acceptance counts")
                    close(row[prefix + "_acceptance"], accepted/len(values), "acceptance rate")
                    checks["operating_fields"] += 3
                require(row["calibration_ties_at_threshold"] ==
                        int(np.count_nonzero(calibration == threshold)), "threshold ties")
    for method in ("maxlogit", "knn", "mahalanobis"):
        require(np.array_equal(produced[f"T1_{method}"], produced[f"calibrated_{method}"]),
                "temperature invariant score")
    destination = RUN / f"score-audit-seed{seed}"
    require(not destination.exists(), "score audit already exists")
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    record = {"state": "passed", "seed": seed, "checked": checks,
              "scoring_manifest_sha256": sha(scored / "manifest.json"),
              "extraction_manifest_sha256": sha(extracted / "manifest.json"),
              "source_sha256": sha(__file__),
              "math_audit_source_sha256": sha(ROOT / "tools/audit_isolated_scoring.py"),
              "audited_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "Fixed six methods and ID-only thresholds on a previously unscored OOD population; no post-test tuning."}
    (destination / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed, "checked": checks}), flush=True)


if __name__ == "__main__":
    main()

