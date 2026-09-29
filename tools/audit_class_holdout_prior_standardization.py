"""Independent sklearn weighted-AP audit of all frozen score populations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-prior-standardization"
HOLD = ROOT / "runs/cifar100-class-holdout"
C10 = ROOT / "runs/class-holdout-cifar10-extension"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
POPS = ("heldout_cifar100", "cifar10")
ORIENTATIONS = ("aupr_out", "aupr_in")


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
        return {name: source[name] for name in source.files}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    output = RUN / "raw-audit.json"
    require(not output.exists(), "write-once AP raw audit exists")
    declaration = read(RUN / "declaration.json")
    manifest = read(RUN / "manifest.json")
    require(declaration["state"] == "frozen_before_20pct_standardized_ap_calculation" and
            declaration["target_ood_prior"] == .2 and
            declaration["n_id"] == 8000 and
            manifest["state"] == "raw_complete_pending_independent_audit" and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            manifest["raw_values_sha256"] == sha(RUN / "raw-values.npz") and
            manifest["producer_source_sha256"] ==
            sha(ROOT / "tools/run_class_holdout_prior_standardization.py") and
            manifest["shape"] == [3, 2, 6, 2, 2] and
            manifest["target_ood_prior"] == .2 and
            manifest["id_row_weight"] == .8 / 8000 and
            manifest["ood_row_weights"] == {"heldout_cifar100": .2 / 2000,
                                            "cifar10": .2 / 10000},
            "frozen complete weighted AP declaration/output")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, "declared source changed: " + name)
    raw = arrays(RUN / "raw-values.npz")
    require(set(raw) == {"standardized", "original_natural", "seed",
                         "target_ood_prior"} and
            raw["standardized"].shape ==
            raw["original_natural"].shape == (3, 2, 6, 2, 2) and
            np.array_equal(raw["seed"], np.arange(3)) and
            float(raw["target_ood_prior"]) == .2 and
            np.isfinite(raw["standardized"]).all() and
            np.isfinite(raw["original_natural"]).all(),
            "all 144 original and standardized raw values")
    checks = {"sklearn_weighted_ap": 0, "natural_ap": 0,
              "heldout_identity": 0, "weight_mass_checks": 0}
    for seed in range(3):
        inside = arrays(HOLD / f"score-seed{seed}" / "id_test_scores.npz")
        oods = (arrays(HOLD / f"score-seed{seed}" / "ood_test_scores.npz"),
                arrays(C10 / f"score-seed{seed}" / "cifar10_scores.npz"))
        manifests = (read(HOLD / f"score-seed{seed}" / "manifest.json"),
                     read(C10 / f"score-seed{seed}" / "manifest.json"))
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                key = f"{variant}_{method}"
                for pi, population in enumerate(POPS):
                    n_ood = declaration["n_ood"][population]
                    x, y = inside[key], oods[pi][key]
                    require(x.shape == (8000,) and y.shape == (n_ood,) and
                            np.isfinite(x).all() and np.isfinite(y).all(),
                            "complete original score vectors")
                    labels = np.r_[np.zeros(8000, dtype=np.int8),
                                   np.ones(n_ood, dtype=np.int8)]
                    scores = np.r_[x, y]
                    weights = np.r_[np.full(8000, .8 / 8000),
                                    np.full(n_ood, .2 / n_ood)]
                    require(abs(weights[labels == 0].sum() - .8) < 1e-13 and
                            abs(weights[labels == 1].sum() - .2) < 1e-13,
                            "exact class-prior mass")
                    checks["weight_mass_checks"] += 1
                    for oi, orientation in enumerate(ORIENTATIONS):
                        target = labels if oi == 0 else 1 - labels
                        directed = scores if oi == 0 else -scores
                        reference = average_precision_score(target, directed,
                                                            sample_weight=weights)
                        natural = average_precision_score(target, directed)
                        reported_natural = (manifests[pi]["method_results"][variant]
                                            [method]["metrics"][orientation])
                        observed = raw["standardized"][seed, vi, mi, pi, oi]
                        recorded_natural = raw["original_natural"][seed, vi, mi, pi, oi]
                        require(abs(observed - reference) < 1e-12,
                                "independent weighted sklearn AP")
                        require(abs(recorded_natural - natural) < 1e-12 and
                                abs(recorded_natural - reported_natural) < 1e-12,
                                "original natural-prevalence AP")
                        checks["sklearn_weighted_ap"] += 1
                        checks["natural_ap"] += 1
                        if pi == 0:
                            require(abs(observed - recorded_natural) < 1e-12,
                                    "20%-prior heldout AP identity")
                            checks["heldout_identity"] += 1
    record = {
        "state": "passed", "checked": checks,
        "declaration_sha256": sha(RUN / "declaration.json"),
        "run_manifest_sha256": sha(RUN / "manifest.json"),
        "raw_values_sha256": sha(RUN / "raw-values.npz"),
        "audit_source_sha256": sha(Path(__file__)),
        "boundary": "All 144 AP values independently checked with weighted sklearn, original unweighted AP and exact heldout 20%-prior identity.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checks}))


if __name__ == "__main__":
    main()
