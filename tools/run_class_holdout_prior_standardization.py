"""Exact weighted grouped-threshold AP under fixed 20% OOD display prior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


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


def grouped_weighted_ap(id_scores: np.ndarray, ood_scores: np.ndarray,
                        ood_prior: float, ood_positive: bool) -> float:
    """Noninterpolated AP, exact score tie groups and constant class weights."""
    inside = np.asarray(id_scores, dtype=np.float64)
    outside = np.asarray(ood_scores, dtype=np.float64)
    require(inside.shape == (8000,) and outside.ndim == 1 and
            len(outside) in (2000, 10000) and
            np.isfinite(inside).all() and np.isfinite(outside).all() and
            0 < ood_prior < 1, "complete finite score populations and target prior")
    scores = np.r_[inside, outside]
    labels = np.r_[np.zeros(len(inside), dtype=np.uint8),
                   np.ones(len(outside), dtype=np.uint8)]
    weights = np.r_[np.full(len(inside), (1 - ood_prior) / len(inside)),
                    np.full(len(outside), ood_prior / len(outside))]
    if not ood_positive:
        scores = -scores
        labels = 1 - labels
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_weights = weights[order]
    ranked_positive = ranked_weights * labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores) - 1]
    cumulative_positive = np.cumsum(ranked_positive)[ends]
    cumulative_total = np.cumsum(ranked_weights)[ends]
    recall = cumulative_positive / ranked_positive.sum()
    precision = cumulative_positive / cumulative_total
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def main() -> None:
    raw_path = RUN / "raw-values.npz"
    manifest_path = RUN / "manifest.json"
    require(not raw_path.exists() and not manifest_path.exists(),
            "write-once standardized AP outputs already exist")
    declaration = read(RUN / "declaration.json")
    require(declaration["state"] == "frozen_before_20pct_standardized_ap_calculation" and
            declaration["seeds"] == [0, 1, 2] and
            declaration["variants"] == list(VARIANTS) and
            declaration["methods"] == list(METHODS) and
            declaration["populations"] == list(POPS) and
            declaration["orientations"] == list(ORIENTATIONS) and
            declaration["target_ood_prior"] == .2 and
            declaration["n_id"] == 8000 and
            declaration["n_ood"] == {"heldout_cifar100": 2000, "cifar10": 10000},
            "complete frozen 20% display-prior declaration")
    for name, expected in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == expected,
                "declared original score/completion changed: " + name)
    standardized = np.empty((3, 2, 6, 2, 2), dtype=np.float64)
    original = np.empty((3, 2, 6, 2, 2), dtype=np.float64)
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
                    x, y = inside[key], oods[pi][key]
                    require(x.shape == (8000,) and y.shape ==
                            (declaration["n_ood"][population],),
                            "full frozen score vector shape")
                    for oi, orientation in enumerate(ORIENTATIONS):
                        fixed = grouped_weighted_ap(x, y, .2, oi == 0)
                        natural = (manifests[pi]["method_results"][variant][method]
                                   ["metrics"][orientation])
                        standardized[seed, vi, mi, pi, oi] = fixed
                        original[seed, vi, mi, pi, oi] = natural
                        if pi == 0:
                            require(abs(fixed - natural) < 1e-12,
                                    "natural 20% heldout AP identity")
    with raw_path.open("wb") as stream:
        np.savez_compressed(stream, standardized=standardized,
                            original_natural=original,
                            seed=np.arange(3, dtype=np.int8),
                            target_ood_prior=np.array(.2))
    require(all(sha(ROOT / name) == expected for name, expected in
                declaration["inputs_sha256"].items()),
            "frozen source changed during standardization")
    manifest = {
        "state": "raw_complete_pending_independent_audit",
        "declaration_sha256": sha(RUN / "declaration.json"),
        "raw_values_sha256": sha(raw_path),
        "producer_source_sha256": sha(Path(__file__)),
        "shape": [3, 2, 6, 2, 2],
        "target_ood_prior": .2,
        "id_row_weight": .8 / 8000,
        "ood_row_weights": {"heldout_cifar100": .2 / 2000,
                            "cifar10": .2 / 10000},
        "boundary": "Exact grouped weighted AP, both orientations, all methods/models/populations; no resampling or fitting.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"state": manifest["state"], "values": standardized.size}))


if __name__ == "__main__":
    main()
