"""Independently regenerate all stratified draws and weighted OOD metrics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-bootstrap"
BASE = ROOT / "runs/cifar100-class-holdout"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")


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


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def draw(replicate: int, inside_groups: list[np.ndarray],
         outside_groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(2035 + replicate)
    result = []
    for groups, count in ((inside_groups, 8000), (outside_groups, 2000)):
        weights = np.zeros(count, dtype=np.uint16)
        for rows in groups:
            chosen = rows[generator.integers(0, len(rows), size=len(rows))]
            np.add.at(weights, chosen, 1)
        result.append(weights)
    return result[0], result[1]


class WeightedReference:
    """Independent unique-score grouping, rather than the producer ranking plan."""

    def __init__(self, inside: np.ndarray, outside: np.ndarray):
        self.inside = np.asarray(inside, dtype=np.float64)
        self.outside = np.asarray(outside, dtype=np.float64)
        _, inverse = np.unique(np.r_[inside, outside], return_inverse=True)
        self.id_groups = inverse[:len(inside)]
        self.ood_groups = inverse[len(inside):]
        self.group_count = int(inverse.max()) + 1
        self.id_order = np.argsort(self.inside, kind="stable")

    def evaluate(self, wi: np.ndarray, wo: np.ndarray) -> np.ndarray:
        n_id, n_ood = int(wi.sum()), int(wo.sum())
        identity = np.bincount(self.id_groups, weights=wi,
                               minlength=self.group_count)
        anomaly = np.bincount(self.ood_groups, weights=wo,
                              minlength=self.group_count)
        below = np.cumsum(identity) - identity
        auroc = float(np.dot(anomaly, below + identity / 2) / (n_id * n_ood))
        id_cumulative = np.cumsum(identity)
        ood_cumulative = np.cumsum(anomaly)
        ood_from_high = np.cumsum(anomaly[::-1])
        id_from_high = np.cumsum(identity[::-1])
        out_precision = np.divide(ood_from_high, ood_from_high + id_from_high,
                                  out=np.zeros_like(ood_from_high),
                                  where=(ood_from_high + id_from_high) > 0)
        in_precision = np.divide(id_cumulative, id_cumulative + ood_cumulative,
                                 out=np.zeros_like(id_cumulative),
                                 where=(id_cumulative + ood_cumulative) > 0)
        ap_out = float(np.dot(anomaly[::-1], out_precision) / n_ood)
        ap_in = float(np.dot(identity, in_precision) / n_id)
        ordered_counts = np.cumsum(wi[self.id_order])
        position = int(np.searchsorted(ordered_counts,
                                       math.ceil(.95 * n_id), side="left"))
        threshold = self.inside[self.id_order[position]]
        fpr95 = float(wo[self.outside <= threshold].sum() / n_ood)
        return np.array([auroc, ap_out, ap_in, fpr95], dtype=np.float64)


def main() -> None:
    destination = RUN / "raw-audit.json"
    require(not destination.exists(), "full raw bootstrap audit exists; preserve it")
    declaration = read(RUN / "declaration.json")
    launch = read(RUN / "launch.json")
    status = read(RUN / "status.json")
    require(declaration["state"] == "frozen_before_stratified_draws" and
            declaration["replicates"] == 1000 and
            declaration["rng_seed_base"] == 2035 and
            launch["state"] == "fixed_1000_draw_schedule_launched" and
            launch["declaration_sha256"] == sha(RUN / "declaration.json") and
            status["state"] == "raw_complete_pending_independent_audit" and
            status["completed_blocks"] == list(range(10)) and
            status["completed_replicates"] == 1000 and
            status["launch_sha256"] == sha(RUN / "launch.json"),
            "full fixed bootstrap schedule must complete first")
    for item in declaration["inputs_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "frozen bootstrap input changed: " + item["path"])
    for name, digest in launch["source_sha256"].items():
        require(sha(ROOT / name) ==
                sha(RUN / "source_snapshot" / Path(name).name) == digest,
                "executed bootstrap source changed: " + name)
    test = arrays(BASE / "test-population.npz")
    id_labels = test["id_original_fine_labels"]
    ood_labels = test["ood_original_fine_labels"]
    id_groups = [np.where(id_labels == name)[0] for name in sorted(set(id_labels))]
    ood_groups = [np.where(ood_labels == name)[0]
                  for name in sorted(set(ood_labels))]
    require(len(id_groups) == 80 and len(ood_groups) == 20 and
            all(len(group) == 100 for group in id_groups + ood_groups),
            "all official test fine-class strata required")
    reference = []
    id_masks = np.empty((3, 2, 6, 8000), dtype=np.uint8)
    ood_masks = np.empty((3, 2, 6, 2000), dtype=np.uint8)
    raw_scores = []
    for seed in range(3):
        folder = BASE / f"score-seed{seed}"
        id_scores = arrays(folder / "id_test_scores.npz")
        ood_scores = arrays(folder / "ood_test_scores.npz")
        masks = arrays(folder / "acceptance_masks.npz")
        seed_ref, seed_raw = [], []
        for vi, variant in enumerate(VARIANTS):
            method_ref, method_raw = [], []
            for mi, method in enumerate(METHODS):
                key = f"{variant}_{method}"
                inside, outside = id_scores[key], ood_scores[key]
                require(inside.shape == (8000,) and outside.shape == (2000,),
                        "full source score shapes")
                method_ref.append(WeightedReference(inside, outside))
                method_raw.append((inside, outside))
                id_masks[seed, vi, mi] = masks[f"{key}_target95_test_ID"]
                ood_masks[seed, vi, mi] = masks[f"{key}_target95_OOD"]
            seed_ref.append(method_ref)
            seed_raw.append(method_raw)
        reference.append(seed_ref)
        raw_scores.append(seed_raw)
    checks = {"regenerated_weight_vectors": 0,
              "within_class_weight_totals": 0,
              "independent_ranking_fields": 0,
              "independent_acceptance_counts": 0,
              "expanded_sklearn_ranking_fields": 0,
              "block_hashes": 0}
    block_hashes = {}
    for block in range(10):
        stem = f"block-{block:03d}"
        folder = RUN / "blocks"
        marker_path = folder / f"{stem}.json"
        weight_path = folder / f"{stem}-weights.npz"
        metric_path = folder / f"{stem}-metrics.npz"
        marker = read(marker_path)
        require(marker["state"] == "committed" and
                marker["start"] == block * 100 and
                marker["end_exclusive"] == (block + 1) * 100 and
                marker["launch_sha256"] == sha(RUN / "launch.json") and
                marker["weights_sha256"] == sha(weight_path) and
                marker["metrics_sha256"] == sha(metric_path),
                "immutable block source/outputs changed")
        for path in (marker_path, weight_path, metric_path):
            block_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            checks["block_hashes"] += 1
        weights = arrays(weight_path)
        metrics = arrays(metric_path)
        replicate_ids = np.arange(block * 100, (block + 1) * 100,
                                   dtype=np.int32)
        require(np.array_equal(weights["replicates"], replicate_ids) and
                np.array_equal(metrics["replicates"], replicate_ids) and
                weights["id_weights"].shape == (100, 8000) and
                weights["ood_weights"].shape == (100, 2000) and
                metrics["ranking"].shape == (100, 3, 2, 6, 4) and
                metrics["id_accepted_counts"].shape == (100, 3, 2, 6) and
                metrics["ood_accepted_counts"].shape == (100, 3, 2, 6),
                "complete block arrays missing")
        for local, replicate in enumerate(replicate_ids):
            wi, wo = draw(int(replicate), id_groups, ood_groups)
            require(np.array_equal(wi, weights["id_weights"][local]) and
                    np.array_equal(wo, weights["ood_weights"][local]),
                    "independently regenerated paired draw differs")
            checks["regenerated_weight_vectors"] += 2
            require(int(wi.sum()) == 8000 and int(wo.sum()) == 2000 and
                    all(int(wi[group].sum()) == 100 for group in id_groups) and
                    all(int(wo[group].sum()) == 100 for group in ood_groups),
                    "class-stratified fixed draw counts")
            checks["within_class_weight_totals"] += 100
            expected_id = np.tensordot(id_masks, wi.astype(np.int64), axes=([3], [0]))
            expected_ood = np.tensordot(ood_masks, wo.astype(np.int64), axes=([3], [0]))
            require(np.array_equal(metrics["id_accepted_counts"][local],
                                   expected_id) and
                    np.array_equal(metrics["ood_accepted_counts"][local],
                                   expected_ood),
                    "independent frozen-mask weighted acceptance")
            checks["independent_acceptance_counts"] += 72
            for seed in range(3):
                for vi in range(2):
                    for mi in range(6):
                        value = reference[seed][vi][mi].evaluate(wi, wo)
                        observed = metrics["ranking"][local, seed, vi, mi]
                        require(np.allclose(observed, value, rtol=0, atol=1e-12),
                                "full independent weighted ranking differs")
                        checks["independent_ranking_fields"] += 4
                        if replicate < 10:
                            inside, outside = raw_scores[seed][vi][mi]
                            x = np.repeat(inside, wi)
                            y = np.repeat(outside, wo)
                            labels = np.r_[np.zeros(len(x)), np.ones(len(y))]
                            threshold = np.partition(x, math.ceil(.95 * len(x)) - 1)[
                                math.ceil(.95 * len(x)) - 1]
                            expanded = np.array([
                                roc_auc_score(labels, np.r_[x, y]),
                                average_precision_score(labels, np.r_[x, y]),
                                average_precision_score(1 - labels,
                                                        -np.r_[x, y]),
                                np.mean(y <= threshold),
                            ])
                            require(np.allclose(observed, expanded,
                                                rtol=0, atol=1e-12),
                                    "expanded sklearn ranking reference differs")
                            checks["expanded_sklearn_ranking_fields"] += 4
        print(json.dumps({"audited_block": block,
                          "replicates": (block + 1) * 100}), flush=True)
    record = {
        "state": "passed", "checked": checks,
        "declaration_sha256": sha(RUN / "declaration.json"),
        "launch_sha256": sha(RUN / "launch.json"),
        "status_sha256": sha(RUN / "status.json"),
        "block_hashes": block_hashes,
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "All 1000 paired class-stratified draws and weighted metrics independently recomputed; first ten expanded with sklearn; no new model or method selection.",
    }
    destination.write_text(json.dumps(record, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checks}), flush=True)


if __name__ == "__main__":
    main()
