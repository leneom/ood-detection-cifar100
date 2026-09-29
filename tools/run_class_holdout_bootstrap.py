"""Fixed 1,000-draw, paired within-class bootstrap of frozen OOD scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import time

import numpy as np

from bootstrap_core import RankingPlan


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-bootstrap"
BASE = ROOT / "runs/cifar100-class-holdout"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")
SOURCE = ("tools/run_class_holdout_bootstrap.py", "tools/bootstrap_core.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def saved(path: Path, **data: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def groups(labels: np.ndarray, count: int) -> list[np.ndarray]:
    names = np.unique(labels)
    require(len(names) == count, "fixed fine-class census")
    result = [np.flatnonzero(labels == name) for name in names]
    require(all(len(group) == 100 for group in result) and
            np.array_equal(np.sort(np.concatenate(result)), np.arange(len(labels))),
            "exact 100-image official test strata")
    return result


def draw_counts(replicate: int, inside_groups: list[np.ndarray],
                outside_groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2035 + replicate)
    weights = []
    for indexed, length in ((inside_groups, 8000), (outside_groups, 2000)):
        counts = np.zeros(length, dtype=np.uint16)
        for rows in indexed:
            counts[rows] = np.bincount(rng.integers(0, 100, size=100),
                                       minlength=100).astype(np.uint16)
        require(int(counts.sum()) == length and
                all(int(counts[rows].sum()) == 100 for rows in indexed),
                "within-class draw counts differ")
        weights.append(counts)
    return weights[0], weights[1]


def inputs() -> tuple[list[np.ndarray], list[np.ndarray], list, np.ndarray, np.ndarray]:
    test = arrays(BASE / "test-population.npz")
    id_groups = groups(test["id_original_fine_labels"], 80)
    ood_groups = groups(test["ood_original_fine_labels"], 20)
    plans = []
    id_masks = np.empty((3, 2, 6, 8000), dtype=np.uint8)
    ood_masks = np.empty((3, 2, 6, 2000), dtype=np.uint8)
    for seed in range(3):
        folder = BASE / f"score-seed{seed}"
        inside = arrays(folder / "id_test_scores.npz")
        outside = arrays(folder / "ood_test_scores.npz")
        masks = arrays(folder / "acceptance_masks.npz")
        audit = read(BASE / f"score-audit-seed{seed}" / "audit.json")
        require(audit["state"] == "passed" and audit["seed"] == seed and
                audit["scoring_manifest_sha256"] == sha(folder / "manifest.json"),
                "complete frozen seed score audit required")
        seed_plans = []
        for vi, variant in enumerate(VARIANTS):
            method_plans = []
            for mi, method in enumerate(METHODS):
                key = f"{variant}_{method}"
                require(inside[key].shape == (8000,) and
                        outside[key].shape == (2000,) and
                        np.isfinite(inside[key]).all() and
                        np.isfinite(outside[key]).all(),
                        "finite fixed 8000/2000 scores required")
                method_plans.append(RankingPlan(inside[key], outside[key]))
                for source, target, suffix in (
                    (id_masks, 8000, "test_ID"),
                    (ood_masks, 2000, "OOD"),
                ):
                    values = masks[f"{key}_target95_{suffix}"]
                    require(values.shape == (target,) and
                            np.isin(values, (0, 1)).all(),
                            "frozen 95% ID-only acceptance mask required")
                    source[seed, vi, mi] = values
            seed_plans.append(method_plans)
        plans.append(seed_plans)
    return id_groups, ood_groups, plans, id_masks, ood_masks


def run(stop_after_blocks: int) -> None:
    require(1 <= stop_after_blocks <= 10, "fixed block boundary 1..10")
    declaration = read(RUN / "declaration.json")
    require(declaration["state"] == "frozen_before_stratified_draws" and
            declaration["replicates"] == 1000 and
            declaration["rng_seed_base"] == 2035 and
            declaration["id_classes"] == 80 and
            declaration["ood_classes"] == 20 and
            declaration["draws_per_class"] == 100 and
            declaration["seeds"] == [0, 1, 2] and
            declaration["variants"] == list(VARIANTS) and
            declaration["methods"] == list(METHODS) and
            declaration["metrics"] == list(METRICS) and
            declaration["fixed_id_target"] == 95,
            "fixed complete bootstrap declaration required")
    for item in declaration["inputs_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "original complete input changed: " + item["path"])
    source_hashes = {name: sha(ROOT / name) for name in SOURCE}
    launch_path = RUN / "launch.json"
    if launch_path.exists():
        launch = read(launch_path)
        require(launch["declaration_sha256"] == sha(RUN / "declaration.json") and
                launch["source_sha256"] == source_hashes and
                launch["replicates"] == 1000 and
                launch["block_size"] == 100,
                "incompatible source/input resume")
    else:
        snapshot = RUN / "source_snapshot"
        require(not snapshot.exists(), "partial source snapshot; preserve it")
        snapshot.mkdir()
        for name in SOURCE:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        write(launch_path, {
            "state": "fixed_1000_draw_schedule_launched",
            "launched_at": now(),
            "declaration_sha256": sha(RUN / "declaration.json"),
            "source_sha256": source_hashes,
            "replicates": 1000, "block_size": 100,
            "boundary": "Paired, within-class frozen-score image resampling only.",
        })
    id_groups, ood_groups, plans, id_masks, ood_masks = inputs()
    blocks = RUN / "blocks"
    blocks.mkdir(exist_ok=True)
    completed = []
    for block in range(10):
        stem = f"block-{block:03d}"
        marker_path = blocks / f"{stem}.json"
        weight_path = blocks / f"{stem}-weights.npz"
        metric_path = blocks / f"{stem}-metrics.npz"
        if marker_path.exists():
            marker = read(marker_path)
            require(marker["state"] == "committed" and
                    marker["start"] == block * 100 and
                    marker["end_exclusive"] == (block + 1) * 100 and
                    marker["launch_sha256"] == sha(launch_path) and
                    marker["weights_sha256"] == sha(weight_path) and
                    marker["metrics_sha256"] == sha(metric_path),
                    "completed block changed")
            completed.append(block)
            continue
        if block >= stop_after_blocks:
            break
        require(not weight_path.exists() and not metric_path.exists(),
                "partial block retained; inspect before retrying")
        write(RUN / "status.json", {"state": "running", "pid": os.getpid(),
                                    "active_block": block,
                                    "completed_blocks": completed,
                                    "updated_at": now()})
        begun = time.monotonic()
        id_weights = np.empty((100, 8000), dtype=np.uint16)
        ood_weights = np.empty((100, 2000), dtype=np.uint16)
        ranking = np.empty((100, 3, 2, 6, 4), dtype=np.float64)
        id_accepted = np.empty((100, 3, 2, 6), dtype=np.int32)
        ood_accepted = np.empty((100, 3, 2, 6), dtype=np.int32)
        for local in range(100):
            replicate = block * 100 + local
            wi, wo = draw_counts(replicate, id_groups, ood_groups)
            id_weights[local], ood_weights[local] = wi, wo
            id_accepted[local] = np.einsum("svmi,i->svm", id_masks,
                                           wi.astype(np.int32), optimize=True)
            ood_accepted[local] = np.einsum("svmi,i->svm", ood_masks,
                                            wo.astype(np.int32), optimize=True)
            for seed in range(3):
                for vi in range(2):
                    for mi in range(6):
                        ranking[local, seed, vi, mi] = plans[seed][vi][mi].evaluate(
                            wi, wo)[:4]
        saved(weight_path, replicates=np.arange(block * 100, (block + 1) * 100,
                                                dtype=np.int32),
              id_weights=id_weights, ood_weights=ood_weights)
        saved(metric_path, replicates=np.arange(block * 100, (block + 1) * 100,
                                                dtype=np.int32),
              ranking=ranking, id_accepted_counts=id_accepted,
              ood_accepted_counts=ood_accepted)
        write(marker_path, {"state": "committed", "start": block * 100,
                            "end_exclusive": (block + 1) * 100,
                            "launch_sha256": sha(launch_path),
                            "weights_sha256": sha(weight_path),
                            "metrics_sha256": sha(metric_path),
                            "elapsed_seconds": time.monotonic() - begun,
                            "committed_at": now()})
        completed.append(block)
        print(json.dumps({"block": block, "completed_replicates":
                          len(completed) * 100, "seconds":
                          round(time.monotonic() - begun, 2)}), flush=True)
    for item in declaration["inputs_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "original input changed during bootstrap")
    require(source_hashes == {name: sha(ROOT / name) for name in SOURCE},
            "bootstrap source changed during execution")
    write(RUN / "status.json", {
        "state": ("raw_complete_pending_independent_audit"
                  if len(completed) == 10 else "paused_at_fixed_block_boundary"),
        "completed_blocks": completed,
        "completed_replicates": 100 * len(completed),
        "expected_replicates": 1000,
        "launch_sha256": sha(launch_path),
        "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after-blocks", type=int, default=10)
    args = parser.parse_args()
    lock_path = RUN / "controller.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run(args.stop_after_blocks)
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
