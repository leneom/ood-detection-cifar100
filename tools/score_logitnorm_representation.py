"""Metrics for LogitNorm and seed-matched CE models on identical ID/OOD rows (CPU only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ood.metrics import detection_metrics, id_threshold

PHASE = ROOT / "runs" / "logitnorm-representation"
ISOLATED = ROOT / "runs" / "validation-isolation"
ORIGINAL = ("cifar10", "mnist", "svhn", "textures")
LATER = {"tinyimagenet": "tinyimagenet-confirmation", "fashionmnist": "fashionmnist-extension",
         "places365": "places365-extension", "oxford_pets": "oxford-pets-extension"}
POPULATIONS = (*ORIGINAL, *LATER)
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
OBJECTIVES = ("ce", "logitnorm")
TARGETS = (80, 90, 95, 99)
EVALUATION_SOURCES = ("tools/extract_logitnorm_eval.py", "tools/score_logitnorm_representation.py",
                      "tools/build_logitnorm_report.py", "tools/freeze_logitnorm_evaluation.py",
                      "src/ood/metrics.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def ce_scores(seed, role):
    if role in ("id_test", "id_validation"):
        saved = arrays(ISOLATED / f"score-seed{seed}" / f"{role}_scores.npz")
    elif role in ORIGINAL:
        saved = arrays(ISOLATED / f"score-seed{seed}" / f"{role}_scores.npz")
    else:
        saved = arrays(ROOT / "runs" / LATER[role] / f"score-seed{seed}" / "scores.npz")
    return {m: saved[f"T1_{m}"].astype(np.float64) for m in METHODS}


def ln_scores(seed, role):
    saved = arrays(PHASE / f"extract-seed{seed}" / f"{role}_scores.npz")
    return {m: saved[m].astype(np.float64) for m in METHODS}


def checked(seed):
    declaration = read(PHASE / "declaration.json")
    evaluation = read(PHASE / "evaluation-declaration.json")
    if evaluation["training_declaration_sha256"] != sha(PHASE / "declaration.json"):
        raise ValueError("evaluation declaration is not bound to the training declaration")
    for name, expected in evaluation["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("evaluation source changed: " + name)
    for name, expected in declaration["ce_comparators"][f"seed{seed}"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("CE comparator changed: " + name)
    folder = PHASE / f"extract-seed{seed}"
    status, manifest = read(folder / "status.json"), read(folder / "manifest.json")
    if status["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("LogitNorm extraction manifest changed")
    for name, expected in manifest["outputs"].items():
        if sha(folder / name) != expected:
            raise ValueError("LogitNorm extraction output changed: " + name)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    extraction = checked(seed)
    destination = PHASE / f"score-seed{seed}"
    if destination.exists():
        raise FileExistsError("scores already exist for this seed")
    rows = arrays(ISOLATED / "partition.npz")["threshold_rows"]
    frozen_ce = read(ISOLATED / f"score-seed{seed}" / "manifest.json")["datasets"]["cifar10"]["methods"]["T1"]
    loaders = {"ce": lambda role: ce_scores(seed, role), "logitnorm": lambda role: ln_scores(seed, role)}
    accuracy = {}
    ce_raw = arrays(ISOLATED / f"extract-seed{seed}" / "id_test_raw.npz")
    ln_raw = arrays(PHASE / f"extract-seed{seed}" / "id_test_raw.npz")
    if not np.array_equal(ce_raw["labels"], ln_raw["labels"]):
        raise ValueError("ID-test labels differ")
    accuracy["ce"] = float((ce_raw["logits"].argmax(1) == ce_raw["labels"]).mean())
    accuracy["logitnorm"] = float((ln_raw["logits"].argmax(1) == ln_raw["labels"]).mean())
    results = {}
    for objective in OBJECTIVES:
        load = loaders[objective]
        inside, validation = load("id_test"), load("id_validation")
        thresholds = {m: {t: id_threshold(validation[m][rows], t / 100) for t in TARGETS} for m in METHODS}
        if objective == "ce":
            for m in METHODS:
                for t in TARGETS:
                    if thresholds[m][t] != frozen_ce[m]["operating_grid"][f"target{t}"]["threshold"]:
                        raise ValueError(f"CE threshold differs from frozen isolated value: {m}")
        results[objective] = {}
        for population in POPULATIONS:
            outside = load(population)
            results[objective][population] = {}
            for m in METHODS:
                metrics = detection_metrics(inside[m], outside[m])
                grid = {}
                for t in TARGETS:
                    threshold = thresholds[m][t]
                    grid[f"target{t}"] = {"threshold": threshold,
                                          "id_test_acceptance": float(np.mean(inside[m] <= threshold)),
                                          "ood_acceptance": float(np.mean(outside[m] <= threshold)),
                                          "ood_count": len(outside[m])}
                results[objective][population][m] = {"metrics": metrics, "operating_grid": grid}
    destination.mkdir()
    manifest = {"state": "scored_pending_independent_audit", "seed": seed,
                "logitnorm_checkpoint_sha256": extraction["checkpoint_sha256"],
                "logitnorm_checkpoint_epoch": extraction["checkpoint_epoch"],
                "ce_checkpoint_sha256": read(ISOLATED / f"seed{seed}" / "final-selection.json")["checkpoint_sha256"],
                "evaluation_declaration_sha256": sha(PHASE / "evaluation-declaration.json"),
                "id_test_accuracy": accuracy, "populations": list(POPULATIONS), "methods": list(METHODS),
                "targets": list(TARGETS), "results": results,
                "created_at": datetime.now(timezone.utc).isoformat()}
    write(destination / "manifest.json", manifest)
    write(destination / "status.json", {"state": manifest["state"], "seed": seed,
                                        "manifest_sha256": sha(destination / "manifest.json")})
    print(json.dumps({"seed": seed, "state": manifest["state"], "accuracy": accuracy}), flush=True)


if __name__ == "__main__":
    main()
