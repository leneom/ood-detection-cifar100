"""Extract frozen LogitNorm classifier outputs, features and the six original T=1 scores.

Every input must be byte-identical to the CE extractions it is compared with:
ID and original-four pixel fingerprints from runs/validation-isolation, and the
recorded cache SHA-256 of each later population. Run after all training ends.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ood.data import load_cifar, stratified_split
from src.ood.datasets import load_ood
from src.ood.evaluate import image_fingerprint
from src.ood.features import extract, KNN, Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import atomic_json, environment, file_hash, seed_all
from src.ood.scores import METHODS as LOGIT_METHODS, score_logits

PHASE = ROOT / "runs" / "logitnorm-representation"
ISOLATED = ROOT / "runs" / "validation-isolation"
ORIGINAL = ("cifar10", "mnist", "svhn", "textures")
LATER = {"tinyimagenet": ("tinyimagenet-confirmation", "tinyimagenet-validation.npz"),
         "fashionmnist": ("fashionmnist-extension", "fashionmnist-test-32x32.npz"),
         "places365": ("places365-extension", "places365-validation.npz"),
         "oxford_pets": ("oxford-pets-extension", "pets-test.npz")}
SOURCE = ("tools/extract_logitnorm_eval.py", "src/ood/data.py", "src/ood/datasets.py", "src/ood/cache.py",
          "src/ood/textures_cache.py", "src/ood/evaluate.py", "src/ood/features.py", "src/ood/models.py",
          "src/ood/runtime.py", "src/ood/scores.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_arrays(path, **arrays):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def validate_selection(seed):
    declaration = read(PHASE / "declaration.json")
    if declaration["state"] != "frozen_before_training":
        raise ValueError("frozen training declaration required")
    run = PHASE / f"seed{seed}"
    status, selection = read(run / "status.json"), read(run / "final-selection.json")
    history, config = read(run / "history.json"), read(run / "config.json")
    if status["state"] != "complete" or status["epoch"] != 100 or len(history) != 100:
        raise ValueError("seed training incomplete")
    if not all(np.isfinite(row["train_loss"]) for row in history):
        raise ValueError("non-finite training loss recorded")
    best = max(history, key=lambda row: row["val_accuracy"])
    checkpoint = run / "final-best.pt"
    if (selection["epoch"] != best["epoch"] or selection["count"] != 2000 or
            selection["selection_population"] != "ID_checkpoint_validation" or
            selection["checkpoint_sha256"] != file_hash(checkpoint)):
        raise ValueError("frozen checkpoint selection invalid")
    if (config["seed"] != seed or config["objective"] != "logitnorm" or config["tau"] != declaration["tau"] or
            config["validation_partition"]["partition_sha256"] != file_hash(ISOLATED / "partition.npz")):
        raise ValueError("training configuration or partition differs")
    for name, expected in read(run / "source_hashes.json").items():
        if declaration["training_sources"][f"src/ood/{name}"] != expected:
            raise ValueError("model was trained by an undeclared source: " + name)
    return run, checkpoint, config, selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for feature extraction")
    run, checkpoint, config, selection = validate_selection(a.seed)
    output = PHASE / f"extract-seed{a.seed}"
    if output.exists():
        raise FileExistsError("extract output exists; preserve it and inspect before a new attempt")
    ce_manifest = read(ISOLATED / f"extract-seed{a.seed}" / "manifest.json")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCE}
    seed_all(0)
    torch.set_grad_enabled(False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != selection["epoch"] or state["config"] != config:
        raise ValueError("checkpoint payload differs from frozen selection")
    model = build_model(config["model"], config["classes"]).to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    root = ROOT / "data"
    train_images, train_labels = load_cifar(root, "cifar100", True)
    test_images, test_labels = load_cifar(root, "cifar100", False)
    train_idx, val_idx = stratified_split(train_labels.numpy(), 50, 2026)
    saved = np.load(run / "split_indices.npz", allow_pickle=False)
    if not (np.array_equal(saved["train"], train_idx) and np.array_equal(saved["val"], val_idx)):
        raise ValueError("saved training/validation index mismatch")
    fingerprints = ce_manifest["input_pixels"]
    populations = {}
    for role, images in (("id_train", train_images[train_idx]), ("id_validation", train_images[val_idx]),
                         ("id_test", test_images)):
        if image_fingerprint(images) != fingerprints[role]:
            raise ValueError("ID pixels differ from the CE extraction: " + role)
    for name in ORIGINAL:
        images = load_ood(root, name, len(test_images))
        if image_fingerprint(images) != fingerprints["ood"][name]:
            raise ValueError("OOD pixels differ from the CE extraction: " + name)
        populations[name] = images
    for name, (folder, cache) in LATER.items():
        recorded = read(ROOT / "runs" / folder / f"extract-seed{a.seed}" / "manifest.json")["cache_sha256"]
        if file_hash(root / cache) != recorded:
            raise ValueError("cache differs from the CE extraction: " + name)
        with np.load(root / cache, allow_pickle=False) as z:
            populations[name] = torch.from_numpy(z["images"].copy())

    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    for name in SOURCE:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    atomic_json(output / "status.json", {"state": "extracting", "seed": a.seed,
                                         "started_at": datetime.now(timezone.utc).isoformat()})
    outputs = {}

    def saved_file(path, **arrays):
        save_arrays(path, **arrays)
        outputs[path.name] = file_hash(path)

    def t1_scores(logits):
        return {method: score_logits(logits, method, 1.).numpy() for method in LOGIT_METHODS}

    train_logits, train_features = extract(model, train_images[train_idx], a.batch_size, "cuda")
    saved_file(output / "id_train_raw.npz", logits=train_logits.numpy(), features=train_features.numpy(),
               labels=train_labels[train_idx].numpy(), original_indices=train_idx)
    knn = KNN(k=50, device="cuda").fit(train_features)
    maha = Mahalanobis(shrinkage=.01, device="cuda").fit(train_features, train_labels[train_idx])

    def all_scores(logits, features):
        return {**t1_scores(logits), "knn": knn.score(features).numpy(), "mahalanobis": maha.score(features).numpy()}

    for role, images, labels in (("id_validation", train_images[val_idx], train_labels[val_idx]),
                                 ("id_test", test_images, test_labels)):
        logits, features = extract(model, images, a.batch_size, "cuda")
        saved_file(output / f"{role}_raw.npz", logits=logits.numpy(), features=features.numpy(),
                   labels=labels.numpy())
        saved_file(output / f"{role}_scores.npz", **all_scores(logits, features))
    for name, images in populations.items():
        logits, features = extract(model, images, a.batch_size, "cuda")
        saved_file(output / f"{name}_raw.npz", logits=logits.numpy(), features=features.numpy())
        saved_file(output / f"{name}_scores.npz", **all_scores(logits, features))
        print(json.dumps({"seed": a.seed, "population": name, "count": len(images)}), flush=True)
    if file_hash(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint changed during extraction")
    if any(file_hash(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("extraction source changed during execution")
    manifest = {"state": "extracted_pending_scoring_and_audit", "seed": a.seed,
                "checkpoint_sha256": selection["checkpoint_sha256"], "checkpoint_epoch": selection["epoch"],
                "declaration_sha256": file_hash(PHASE / "declaration.json"),
                "input_checks": "ID and original-four pixel fingerprints and later-population cache SHA-256 "
                                "equal the CE extractions",
                "feature_fit": {"population": "ID training only", "count": 45000, "knn_k": 50,
                                "mahalanobis_shrinkage": .01},
                "environment": environment(), "source_hashes": source_hashes, "outputs": outputs,
                "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", {"state": manifest["state"], "seed": a.seed,
                                         "manifest_sha256": file_hash(output / "manifest.json")})
    print(json.dumps({"seed": a.seed, "state": manifest["state"], "outputs": len(outputs)}), flush=True)


if __name__ == "__main__":
    main()
