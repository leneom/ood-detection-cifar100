"""Extract frozen validation-isolation classifier outputs and ID-fitted features.

No temperature, threshold, or detector choice is made here. Run only after
all three fixed GPU training schedules complete and no trainer uses the GPU.
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

from src.ood.data import load_cifar, stratified_split, index_hash
from src.ood.datasets import load_ood, description
from src.ood.evaluate import image_fingerprint
from src.ood.features import extract, KNN, Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import atomic_json, environment, file_hash, seed_all
from src.ood.scores import METHODS as LOGIT_METHODS, score_logits


PHASE = ROOT / "runs" / "validation-isolation"
OOD = ("cifar10", "mnist", "svhn", "textures", "noise")
SOURCE = ("tools/extract_isolated_eval.py", "src/ood/data.py", "src/ood/datasets.py",
          "src/ood/cache.py", "src/ood/textures_cache.py", "src/ood/evaluate.py",
          "src/ood/features.py", "src/ood/models.py", "src/ood/runtime.py",
          "src/ood/metrics.py", "src/ood/scores.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_arrays(path, **arrays):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def validate_selection(seed):
    phase = read(PHASE / "status.json")
    if phase["state"] != "training_complete_pending_evaluation" or phase["trainer_running"]:
        raise RuntimeError("all three training jobs must complete before GPU extraction")
    run = PHASE / f"seed{seed}"
    status, selection = read(run / "status.json"), read(run / "final-selection.json")
    history, config = read(run / "history.json"), read(run / "config.json")
    if status["state"] != "complete" or status["epoch"] != 100 or len(history) != 100:
        raise ValueError("seed training incomplete")
    best = max(history, key=lambda row: row["val_accuracy"])
    checkpoint = run / "final-best.pt"
    if (selection["epoch"] != best["epoch"] or selection["count"] != 2000 or
            selection["selection_population"] != "ID_checkpoint_validation" or
            selection["checkpoint_sha256"] != file_hash(checkpoint)):
        raise ValueError("frozen checkpoint selection invalid")
    if (config["seed"] != seed or config["validation_partition"]["checkpoint_count"] != 2000 or
            config["validation_partition"]["partition_sha256"] != file_hash(PHASE / "partition.npz")):
        raise ValueError("training configuration or partition differs")
    return run, checkpoint, config, selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("verified Windows CUDA device required for feature extraction")
    if a.batch_size <= 0:
        raise ValueError("batch size must be positive")
    run, checkpoint, config, selection = validate_selection(a.seed)
    output = PHASE / f"extract-seed{a.seed}"
    if output.exists():
        raise FileExistsError("extract output exists; preserve it and inspect before a new attempt")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCE}
    seed_all(0)
    torch.set_grad_enabled(False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != selection["epoch"] or state["config"] != config:
        raise ValueError("checkpoint payload differs from frozen selection")
    model = build_model(config["model"], config["classes"]).to("cuda")
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    train_images, train_labels = load_cifar(config["data_root"], "cifar100", True)
    test_images, test_labels = load_cifar(config["data_root"], "cifar100", False)
    train_idx, val_idx = stratified_split(train_labels.numpy(), 50, 2026)
    saved = np.load(run / "split_indices.npz", allow_pickle=False)
    partition = np.load(PHASE / "partition.npz", allow_pickle=False)
    if not all(np.array_equal(saved[key], expected) for key, expected in
               (("train", train_idx), ("val", val_idx),
                ("checkpoint", partition["checkpoint_original_indices"]))):
        raise ValueError("saved training/validation/checkpoint index mismatch")
    if not np.array_equal(partition["original_validation_indices"], val_idx):
        raise ValueError("original validation partition differs")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCE:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    atomic_json(output / "status.json", {"state": "extracting", "seed": a.seed,
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "started_at": datetime.now(timezone.utc).isoformat()})
    inputs = {"id_train": image_fingerprint(train_images[train_idx]),
              "id_validation": image_fingerprint(train_images[val_idx]),
              "id_test": image_fingerprint(test_images), "ood": {}}
    outputs = {}

    def saved_file(path, **arrays):
        save_arrays(path, **arrays)
        outputs[path.name] = file_hash(path)

    def t1_scores(logits):
        return {method: score_logits(logits, method, 1.).numpy()
                for method in LOGIT_METHODS}

    train_logits, train_features = extract(model, train_images[train_idx], a.batch_size, "cuda")
    saved_file(output / "id_train_raw.npz", logits=train_logits.numpy(),
               features=train_features.numpy(), labels=train_labels[train_idx].numpy(),
               original_indices=train_idx)
    saved_file(output / "id_train_T1_scores.npz", **t1_scores(train_logits))
    val_logits, val_features = extract(model, train_images[val_idx], a.batch_size, "cuda")
    saved_file(output / "id_validation_raw.npz", logits=val_logits.numpy(),
               features=val_features.numpy(), labels=train_labels[val_idx].numpy(),
               original_indices=val_idx)
    saved_file(output / "id_validation_T1_scores.npz", **t1_scores(val_logits))
    test_logits, test_features = extract(model, test_images, a.batch_size, "cuda")
    saved_file(output / "id_test_raw.npz", logits=test_logits.numpy(),
               features=test_features.numpy(), labels=test_labels.numpy())
    saved_file(output / "id_test_T1_scores.npz", **t1_scores(test_logits))
    knn = KNN(k=50, device="cuda").fit(train_features)
    maha = Mahalanobis(shrinkage=.01, device="cuda").fit(train_features, train_labels[train_idx])

    def feature_scores(features):
        return {"knn": knn.score(features).numpy(), "mahalanobis": maha.score(features).numpy()}

    saved_file(output / "id_validation_feature_scores.npz", **feature_scores(val_features))
    saved_file(output / "id_test_feature_scores.npz", **feature_scores(test_features))
    for name in OOD:
        images = load_ood(config["data_root"], name, len(test_images))
        inputs["ood"][name] = image_fingerprint(images)
        logits, features = extract(model, images, a.batch_size, "cuda")
        saved_file(output / f"{name}_raw.npz", logits=logits.numpy(), features=features.numpy())
        saved_file(output / f"{name}_T1_scores.npz", **t1_scores(logits))
        saved_file(output / f"{name}_feature_scores.npz", **feature_scores(features))
        print(json.dumps({"seed": a.seed, "population": name, "count": len(images)}), flush=True)
    if file_hash(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint changed during extraction")
    if any(file_hash(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("extraction source changed during execution")
    result = {"state": "extracted_pending_CPU_calibration_and_audit", "seed": a.seed,
              "checkpoint": str(checkpoint), "checkpoint_sha256": selection["checkpoint_sha256"],
              "checkpoint_epoch": selection["epoch"], "selection_population": "ID_checkpoint_validation_2000",
              "validation_roles_partition_sha256": file_hash(PHASE / "partition.npz"),
              "validation_roles": {name: partition[name + "_rows"].tolist() for name in
                                   ("checkpoint", "temperature", "threshold")},
              "train_indices_sha256": index_hash(train_idx), "val_indices_sha256": index_hash(val_idx),
              "input_pixels": inputs, "population_descriptions": {name: description(name) for name in OOD},
              "feature_fit": {"population": "ID training only", "count": 45000,
                              "knn_k": 50, "mahalanobis_shrinkage": .01},
              "environment": environment(), "source_hashes": source_hashes, "outputs": outputs,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "Raw outputs/ID-trained feature scores only; no ID temperature or thresholds fitted."}
    atomic_json(output / "manifest.json", result)
    atomic_json(output / "status.json", {"state": "extracted_pending_CPU_calibration_and_audit",
                "seed": a.seed, "manifest_sha256": file_hash(output / "manifest.json"),
                "completed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"seed": a.seed, "state": "extracted_pending_CPU_calibration_and_audit",
                      "outputs": len(outputs)}), flush=True)


if __name__ == "__main__":
    main()
