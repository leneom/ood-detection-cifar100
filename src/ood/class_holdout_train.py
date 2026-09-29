"""Fixed 100-epoch ID-only training for the audited CIFAR-100 80-class holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from .data import load_cifar, index_hash, preprocess
from .models import build_model
from .runtime import (seed_all, environment, atomic_json, atomic_checkpoint,
                      predict, file_hash)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "cifar100-class-holdout"
PARTITION = RUN / "partition.npz"
SOURCE_NAMES = (
    "src/ood/class_holdout_train.py", "src/ood/data.py", "src/ood/models.py",
    "src/ood/runtime.py", "tools/prepare_class_holdout.py",
    "tools/audit_class_holdout_partition.py", "docs/CIFAR100_CLASS_HOLDOUT.md",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_partition() -> tuple[dict[str, np.ndarray], dict]:
    declaration = read(RUN / "declaration.json")
    manifest = read(RUN / "partition.json")
    audit = read(RUN / "partition-audit.json")
    if not (declaration["state"] == "declared_before_class_partition" and
            manifest["state"] == "partition_prepared_pending_independent_audit" and
            audit["state"] == "passed" and
            manifest["declaration_sha256"] == file_hash(RUN / "declaration.json") and
            manifest["partition_sha256"] == audit["partition_sha256"] == file_hash(PARTITION) and
            audit["partition_manifest_sha256"] == file_hash(RUN / "partition.json") and
            declaration["protocol_sha256"] == manifest["protocol_sha256"] ==
            file_hash(ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md") and
            manifest["source_sha256"] == file_hash(ROOT / "tools/prepare_class_holdout.py") and
            audit["source_sha256"] == file_hash(ROOT / "tools/audit_class_holdout_partition.py") and
            declaration["train_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/train") and
            declaration["meta_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/meta")):
        raise ValueError("class/ID-role partition is not independently frozen")
    with np.load(PARTITION, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    names = ("train_indices", "validation_indices", "checkpoint_indices",
             "temperature_indices", "threshold_indices", "excluded_ood_train_indices")
    expected_counts = (36000, 4000, 1600, 1200, 1200, 10000)
    if ([len(arrays[name]) for name in names] != list(expected_counts) or
            len(arrays["id_fine_labels"]) != 80 or len(arrays["ood_fine_labels"]) != 20 or
            not np.array_equal(np.sort(np.concatenate((arrays["train_indices"],
                arrays["validation_indices"], arrays["excluded_ood_train_indices"]))),
                np.arange(50000)) or
            not np.array_equal(np.sort(np.concatenate((arrays["checkpoint_indices"],
                arrays["temperature_indices"], arrays["threshold_indices"]))),
                arrays["validation_indices"])):
        raise ValueError("fixed 36k/4k/10k and 1600/1200/1200 row isolation differs")
    evidence = {"partition_sha256": file_hash(PARTITION),
                "partition_manifest_sha256": file_hash(RUN / "partition.json"),
                "partition_audit_sha256": file_hash(RUN / "partition-audit.json"),
                "declaration_sha256": file_hash(RUN / "declaration.json"),
                "checkpoint_population": "ID_checkpoint_validation_1600"}
    return arrays, evidence


def train(seed: int, stop_after: int, resume: bool) -> None:
    if seed not in (0, 1, 2) or not 0 < stop_after <= 100:
        raise ValueError("fixed seeds 0/1/2 and stop-after 1..100 required")
    if not torch.cuda.is_available():
        raise RuntimeError("verified Windows CUDA device required")
    arrays, evidence = frozen_partition()
    sources = {name: file_hash(ROOT / name) for name in SOURCE_NAMES}
    seed_all(seed, deterministic=True)
    image_tensor, original_labels = load_cifar(ROOT / "data", "cifar100", True, False)
    if len(image_tensor) != 50000 or len(original_labels) != 50000:
        raise ValueError("official CIFAR-100 training data differs")
    id_fine = arrays["id_fine_labels"]
    ood_fine = arrays["ood_fine_labels"]
    lookup = torch.full((100,), -1, dtype=torch.long)
    lookup[torch.from_numpy(id_fine)] = torch.arange(80)
    remapped = lookup[original_labels]
    train_rows = arrays["train_indices"]
    validation_rows = arrays["validation_indices"]
    checkpoint_rows = arrays["checkpoint_indices"]
    if (torch.any(remapped[train_rows] < 0) or torch.any(remapped[validation_rows] < 0) or
            not torch.all(remapped[arrays["excluded_ood_train_indices"]] == -1) or
            not np.array_equal(original_labels[train_rows].numpy(),
                               arrays["train_original_fine_labels"]) or
            not np.array_equal(original_labels[validation_rows].numpy(),
                               arrays["validation_original_fine_labels"])):
        raise ValueError("held-out class leakage or original-row label mismatch")
    output = RUN / f"seed{seed}"
    if output.exists() and not resume:
        raise FileExistsError("existing run requires resume or audit")
    if not output.exists() and resume:
        raise FileNotFoundError("resume requested before run exists")
    config = {"dataset": "cifar100_class_holdout", "model": "resnet18",
              "classes": 80, "seed": seed, "epochs": 100, "batch_size": 128,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "milestones": [50, 75, 90], "amp": True, "deterministic": True,
              "device": "cuda", "normalization": "cifar100_mean_std",
              "class_partition": evidence, "id_fine_labels": id_fine.tolist(),
              "ood_fine_labels": ood_fine.tolist()}
    split = {"train_count": 36000, "validation_count": 4000,
             "checkpoint_count": 1600, "temperature_count": 1200,
             "threshold_count": 1200,
             "train_indices_sha256": index_hash(train_rows),
             "validation_indices_sha256": index_hash(validation_rows),
             "checkpoint_indices_sha256": index_hash(checkpoint_rows),
             "partition_sha256": evidence["partition_sha256"],
             "test_used_during_training": False,
             "held_out_training_images_used": False}
    train_images = image_tensor[train_rows].to("cuda")
    train_labels = remapped[train_rows].to("cuda")
    val_images, val_labels = image_tensor[checkpoint_rows], remapped[checkpoint_rows]
    model = build_model("resnet18", 80).to("cuda").to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict] = []
    start_epoch, best_accuracy = 0, -1.0
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        previous_best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if (state["config"] != config or state["split"] != split or
                state["source_hashes"] != sources or
                state["epoch"] != len(state["history"]) or
                previous_best["epoch"] > state["epoch"] or
                previous_best["val_accuracy"] != state["best_accuracy"]):
            raise ValueError("resume config/split/source or checkpoint commit differs")
        with np.load(output / "split_indices.npz", allow_pickle=False) as saved:
            for name, expected in (("train", train_rows), ("validation", validation_rows),
                                   ("checkpoint", checkpoint_rows)):
                if not np.array_equal(saved[name], expected):
                    raise ValueError("resume row indices changed: " + name)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        best_accuracy = state["best_accuracy"]
        history = state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        output.mkdir(parents=True)
        atomic_json(output / "config.json", config)
        atomic_json(output / "environment.json", environment())
        atomic_json(output / "split.json", split)
        atomic_json(output / "source_hashes.json", sources)
        np.savez_compressed(output / "split_indices.npz", train=train_rows,
                            validation=validation_rows, checkpoint=checkpoint_rows)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for name in SOURCE_NAMES:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
    print(json.dumps({"state": "training", "seed": seed, "train": 36000,
                      "checkpoint_validation": 1600, "resume_epoch": start_epoch}), flush=True)
    for epoch in range(start_epoch, stop_after):
        begun = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_images), device="cuda")
        loss_sum, correct = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        for offset in range(0, len(order), 128):
            indices = order[offset:offset + 128]
            inputs = preprocess(train_images[indices], augment=True)
            targets = train_labels[indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == targets).sum())
        _, accuracy = predict(model, val_images, val_labels, 128, "cuda")
        scheduler.step()
        row = {"epoch": epoch + 1, "lr": lr,
               "train_loss": loss_sum / len(order),
               "train_accuracy": correct / len(order),
               "val_accuracy": accuracy,
               "seconds": time.perf_counter() - begun}
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            atomic_checkpoint(output / "best.pt", {"model": model.state_dict(),
                "config": config, "split": split, "epoch": epoch + 1,
                "val_accuracy": accuracy, "source_hashes": sources,
                "checkpoint_role": "ID_checkpoint_validation", "checkpoint_count": 1600})
        atomic_checkpoint(output / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch + 1, "best_accuracy": best_accuracy,
            "history": history, "config": config, "split": split,
            "source_hashes": sources, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        })
        atomic_json(output / "history.json", history)
        atomic_json(output / "status.json", {
            "state": "complete" if epoch + 1 == 100 else
                     "paused" if epoch + 1 == stop_after else "training",
            "epoch": epoch + 1, "planned_epochs": 100,
            "best_val_accuracy": best_accuracy,
            "checkpoint_validation_count": 1600,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps(row), flush=True)
    if stop_after == 100:
        selected = output / "final-best.pt"
        if selected.exists():
            raise FileExistsError("immutable final-best.pt already exists")
        temp = selected.with_suffix(".pt.tmp")
        shutil.copyfile(output / "best.pt", temp)
        temp.replace(selected)
        best = torch.load(selected, map_location="cpu", weights_only=True)
        atomic_json(output / "final-selection.json", {
            "state": "frozen", "epoch": best["epoch"],
            "checkpoint_sha256": file_hash(selected),
            "selection_population": "ID_checkpoint_validation",
            "count": 1600, "source_hashes": sources,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--stop-after", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.seed, args.stop_after, args.resume)


if __name__ == "__main__":
    main()
