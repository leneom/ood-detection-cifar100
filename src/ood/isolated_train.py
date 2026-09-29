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

from .data import load_cifar, stratified_split, index_hash, preprocess, synthetic
from .models import build_model
from .runtime import seed_all, environment, atomic_json, atomic_checkpoint, predict, training_source_hashes, file_hash


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "runs" / "validation-isolation"


def isolated_partition(dataset, val_indices, val_labels, train_indices):
    """Bind every frozen predecessor and return only checkpoint-selection rows."""
    if dataset == "synthetic":
        labels = val_labels.numpy()
        rows = np.concatenate([np.flatnonzero(labels == c)[:2] for c in range(4)])
        return np.sort(rows), {"mode": "synthetic_resume_smoke_only"}
    pre_path = PHASE / "predeclaration.json"
    audit_path = PHASE / "partition-audit.json"
    partition_path = PHASE / "partition.npz"
    metadata_path = PHASE / "partition.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if audit["state"] != "passed" or audit["partition_sha256"] != file_hash(partition_path) or metadata["partition_sha256"] != audit["partition_sha256"] or audit["predeclaration_sha256"] != file_hash(pre_path) or audit["partition_metadata_sha256"] != file_hash(metadata_path):
        raise ValueError("frozen ID-validation partition/audit binding differs")
    for name, expected in pre["original_artifacts"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("frozen predecessor differs: " + name)
    with np.load(partition_path, allow_pickle=False) as z:
        roles = {role: z[role + "_rows"].astype(np.int64) for role in ("checkpoint", "temperature", "threshold")}
        old_indices = z["original_validation_indices"].astype(np.int64)
        old_labels = z["validation_labels"].astype(np.int64)
        for role, n in (("checkpoint", 2000), ("temperature", 1500), ("threshold", 1500)):
            if len(roles[role]) != n or not np.array_equal(z[role + "_original_indices"], old_indices[roles[role]]):
                raise ValueError("ID-validation role/index mismatch: " + role)
    if not np.array_equal(old_indices, val_indices) or not np.array_equal(old_labels, val_labels.numpy()) or not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(5000)):
        raise ValueError("ID-validation roles do not cover the original validation split")
    old_train = np.load(ROOT / "runs/cifar100-seed0/split_indices.npz", allow_pickle=False)["train"]
    if not np.array_equal(old_train, train_indices):
        raise ValueError("training indices differ from original fixed schedule")
    evidence = {"mode": "three_way_ID_validation", "partition_sha256": file_hash(partition_path), "partition_audit_sha256": file_hash(audit_path), "predeclaration_sha256": file_hash(pre_path), "checkpoint_role": "checkpoint", "checkpoint_count": 2000}
    return roles["checkpoint"], evidence


def parser():
    p = argparse.ArgumentParser(description="Three-way-ID-validation isolated classifier training")
    p.add_argument("--dataset", choices=["cifar100", "synthetic"], default="cifar100")
    p.add_argument("--data-root", default="data")
    p.add_argument("--archive-source", default="../data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--model", choices=["resnet18", "tiny"], default="resnet18")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100, help="fixed training schedule horizon")
    p.add_argument("--stop-after", type=int, help="pause at this epoch without changing the schedule")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision; default full precision")
    p.add_argument("--no-deterministic", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p


def run(args):
    if args.dataset == "cifar100" and (args.model != "resnet18" or args.seed not in (0, 1, 2) or args.split_seed != 2026 or args.val_per_class != 50 or args.epochs != 100 or args.batch_size != 128 or args.lr != .1 or args.weight_decay != 5e-4 or not args.amp or args.no_deterministic):
        raise ValueError("real-data training must follow the frozen 100-epoch protocol")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    stop = args.epochs if args.stop_after is None else args.stop_after
    if not 0 < stop <= args.epochs:
        raise ValueError("stop-after must lie between 1 and epochs")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.dataset == "cifar100" and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("the fixed real-data schedule requires the verified CUDA device")
    seed_all(args.seed, not args.no_deterministic)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "config.json").exists():
        raise FileExistsError("existing run: use --resume or a new output directory")
    config = {k: v for k, v in vars(args).items() if k not in ("resume", "stop_after", "output_dir", "download", "archive_source")}
    config["data_root"] = str(Path(args.data_root).resolve())
    config["device"] = device
    config["normalization"] = "cifar100_mean_std"
    sources = training_source_hashes()
    sources["isolated_train.py"] = file_hash(__file__)
    if args.dataset == "synthetic":
        (images, labels), _ = synthetic(seed=args.seed)
        val_per_class, classes = 8, 4
    else:
        images, labels = load_cifar(args.data_root, "cifar100", True, args.download, args.archive_source)
        val_per_class, classes = args.val_per_class, 100
    train_indices, val_indices = stratified_split(labels.numpy(), val_per_class, args.split_seed)
    checkpoint_rows, evidence = isolated_partition(args.dataset, val_indices, labels[val_indices], train_indices)
    config["classes"] = classes
    config["validation_partition"] = evidence
    config["effective_val_per_class"] = val_per_class
    split = {
        "split_seed": args.split_seed, "train_count": len(train_indices), "val_count": len(val_indices),
        "train_indices_sha256": index_hash(train_indices), "val_indices_sha256": index_hash(val_indices),
        "checkpoint_count": len(checkpoint_rows), "checkpoint_original_indices_sha256": index_hash(val_indices[checkpoint_rows]),
        "checkpoint_role": "checkpoint", "test_used_during_training": False, "ood_used_during_training": False,
    }
    train_images, train_labels = images[train_indices].to(device), labels[train_indices].to(device)
    val_images, val_labels = images[val_indices[checkpoint_rows]], labels[val_indices[checkpoint_rows]]
    model = build_model(args.model, classes).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    milestones = sorted(set(max(1, int(args.epochs * f)) for f in (0.5, 0.75, 0.9)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    amp = args.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, start_epoch, best_accuracy = [], 0, -1.
    if args.resume:
        last_path = output / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError("resume requires last.pt")
        # Only load this project's own locally generated training checkpoint.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != config or state["split"] != split or state["source_hashes"] != sources:
            raise ValueError("resume config, data split, or training source differs")
        stored = np.load(output / "split_indices.npz", allow_pickle=False)
        if not all(np.array_equal(stored[k], v) for k, v in (("train", train_indices), ("val", val_indices), ("checkpoint", val_indices[checkpoint_rows]))):
            raise ValueError("saved split indices differ on resume")
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        if best["epoch"] > state["epoch"] or best["val_accuracy"] != state["best_accuracy"] or len(state["history"]) != state["epoch"]:
            raise ValueError("best/last checkpoint epoch commit is inconsistent")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_accuracy, history = state["epoch"], state["best_accuracy"], state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        atomic_json(output / "config.json", config)
        atomic_json(output / "environment.json", environment())
        atomic_json(output / "split.json", split)
        atomic_json(output / "source_hashes.json", sources)
        np.savez_compressed(output / "split_indices.npz", train=train_indices, val=val_indices, checkpoint=val_indices[checkpoint_rows])
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=False)
        for name in ("data.py", "models.py", "runtime.py", "train.py", "isolated_train.py"):
            shutil.copyfile(Path(__file__).parent / name, snapshot / name)
    print(json.dumps({"device": device, "train": len(train_images), "validation": len(val_images), "resume_epoch": start_epoch}), flush=True)
    for epoch in range(start_epoch, stop):
        begun = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_images), device=device)
        loss_sum, correct = 0., 0
        lr = optimizer.param_groups[0]["lr"]
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            inputs, targets = preprocess(train_images[indices], augment=True), train_labels[indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=amp):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss in epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == targets).sum())
        _, accuracy = predict(model, val_images, val_labels, args.batch_size, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / len(order),
               "train_accuracy": correct / len(order), "val_accuracy": accuracy,
               "seconds": time.perf_counter() - begun}
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            atomic_checkpoint(output / "best.pt", {"model": model.state_dict(), "config": config,
                              "split": split, "epoch": epoch + 1, "val_accuracy": accuracy,
                              "source_hashes": sources, "checkpoint_role": "checkpoint", "checkpoint_count": len(checkpoint_rows)})
        atomic_checkpoint(output / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_accuracy": best_accuracy,
            "history": history, "config": config, "split": split, "source_hashes": sources,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        })
        atomic_json(output / "history.json", history)
        atomic_json(output / "status.json", {"state": "complete" if epoch + 1 == args.epochs else "paused" if epoch + 1 == stop else "training",
                    "epoch": epoch + 1, "planned_epochs": args.epochs, "best_val_accuracy": best_accuracy, "checkpoint_validation_count": len(checkpoint_rows),
                    "updated_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(row), flush=True)
    if stop == args.epochs:
        selected = output / "final-best.pt"
        if selected.exists():
            raise FileExistsError("immutable final-best.pt already exists")
        temp_selected = output / "final-best.pt.tmp"
        shutil.copyfile(output / "best.pt", temp_selected)
        temp_selected.replace(selected)
        atomic_json(output / "final-selection.json", {"state": "frozen", "epoch": torch.load(selected, map_location="cpu", weights_only=False)["epoch"], "checkpoint_sha256": file_hash(selected), "selection_population": "ID_checkpoint_validation", "count": len(checkpoint_rows), "source_hashes": sources})
    return history


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
