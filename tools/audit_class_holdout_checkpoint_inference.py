"""Recompute a selected 80-class checkpoint's ID-only validation accuracy on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.data import load_cifar
from src.ood.models import build_model
from src.ood.runtime import predict


RUN = ROOT / "runs" / "cifar100-class-holdout"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    require(not torch.cuda.is_available(), "hide CUDA for independent CPU inference")
    torch.set_num_threads(4)
    folder = RUN / f"seed{seed}"
    destination = RUN / f"checkpoint-inference-audit-seed{seed}"
    require(not destination.exists(), "checkpoint inference audit exists; preserve it")
    training_audit = read(RUN / f"train-audit-seed{seed}" / "audit.json")
    selection = read(folder / "final-selection.json")
    config = read(folder / "config.json")
    partition_audit = read(RUN / "partition-audit.json")
    require(training_audit["state"] == "passed" and
            training_audit["seed"] == seed and
            training_audit["training_epoch_count"] == 100 and
            training_audit["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") and
            training_audit["partition_sha256"] == sha(RUN / "partition.npz") and
            partition_audit["state"] == "passed" and
            config["classes"] == 80 and config["seed"] == seed and
            selection["selection_population"] == "ID_checkpoint_validation" and
            selection["count"] == 1600,
            "completed, independently selected ID checkpoint required")
    with np.load(RUN / "partition.npz", allow_pickle=False) as stored:
        rows = stored["checkpoint_indices"].astype(np.int64)
        id_fine = stored["id_fine_labels"].astype(np.int64)
        train_rows = stored["train_indices"].astype(np.int64)
    require(len(rows) == 1600 and len(id_fine) == 80 and
            np.array_equal(id_fine, np.sort(id_fine)) and
            not np.intersect1d(rows, train_rows).size,
            "checkpoint rows overlap optimization or ID class map differs")
    images, original_fine = load_cifar(ROOT / "data", "cifar100", True, False)
    lookup = torch.full((100,), -1, dtype=torch.long)
    lookup[torch.from_numpy(id_fine)] = torch.arange(80)
    targets = lookup[original_fine[rows]]
    require(len(images) == 50000 and torch.all(targets >= 0) and
            np.array_equal(np.bincount(targets.numpy(), minlength=80),
                           np.full(80, 20)),
            "frozen 1,600 ID checkpoint-role rows or labels differ")
    selected = torch.load(folder / "final-best.pt", map_location="cpu",
                          weights_only=True)
    require(selected["config"] == config and
            selected["epoch"] == selection["epoch"] and
            selected["checkpoint_count"] == 1600,
            "selected payload differs from independent training audit")
    model = build_model("resnet18", 80)
    model.load_state_dict(selected["model"])
    model.eval()
    with torch.inference_mode():
        logits, accuracy = predict(model, images[rows], targets, 128, "cpu")
    predicted = logits.argmax(1)
    correct = int((predicted == targets).sum())
    require(logits.shape == (1600, 80) and torch.isfinite(logits).all() and
            accuracy == correct / 1600 == selected["val_accuracy"] ==
            training_audit["selected_id_checkpoint_accuracy"],
            "recomputed ID checkpoint accuracy differs from selected payload")
    destination.mkdir()
    snapshot = destination / "executed_audit_source.py"
    shutil.copyfile(__file__, snapshot)
    raw_path = destination / "checkpoint-validation.npz"
    with raw_path.open("wb") as stream:
        np.savez_compressed(stream, original_train_indices=rows,
                            original_fine_labels=original_fine[rows].numpy(),
                            id_class_targets=targets.numpy(),
                            predictions=predicted.numpy(), logits=logits.numpy())
    record = {
        "state": "passed", "seed": seed, "selected_epoch": selected["epoch"],
        "id_checkpoint_validation_count": 1600,
        "recomputed_correct_count": correct,
        "recomputed_accuracy": accuracy,
        "selected_checkpoint_sha256": sha(folder / "final-best.pt"),
        "training_audit_sha256": sha(RUN / f"train-audit-seed{seed}" / "audit.json"),
        "partition_sha256": sha(RUN / "partition.npz"),
        "raw_id_validation_sha256": sha(raw_path),
        "source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "CPU inference on frozen ID checkpoint role only; no test or OOD data was loaded.",
    }
    (destination / "audit.json").write_text(json.dumps(record, indent=2,
                                                       allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "epoch": selected["epoch"], "correct": correct,
                      "accuracy": accuracy}), flush=True)


if __name__ == "__main__":
    main()
