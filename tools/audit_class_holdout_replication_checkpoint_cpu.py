"""Independently recompute a selected model's ID checkpoint role on CPU."""
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

BASE = ROOT / "runs/cifar100-class-holdout-replication"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    require(not torch.cuda.is_available(), "CUDA must be hidden for independent CPU audit")
    torch.set_num_threads(4)
    split = BASE / f"split{args.class_seed}"
    folder = split / f"seed{args.seed}"
    destination = split / f"checkpoint-cpu-audit-seed{args.seed}"
    require(not destination.exists(), "write-once CPU inference audit exists")
    training_audit = read(split / f"train-audit-seed{args.seed}/audit.json")
    selection = read(folder / "final-selection.json")
    config = read(folder / "config.json")
    partition_audit = read(split / "partition-audit.json")
    require(training_audit["state"] == "passed" and
            training_audit["class_seed"] == args.class_seed and
            training_audit["seed"] == args.seed and
            training_audit["training_epoch_count"] == 100 and
            training_audit["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") and
            training_audit["partition_sha256"] == sha(split / "partition.npz") and
            partition_audit["state"] == "passed" and
            config["classes"] == 80 and config["seed"] == args.seed and
            selection["selection_population"] == "ID_checkpoint_validation" and
            selection["count"] == 1600,
            "completed, independently selected ID checkpoint required")
    with np.load(split / "partition.npz", allow_pickle=False) as stored:
        rows = stored["checkpoint_indices"].astype(np.int64)
        id_fine = stored["id_fine_labels"].astype(np.int64)
        train_rows = stored["train_indices"].astype(np.int64)
    require(len(rows) == 1600 and len(id_fine) == 80 and
            np.array_equal(id_fine, np.sort(id_fine)) and
            not np.intersect1d(rows, train_rows).size,
            "checkpoint ID rows disjoint from optimization ID")
    images, original_fine = load_cifar(ROOT / "data", "cifar100", True, False)
    lookup = torch.full((100,), -1, dtype=torch.long)
    lookup[torch.from_numpy(id_fine)] = torch.arange(80)
    targets = lookup[original_fine[rows]]
    require(len(images) == 50000 and torch.all(targets >= 0) and
            np.array_equal(np.bincount(targets.numpy(), minlength=80),
                           np.full(80, 20)), "frozen 1,600 ID rows/labels")
    selected = torch.load(folder / "final-best.pt", map_location="cpu",
                          weights_only=True)
    require(selected["config"] == config and
            selected["epoch"] == selection["epoch"] and
            selected["checkpoint_count"] == 1600,
            "selected payload/source binding")
    model = build_model("resnet18", 80)
    model.load_state_dict(selected["model"])
    model.eval()
    with torch.inference_mode():
        logits, accuracy = predict(model, images[rows], targets, 128, "cpu")
    predictions = logits.argmax(1)
    correct = int((predictions == targets).sum())
    require(logits.shape == (1600, 80) and torch.isfinite(logits).all() and
            accuracy == correct / 1600, "finite complete CPU ID-role logits")
    expected = selected["val_accuracy"]
    same = accuracy == expected == training_audit["selected_id_checkpoint_accuracy"]
    # A CPU/CUDA argmax boundary differed for one original seed. Preserve full
    # CPU evidence if this repeats, then demand a separate CUDA-path reproduction.
    destination.mkdir()
    snapshot = destination / "executed_audit_source.py"
    shutil.copyfile(__file__, snapshot)
    raw = destination / "checkpoint-validation.npz"
    with raw.open("wb") as stream:
        np.savez_compressed(stream, original_train_indices=rows,
                            original_fine_labels=original_fine[rows].numpy(),
                            id_class_targets=targets.numpy(),
                            predictions=predictions.numpy(), logits=logits.numpy())
    record = {
        "state": "passed" if same else "cpu_cuda_discrepancy_requires_cuda_audit",
        "class_seed": args.class_seed, "seed": args.seed,
        "selected_epoch": selected["epoch"], "id_checkpoint_validation_count": 1600,
        "recomputed_correct_count": correct, "recomputed_accuracy": accuracy,
        "training_cuda_correct_count": round(expected * 1600),
        "training_cuda_accuracy": expected,
        "selected_checkpoint_sha256": sha(folder / "final-best.pt"),
        "training_audit_sha256": sha(split / f"train-audit-seed{args.seed}/audit.json"),
        "partition_sha256": sha(split / "partition.npz"),
        "raw_id_validation_sha256": sha(raw),
        "source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "CPU inference on frozen ID checkpoint role only; a CPU/CUDA mismatch is not declared passed until CUDA is separately reproduced. No test/OOD file read.",
    }
    (destination / "audit.json").write_text(json.dumps(record, indent=2,
                                                       allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"state": record["state"], "class_seed": args.class_seed,
                      "seed": args.seed, "correct": correct,
                      "training_cuda_correct": record["training_cuda_correct_count"]}),
          flush=True)


if __name__ == "__main__":
    main()
