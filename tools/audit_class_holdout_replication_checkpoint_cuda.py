"""Resolve a CPU/CUDA ID-checkpoint disagreement on the original CUDA path."""
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
from tools.run_class_holdout_training import gpu_samples, idle

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
    require(torch.cuda.is_available(), "original Windows CUDA device required")
    status = read(BASE / "status.json")
    require(status["state"] == "training_complete_pending_independent_audit" and
            status["trainer_running"] is False and len(status["completed"]) == 6,
            "all six schedules must complete before any CUDA inference audit")
    split = BASE / f"split{args.class_seed}"
    folder = split / f"seed{args.seed}"
    cpu_folder = split / f"checkpoint-cpu-audit-seed{args.seed}"
    output = split / f"checkpoint-cuda-audit-seed{args.seed}"
    require(not output.exists(), "write-once CUDA audit already exists")
    training = read(split / f"train-audit-seed{args.seed}/audit.json")
    cpu = read(cpu_folder / "audit.json")
    selection = read(folder / "final-selection.json")
    require(training["state"] == "passed" and
            cpu["state"] == "cpu_cuda_discrepancy_requires_cuda_audit" and
            training["class_seed"] == cpu["class_seed"] == args.class_seed and
            training["seed"] == cpu["seed"] == args.seed and
            training["selected_checkpoint_sha256"] ==
            cpu["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") and
            cpu["raw_id_validation_sha256"] ==
            sha(cpu_folder / "checkpoint-validation.npz") and
            training["partition_sha256"] == cpu["partition_sha256"] ==
            sha(split / "partition.npz"),
            "frozen selected checkpoint and independent CPU evidence")
    with np.load(cpu_folder / "checkpoint-validation.npz", allow_pickle=False) as saved:
        cpu_arrays = {name: saved[name] for name in saved.files}
    with np.load(split / "partition.npz", allow_pickle=False) as partition:
        rows = partition["checkpoint_indices"].astype(np.int64)
        ids = partition["id_fine_labels"].astype(np.int64)
    require(np.array_equal(cpu_arrays["original_train_indices"], rows) and
            len(rows) == 1600 and len(ids) == 80 and
            cpu_arrays["logits"].shape == (1600, 80) and
            np.array_equal(cpu_arrays["predictions"],
                           cpu_arrays["logits"].argmax(axis=1)),
            "full CPU ID-role raw logits/rows")
    images, fine = load_cifar(ROOT / "data", "cifar100", True, False)
    lookup = torch.full((100,), -1, dtype=torch.long)
    lookup[torch.from_numpy(ids)] = torch.arange(80)
    targets = lookup[fine[rows]]
    require(len(images) == 50000 and
            np.array_equal(cpu_arrays["original_fine_labels"], fine[rows].numpy()) and
            np.array_equal(cpu_arrays["id_class_targets"], targets.numpy()) and
            int(np.sum(cpu_arrays["predictions"] == targets.numpy())) ==
            cpu["recomputed_correct_count"], "CPU checkpoint evidence/source labels")
    selected = torch.load(folder / "final-best.pt", map_location="cpu",
                          weights_only=True)
    guard = gpu_samples()
    require(idle(guard), "GPU busy; no CUDA audit output created")
    model = build_model("resnet18", 80).to("cuda").to(
        memory_format=torch.channels_last)
    model.load_state_dict(selected["model"])
    model.eval()
    with torch.inference_mode():
        logits, accuracy = predict(model, images[rows], targets, 128, "cuda")
    predictions = logits.argmax(1)
    correct = int((predictions == targets).sum())
    require(logits.shape == (1600, 80) and torch.isfinite(logits).all() and
            accuracy == correct / 1600 == selected["val_accuracy"] ==
            training["selected_id_checkpoint_accuracy"],
            "original CUDA ID-only selected accuracy must reproduce exactly")
    changed = np.flatnonzero(cpu_arrays["predictions"] != predictions.numpy())
    require(len(changed) > 0 and
            cpu["recomputed_correct_count"] != correct,
            "CPU/CUDA discrepancy must be explicit")
    require(sha(folder / "final-best.pt") == selection["checkpoint_sha256"] and
            sha(split / "partition.npz") == training["partition_sha256"],
            "checkpoint/partition unchanged during audit")
    output.mkdir()
    snapshot = output / "executed_audit_source.py"
    shutil.copyfile(__file__, snapshot)
    raw = output / "checkpoint-validation-cuda.npz"
    with raw.open("wb") as stream:
        np.savez_compressed(stream, original_train_indices=rows,
                            original_fine_labels=fine[rows].numpy(),
                            id_class_targets=targets.numpy(),
                            predictions=predictions.numpy(), logits=logits.numpy())
    record = {
        "state": "passed", "class_seed": args.class_seed, "seed": args.seed,
        "selected_epoch": selected["epoch"], "id_checkpoint_validation_count": 1600,
        "recomputed_correct_count": correct, "recomputed_accuracy": accuracy,
        "cpu_recomputed_correct_count": cpu["recomputed_correct_count"],
        "cpu_cuda_prediction_disagreements": len(changed),
        "disagreement_rows": rows[changed].tolist(),
        "selected_checkpoint_sha256": sha(folder / "final-best.pt"),
        "training_audit_sha256": sha(split / f"train-audit-seed{args.seed}/audit.json"),
        "cpu_audit_sha256": sha(cpu_folder / "audit.json"),
        "partition_sha256": sha(split / "partition.npz"),
        "raw_id_validation_sha256": sha(raw),
        "gpu_guard_samples": guard, "source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Original CUDA ID checkpoint selection reproduced exactly; CPU drift preserved, no test/OOD data read.",
    }
    (output / "audit.json").write_text(json.dumps(record, indent=2,
                                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "class_seed": args.class_seed,
                      "seed": args.seed, "cuda_correct": correct,
                      "cpu_correct": cpu["recomputed_correct_count"],
                      "prediction_disagreements": len(changed)}), flush=True)


if __name__ == "__main__":
    main()
