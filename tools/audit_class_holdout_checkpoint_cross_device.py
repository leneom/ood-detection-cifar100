"""Audit a frozen ID-only checkpoint when CPU and CUDA top-1 differ by one row.

The original training selection was made on CUDA. Preserve both raw inference
paths and require CUDA to reproduce the recorded 1,600-row selection exactly.
No test image or OOD image is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ood.data import load_cifar
from src.ood.models import build_model
from src.ood.runtime import predict


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"


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


def gpu_samples() -> list[dict]:
    samples = []
    for n in range(3):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        require(len(rows) == 1, "exactly one GPU expected")
        used, util, memory, power, total = (part.strip() for part in rows[0].split(","))
        samples.append({"used_mib": int(used), "util_percent": int(util),
                        "memory_util_percent": int(memory), "power_w": float(power),
                        "total_mib": int(total)})
        if n < 2:
            time.sleep(2)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2,), required=True)
    seed = parser.parse_args().seed
    require(torch.cuda.is_available(), "CUDA required to reproduce original selection")
    torch.set_num_threads(4)
    folder = RUN / f"seed{seed}"
    destination = RUN / f"checkpoint-inference-audit-seed{seed}"
    require(not destination.exists(), "audit output exists; preserve it")
    training = read(RUN / f"train-audit-seed{seed}" / "audit.json")
    selection = read(folder / "final-selection.json")
    config = read(folder / "config.json")
    partition_audit = read(RUN / "partition-audit.json")
    require(training["state"] == partition_audit["state"] == "passed" and
            training["seed"] == seed and training["training_epoch_count"] == 100 and
            training["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") and
            training["partition_sha256"] == sha(RUN / "partition.npz") and
            config["classes"] == 80 and config["seed"] == seed and
            selection["selection_population"] == "ID_checkpoint_validation" and
            selection["count"] == 1600,
            "frozen, audited seed-2 ID checkpoint required")
    with np.load(RUN / "partition.npz", allow_pickle=False) as stored:
        rows = stored["checkpoint_indices"].astype(np.int64)
        ids = stored["id_fine_labels"].astype(np.int64)
        train_rows = stored["train_indices"].astype(np.int64)
    require(len(rows) == 1600 and len(ids) == 80 and
            np.array_equal(ids, np.sort(ids)) and
            not np.intersect1d(rows, train_rows).size,
            "ID-only checkpoint role differs")
    images, original_fine = load_cifar(ROOT / "data", "cifar100", True, False)
    lookup = torch.full((100,), -1, dtype=torch.long)
    lookup[torch.from_numpy(ids)] = torch.arange(80)
    targets = lookup[original_fine[rows]]
    require(len(images) == 50000 and torch.all(targets >= 0) and
            np.array_equal(np.bincount(targets.numpy(), minlength=80),
                           np.full(80, 20)),
            "ID validation labels or class mapping differ")
    selected = torch.load(folder / "final-best.pt", map_location="cpu",
                          weights_only=True)
    require(selected["config"] == config and
            selected["epoch"] == selection["epoch"] and
            selected["checkpoint_count"] == 1600,
            "selected payload differs")
    cpu_model = build_model("resnet18", 80)
    cpu_model.load_state_dict(selected["model"])
    cpu_model.eval()
    with torch.inference_mode():
        cpu_logits, cpu_accuracy = predict(cpu_model, images[rows], targets,
                                           128, "cpu")
    cpu_pred = cpu_logits.argmax(1)
    require(cpu_logits.shape == (1600, 80) and
            torch.isfinite(cpu_logits).all() and
            cpu_accuracy == int((cpu_pred == targets).sum()) / 1600,
            "independent CPU inference invalid")
    guard = gpu_samples()
    require(all(row["used_mib"] <= 4096 and row["util_percent"] <= 55 and
                row["memory_util_percent"] <= 10 and row["power_w"] <= 55
                for row in guard), "GPU busy; no audit output created")
    gpu_model = build_model("resnet18", 80).to("cuda").to(
        memory_format=torch.channels_last)
    gpu_model.load_state_dict(selected["model"])
    gpu_model.eval()
    with torch.inference_mode():
        gpu_logits, gpu_accuracy = predict(gpu_model, images[rows], targets,
                                           128, "cuda")
    gpu_pred = gpu_logits.argmax(1)
    disagreements = torch.nonzero(cpu_pred != gpu_pred).flatten().tolist()
    cpu_correct = int((cpu_pred == targets).sum())
    gpu_correct = int((gpu_pred == targets).sum())
    require(gpu_logits.shape == (1600, 80) and
            torch.isfinite(gpu_logits).all() and
            gpu_accuracy == gpu_correct / 1600 ==
            selected["val_accuracy"] ==
            training["selected_id_checkpoint_accuracy"] and
            len(disagreements) == 1 and
            cpu_correct == gpu_correct + 1 and
            cpu_accuracy == cpu_correct / 1600,
            "original CUDA ID selection not reproduced or CPU drift differs from one row")
    changed = disagreements[0]
    require(sha(folder / "final-best.pt") == selection["checkpoint_sha256"] and
            sha(RUN / "partition.npz") == training["partition_sha256"],
            "frozen checkpoint or partition changed mid-audit")
    destination.mkdir()
    snapshot = destination / "executed_audit_source.py"
    shutil.copyfile(__file__, snapshot)
    outputs = {}
    for name, logits, predictions in (
        ("checkpoint-validation.npz", cpu_logits, cpu_pred),
        ("checkpoint-validation-cuda.npz", gpu_logits, gpu_pred),
    ):
        path = destination / name
        with path.open("wb") as stream:
            np.savez_compressed(stream, original_train_indices=rows,
                                original_fine_labels=original_fine[rows].numpy(),
                                id_class_targets=targets.numpy(),
                                predictions=predictions.numpy(),
                                logits=logits.numpy())
        outputs[name] = sha(path)
    record = {
        "state": "passed", "seed": seed,
        "selected_epoch": selected["epoch"],
        "id_checkpoint_validation_count": 1600,
        "recomputed_correct_count": gpu_correct,
        "recomputed_accuracy": gpu_accuracy,
        "cpu_recomputed_correct_count": cpu_correct,
        "cpu_recomputed_accuracy": cpu_accuracy,
        "cpu_cuda_prediction_disagreements": len(disagreements),
        "disagreement": {"validation_position": changed,
                         "original_train_row": int(rows[changed]),
                         "target": int(targets[changed]),
                         "cpu_prediction": int(cpu_pred[changed]),
                         "cuda_prediction": int(gpu_pred[changed])},
        "selected_checkpoint_sha256": sha(folder / "final-best.pt"),
        "training_audit_sha256": sha(RUN / f"train-audit-seed{seed}" / "audit.json"),
        "partition_sha256": sha(RUN / "partition.npz"),
        "raw_id_validation_sha256": outputs["checkpoint-validation.npz"],
        "cuda_raw_id_validation_sha256": outputs["checkpoint-validation-cuda.npz"],
        "gpu_guard_samples": guard,
        "source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Original CUDA ID-checkpoint accuracy reproduced exactly; independent CPU differs by one borderline row. Both raw ID-only paths saved. No test/OOD image used.",
    }
    (destination / "audit.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "cuda_correct": gpu_correct, "cpu_correct": cpu_correct,
                      "disagreement_row": int(rows[changed])}), flush=True)


if __name__ == "__main__":
    main()
