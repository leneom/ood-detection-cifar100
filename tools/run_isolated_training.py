"""Run the predeclared validation-isolation classifiers, one GPU job at a time."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.runtime import atomic_json, file_hash
PHASE = ROOT / "runs" / "validation-isolation"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_complete(seed_dir, trainer_hash):
    status = read(seed_dir / "status.json")
    selection = read(seed_dir / "final-selection.json")
    history = read(seed_dir / "history.json")
    config = read(seed_dir / "config.json")
    split = read(seed_dir / "split.json")
    if status["state"] != "complete" or status["epoch"] != 100 or len(history) != 100:
        raise ValueError(f"incomplete 100-epoch run: {seed_dir}")
    if config["validation_partition"]["checkpoint_count"] != 2000 or split["checkpoint_count"] != 2000:
        raise ValueError("checkpoint population is not 2,000 ID validation rows")
    if file_hash(seed_dir / "source_snapshot" / "isolated_train.py") != trainer_hash:
        raise ValueError("executed trainer snapshot differs")
    best = max(history, key=lambda row: row["val_accuracy"])
    if (selection["epoch"] != best["epoch"] or
            selection["count"] != 2000 or
            selection["selection_population"] != "ID_checkpoint_validation" or
            selection["checkpoint_sha256"] != file_hash(seed_dir / "final-best.pt")):
        raise ValueError("frozen checkpoint does not match checkpoint-only ID accuracy selection")
    return {"seed": config["seed"], "epoch": 100, "selected_epoch": best["epoch"],
            "checkpoint_accuracy": best["val_accuracy"], "final_checkpoint_sha256": selection["checkpoint_sha256"]}


def run_schedule():
    if not torch.cuda.is_available():
        raise RuntimeError("the verified Windows CUDA device is unavailable")
    state = read(PHASE / "status.json")
    if state["state"] != "partition_ready_training_not_started" or state["trainer_running"]:
        raise RuntimeError("phase already launched; inspect its process/status before recovery")
    subprocess.run([sys.executable, "-X", "utf8", "tools/verify_validation_ready.py"],
                   cwd=ROOT, check=True)
    trainer_hash = file_hash(ROOT / "src/ood/isolated_train.py")
    source_hash = file_hash(__file__)
    launch = {"state": "fixed_training_schedule_launched", "launched_at": now(),
              "controller_source_sha256": source_hash, "trainer_source_sha256": trainer_hash,
              "protocol_sha256": file_hash(ROOT / "docs/VALIDATION_ISOLATION.md"),
              "partition_sha256": file_hash(PHASE / "partition.npz"),
              "partition_audit_sha256": file_hash(PHASE / "partition-audit.json"),
              "predeclaration_sha256": file_hash(PHASE / "predeclaration.json"),
              "python": sys.executable, "device": torch.cuda.get_device_name(0),
              "seeds": [0, 1, 2], "epochs_per_seed": 100,
              "selection_population": "ID_checkpoint_validation_2000"}
    if (PHASE / "training-launch.json").exists():
        raise FileExistsError("training launch record already exists")
    atomic_json(PHASE / "training-launch.json", launch)
    completed = []
    for seed in range(3):
        if file_hash(ROOT / "src/ood/isolated_train.py") != trainer_hash:
            raise ValueError("trainer changed after fixed schedule launch")
        seed_dir = PHASE / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        if (seed_dir / "final-selection.json").exists():
            completed.append(verify_complete(seed_dir, trainer_hash))
            continue
        resume = (seed_dir / "config.json").exists()
        atomic_json(PHASE / "status.json", {"state": "training", "trainer_running": True,
                    "controller_pid": os.getpid(), "current_seed": seed, "resuming": resume,
                    "completed_seeds": completed, "updated_at": now(),
                    "training_launch_sha256": file_hash(PHASE / "training-launch.json")})
        command = [sys.executable, "-X", "utf8", "-m", "src.ood.isolated_train",
                   "--dataset", "cifar100", "--data-root", "data", "--archive-source", "../data",
                   "--model", "resnet18", "--seed", str(seed), "--split-seed", "2026",
                   "--val-per-class", "50", "--epochs", "100", "--batch-size", "128",
                   "--lr", "0.1", "--weight-decay", "0.0005", "--device", "cuda",
                   "--amp", "--output-dir", str(seed_dir)]
        if resume:
            command.append("--resume")
        with (seed_dir / "stdout.log").open("a", encoding="utf-8") as out, \
             (seed_dir / "stderr.log").open("a", encoding="utf-8") as err:
            out.write(json.dumps({"controller_started_at": now(), "resume": resume}) + "\n")
            out.flush()
            subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
        completed.append(verify_complete(seed_dir, trainer_hash))
    atomic_json(PHASE / "status.json", {"state": "training_complete_pending_evaluation",
                "trainer_running": False, "completed_seeds": completed, "updated_at": now(),
                "training_launch_sha256": file_hash(PHASE / "training-launch.json")})


def main():
    PHASE.mkdir(parents=True, exist_ok=True)
    with (PHASE / "controller.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run_schedule()
        except Exception as exc:
            atomic_json(PHASE / "status.json", {"state": "training_failed",
                        "trainer_running": False, "controller_pid": os.getpid(),
                        "error": repr(exc), "traceback": traceback.format_exc(),
                        "updated_at": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
