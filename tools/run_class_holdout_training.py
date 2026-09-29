"""Guarded, source-bound sequential 100-epoch class-holdout training."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "cifar100-class-holdout"
sys.path.insert(0, str(ROOT))
from src.ood.class_holdout_train import frozen_partition
from src.ood.runtime import atomic_json, file_hash

MAX_USED_MIB = 4096
MAX_UTIL_PERCENT = 55
MAX_MEMORY_UTIL_PERCENT = 10
MAX_POWER_W = 55
SAMPLES = 3
GAP_SECONDS = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_samples() -> list[dict]:
    result = []
    for attempt in range(SAMPLES):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        if len(rows) != 1:
            raise RuntimeError("exactly one verified GPU is required")
        used, util, mem_util, power, total = (part.strip() for part in rows[0].split(","))
        result.append({"used_mib": int(used), "util_percent": int(util),
                       "memory_util_percent": int(mem_util), "power_w": float(power),
                       "total_mib": int(total)})
        if attempt + 1 < SAMPLES:
            time.sleep(GAP_SECONDS)
    return result


def idle(samples: list[dict]) -> bool:
    return all(sample["used_mib"] <= MAX_USED_MIB and
               sample["util_percent"] <= MAX_UTIL_PERCENT and
               sample["memory_util_percent"] <= MAX_MEMORY_UTIL_PERCENT and
               sample["power_w"] <= MAX_POWER_W for sample in samples)


def complete(seed: int, source_hash: str) -> dict | None:
    folder = RUN / f"seed{seed}"
    if not folder.exists():
        return None
    if not (folder / "final-selection.json").exists():
        if (folder / "config.json").exists() and (folder / "last.pt").exists():
            return None
        raise ValueError(f"seed {seed} partial folder requires separate review")
    status = read(folder / "status.json")
    selection = read(folder / "final-selection.json")
    history = read(folder / "history.json")
    config = read(folder / "config.json")
    if not (status["state"] == "complete" and status["epoch"] == 100 and
            len(history) == 100 and config["seed"] == seed and config["classes"] == 80 and
            selection["state"] == "frozen" and selection["count"] == 1600 and
            selection["source_hashes"]["src/ood/class_holdout_train.py"] == source_hash and
            selection["checkpoint_sha256"] == file_hash(folder / "final-best.pt") and
            selection["epoch"] == max(history, key=lambda row: row["val_accuracy"])["epoch"]):
        raise ValueError(f"seed {seed} completed training does not verify")
    return {"seed": seed, "epochs": 100, "selected_epoch": selection["epoch"],
            "checkpoint_accuracy": max(row["val_accuracy"] for row in history),
            "checkpoint_sha256": selection["checkpoint_sha256"]}


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("verified Windows CUDA device required")
    _, evidence = frozen_partition()
    status_path = RUN / "status.json"
    status = read(status_path)
    if status["state"] not in ("partition_audited_training_not_started",
                               "training_deferred_gpu_busy"):
        raise RuntimeError("training phase already active/completed/failed; inspect before retry")
    if (status["partition_sha256"] != evidence["partition_sha256"] or
            status["partition_audit_sha256"] != evidence["partition_audit_sha256"]):
        raise ValueError("status differs from audited partition")
    source_hash = file_hash(ROOT / "src/ood/class_holdout_train.py")
    controller_hash = file_hash(Path(__file__))
    launch_path = RUN / "training-launch.json"
    if launch_path.exists():
        launch = read(launch_path)
        if not (launch["trainer_source_sha256"] == source_hash and
                launch["controller_source_sha256"] == controller_hash and
                launch["partition_sha256"] == evidence["partition_sha256"] and
                launch["partition_audit_sha256"] == evidence["partition_audit_sha256"]):
            raise ValueError("frozen training launch differs")
    completed = []
    for seed in range(3):
        record = complete(seed, source_hash)
        if record is not None:
            completed.append(record)
            continue
        if file_hash(ROOT / "src/ood/class_holdout_train.py") != source_hash:
            raise ValueError("trainer source changed since launch")
        samples = gpu_samples()
        if not idle(samples):
            atomic_json(status_path, {"state": "training_deferred_gpu_busy",
                "trainer_running": False, "next_seed": seed,
                "completed_seeds": completed,
                "partition_sha256": evidence["partition_sha256"],
                "partition_audit_sha256": evidence["partition_audit_sha256"],
                "gpu_samples": samples, "updated_at": now()})
            print(json.dumps({"state": "training_deferred_gpu_busy",
                              "next_seed": seed, "samples": samples}), flush=True)
            return
        if not launch_path.exists():
            launch = {"state": "fixed_training_schedule_launched",
                      "launched_at": now(), "controller_pid": os.getpid(),
                      "trainer_source_sha256": source_hash,
                      "controller_source_sha256": controller_hash,
                      "protocol_sha256": file_hash(ROOT / "docs/CIFAR100_CLASS_HOLDOUT.md"),
                      "declaration_sha256": evidence["declaration_sha256"],
                      "partition_sha256": evidence["partition_sha256"],
                      "partition_audit_sha256": evidence["partition_audit_sha256"],
                      "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
                      "seeds": [0, 1, 2], "epochs_per_seed": 100,
                      "checkpoint_role": "ID_checkpoint_validation_1600",
                      "gpu_guard": {"samples": SAMPLES, "gap_seconds": GAP_SECONDS,
                          "max_used_mib": MAX_USED_MIB,
                          "max_util_percent": MAX_UTIL_PERCENT,
                          "max_memory_util_percent": MAX_MEMORY_UTIL_PERCENT,
                          "max_power_w": MAX_POWER_W},
                      "first_gpu_samples": samples}
            atomic_json(launch_path, launch)
        folder = RUN / f"seed{seed}"
        resume = folder.exists()
        atomic_json(status_path, {"state": "training_active", "trainer_running": True,
            "controller_pid": os.getpid(), "current_seed": seed, "resuming": resume,
            "completed_seeds": completed,
            "partition_sha256": evidence["partition_sha256"],
            "partition_audit_sha256": evidence["partition_audit_sha256"],
            "training_launch_sha256": file_hash(launch_path),
            "gpu_samples": samples, "updated_at": now()})
        command = [sys.executable, "-X", "utf8", "-m", "src.ood.class_holdout_train",
                   "--seed", str(seed)]
        if resume:
            command.append("--resume")
        with (RUN / f"seed{seed}.stdout.log").open("a", encoding="utf-8") as out, \
             (RUN / f"seed{seed}.stderr.log").open("a", encoding="utf-8") as err:
            out.write(json.dumps({"controller_started_at": now(), "resume": resume}) + "\n")
            out.flush()
            subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
        record = complete(seed, source_hash)
        if record is None:
            raise ValueError(f"seed {seed} exited without full 100-epoch selection")
        completed.append(record)
    atomic_json(status_path, {"state": "training_complete_pending_independent_audit",
        "trainer_running": False, "completed_seeds": completed,
        "partition_sha256": evidence["partition_sha256"],
        "partition_audit_sha256": evidence["partition_audit_sha256"],
        "training_launch_sha256": file_hash(launch_path), "updated_at": now()})
    print(json.dumps({"state": "training_complete_pending_independent_audit",
                      "seeds": [0, 1, 2]}), flush=True)


def main() -> None:
    lock_path = RUN / "training-controller.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run()
        except Exception as error:
            atomic_json(RUN / "status.json", {"state": "training_failed_preserve_partial",
                "trainer_running": False, "controller_pid": os.getpid(),
                "error": repr(error), "traceback": traceback.format_exc(),
                "updated_at": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
