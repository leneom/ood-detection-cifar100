"""One locked GPU controller for six fixed full-budget replication models."""
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
from tools.run_class_holdout_training import (gpu_samples, idle, SAMPLES, GAP_SECONDS,
    MAX_USED_MIB, MAX_UTIL_PERCENT, MAX_MEMORY_UTIL_PERCENT, MAX_POWER_W)
from tools.train_class_holdout_replication import BASE, SOURCES, frozen_partition


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(class_seed: int, model_seed: int, evidence: dict,
             source_hashes: dict) -> dict | None:
    folder = BASE / f"split{class_seed}/seed{model_seed}"
    if not folder.exists():
        return None
    if not (folder / "final-selection.json").exists():
        if (folder / "config.json").exists() and (folder / "last.pt").exists():
            return None
        raise ValueError(f"partial folder requires review: {folder}")
    status = read(folder / "status.json")
    selection = read(folder / "final-selection.json")
    history = read(folder / "history.json")
    config = read(folder / "config.json")
    selected = max(history, key=lambda row: row["val_accuracy"])
    if not (status["state"] == "complete" and status["epoch"] == 100 and
            len(history) == 100 and
            [row["epoch"] for row in history] == list(range(1, 101)) and
            config["seed"] == model_seed and config["classes"] == 80 and
            config["epochs"] == 100 and config["class_partition"] == evidence and
            selection["state"] == "frozen" and selection["count"] == 1600 and
            selection["source_hashes"] == source_hashes and
            read(folder / "source_hashes.json") == source_hashes and
            selection["checkpoint_sha256"] == file_hash(folder / "final-best.pt") and
            selection["epoch"] == selected["epoch"] and
            selected["val_accuracy"] == status["best_val_accuracy"]):
        raise ValueError(f"frozen training evidence differs: {class_seed}/{model_seed}")
    return {"class_seed": class_seed, "training_seed": model_seed,
            "epochs": 100, "selected_epoch": selection["epoch"],
            "checkpoint_accuracy": selected["val_accuracy"],
            "checkpoint_sha256": selection["checkpoint_sha256"]}


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("verified Windows CUDA device required")
    require_python = ROOT / "data/cifar-100-python/train"
    if not require_python.exists():
        raise FileNotFoundError(require_python)
    evidence = {class_seed: frozen_partition(class_seed)[1]
                for class_seed in (2032, 2033)}
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    controller_hash = file_hash(Path(__file__))
    old_controller_hash = file_hash(ROOT / "tools/run_class_holdout_training.py")
    launch_path = BASE / "training-launch.json"
    if launch_path.exists():
        launch = read(launch_path)
        canonical_evidence = {str(key): value for key, value in evidence.items()}
        if not (launch["base_controller_sha256"] == old_controller_hash and
                launch["source_hashes"] == source_hashes and
                launch["partition_evidence"] == canonical_evidence and
                launch["python"] == sys.executable):
            raise ValueError("immutable training launch/source differs")
        if launch["controller_sha256"] != controller_hash:
            # The initial controller serialized integer class-seed dict keys
            # to JSON strings, then compared them without normalizing on resume.
            # Preserve the launch and require a hashed controller-only amendment.
            amendment = read(BASE / "controller-amendment.json")
            if not (amendment["state"] == "controller_resume_key_fix" and
                    amendment["immutable_launch_sha256"] == file_hash(launch_path) and
                    amendment["original_controller_sha256"] ==
                    launch["controller_sha256"] ==
                    file_hash(BASE / "controller-original-source.py") and
                    amendment["revised_controller_sha256"] == controller_hash and
                    amendment["failure_status_snapshot_sha256"] ==
                    file_hash(BASE / "controller-resume1-failure.json") and
                    amendment["source_hashes"] == source_hashes and
                    amendment["partition_evidence"] == canonical_evidence):
                raise ValueError("controller-only resume amendment differs")
    completed = []
    for class_seed in (2032, 2033):
        for model_seed in (0, 1, 2):
            record = complete(class_seed, model_seed, evidence[class_seed], source_hashes)
            if record is not None:
                completed.append(record)
                continue
            if ({name: file_hash(ROOT / name) for name in SOURCES} != source_hashes or
                    file_hash(Path(__file__)) != controller_hash):
                raise ValueError("frozen training source changed since launch")
            fresh_evidence = frozen_partition(class_seed)[1]
            if fresh_evidence != evidence[class_seed]:
                raise ValueError("audited partition changed since launch")
            samples = gpu_samples()
            if not idle(samples):
                atomic_json(BASE / "status.json", {
                    "state": "training_deferred_gpu_busy", "trainer_running": False,
                    "next_class_seed": class_seed, "next_training_seed": model_seed,
                    "completed": completed, "partition_evidence": evidence,
                    "gpu_samples": samples, "updated_at": now()})
                print(json.dumps({"state": "training_deferred_gpu_busy",
                                  "class_seed": class_seed,
                                  "training_seed": model_seed}), flush=True)
                return
            if not launch_path.exists():
                atomic_json(launch_path, {
                    "state": "fixed_six_model_schedule_launched", "launched_at": now(),
                    "controller_pid": os.getpid(), "controller_sha256": controller_hash,
                    "base_controller_sha256": old_controller_hash,
                    "source_hashes": source_hashes, "partition_evidence": evidence,
                    "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
                    "class_seeds": [2032, 2033], "training_seeds": [0, 1, 2],
                    "epochs_per_model": 100,
                    "checkpoint_population": "ID_checkpoint_validation_1600",
                    "gpu_guard": {"samples": SAMPLES, "gap_seconds": GAP_SECONDS,
                                  "max_used_mib": MAX_USED_MIB,
                                  "max_util_percent": MAX_UTIL_PERCENT,
                                  "max_memory_util_percent": MAX_MEMORY_UTIL_PERCENT,
                                  "max_power_w": MAX_POWER_W},
                    "first_gpu_samples": samples})
            folder = BASE / f"split{class_seed}/seed{model_seed}"
            resume = folder.exists()
            atomic_json(BASE / "status.json", {
                "state": "training_active", "trainer_running": True,
                "controller_pid": os.getpid(), "class_seed": class_seed,
                "training_seed": model_seed, "resuming": resume,
                "completed": completed, "partition_evidence": evidence,
                "training_launch_sha256": file_hash(launch_path),
                "gpu_samples": samples, "updated_at": now()})
            command = [sys.executable, "-X", "utf8",
                       str(ROOT / "tools/train_class_holdout_replication.py"),
                       "--class-seed", str(class_seed), "--seed", str(model_seed)]
            if resume:
                command.append("--resume")
            stdout = BASE / f"split{class_seed}/seed{model_seed}.stdout.log"
            stderr = BASE / f"split{class_seed}/seed{model_seed}.stderr.log"
            with stdout.open("a", encoding="utf-8") as out, \
                 stderr.open("a", encoding="utf-8") as err:
                out.write(json.dumps({"controller_started_at": now(),
                                      "resume": resume}) + "\n")
                out.flush()
                subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
            record = complete(class_seed, model_seed, evidence[class_seed], source_hashes)
            if record is None:
                raise ValueError(f"model exited before 100 epochs: {class_seed}/{model_seed}")
            completed.append(record)
    atomic_json(BASE / "status.json", {
        "state": "training_complete_pending_independent_audit",
        "trainer_running": False, "completed": completed,
        "partition_evidence": evidence,
        "training_launch_sha256": file_hash(launch_path), "updated_at": now()})
    print(json.dumps({"state": "training_complete_pending_independent_audit",
                      "completed_models": len(completed)}), flush=True)


def main() -> None:
    lock_path = BASE / "training-controller.lock"
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
            atomic_json(BASE / "status.json", {
                "state": "training_failed_preserve_partial", "trainer_running": False,
                "controller_pid": os.getpid(), "error": repr(error),
                "traceback": traceback.format_exc(), "updated_at": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
