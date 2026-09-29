"""Bind independently audited ID-only inference to one frozen checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
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
    split = BASE / f"split{args.class_seed}"
    folder = split / f"seed{args.seed}"
    destination = split / f"checkpoint-inference-audit-seed{args.seed}"
    require(not destination.exists(), "write-once consolidated ID inference audit exists")
    training_path = split / f"train-audit-seed{args.seed}/audit.json"
    cpu_folder = split / f"checkpoint-cpu-audit-seed{args.seed}"
    cpu_path = cpu_folder / "audit.json"
    training = read(training_path)
    cpu = read(cpu_path)
    selection = read(folder / "final-selection.json")
    checkpoint_hash = sha(folder / "final-best.pt")
    require(training["state"] == "passed" and
            training["class_seed"] == cpu["class_seed"] == args.class_seed and
            training["seed"] == cpu["seed"] == args.seed and
            training["training_epoch_count"] == 100 and
            selection["state"] == "frozen" and
            selection["count"] == cpu["id_checkpoint_validation_count"] == 1600 and
            training["selected_checkpoint_sha256"] ==
            cpu["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == checkpoint_hash and
            cpu["training_audit_sha256"] == sha(training_path) and
            cpu["raw_id_validation_sha256"] ==
            sha(cpu_folder / "checkpoint-validation.npz") and
            cpu["source_sha256"] ==
            sha(cpu_folder / "executed_audit_source.py"),
            "complete training/CPU ID-only inference sources")
    cuda_path = split / f"checkpoint-cuda-audit-seed{args.seed}/audit.json"
    if cpu["state"] == "passed":
        require(cpu["recomputed_accuracy"] ==
                training["selected_id_checkpoint_accuracy"],
                "exact CPU ID checkpoint reproduction")
        require(not cuda_path.exists(), "unexpected CUDA discrepancy audit")
        primary = cpu
        primary_path = cpu_folder / "checkpoint-validation.npz"
        cuda_hash = None
    else:
        require(cpu["state"] == "cpu_cuda_discrepancy_requires_cuda_audit",
                "explicit CPU/CUDA discrepancy state")
        cuda = read(cuda_path)
        primary_path = cuda_path.parent / "checkpoint-validation-cuda.npz"
        require(cuda["state"] == "passed" and
                cuda["class_seed"] == args.class_seed and cuda["seed"] == args.seed and
                cuda["selected_checkpoint_sha256"] == checkpoint_hash and
                cuda["training_audit_sha256"] == sha(training_path) and
                cuda["cpu_audit_sha256"] == sha(cpu_path) and
                cuda["raw_id_validation_sha256"] == sha(primary_path) and
                cuda["source_sha256"] ==
                sha(cuda_path.parent / "executed_audit_source.py") and
                cuda["recomputed_accuracy"] ==
                training["selected_id_checkpoint_accuracy"],
                "original CUDA path must reproduce selected ID checkpoint")
        primary = cuda
        cuda_hash = sha(cuda_path)
    destination.mkdir()
    snapshot = destination / "executed_audit_source.py"
    shutil.copyfile(__file__, snapshot)
    record = {
        "state": "passed", "class_seed": args.class_seed, "seed": args.seed,
        "selected_epoch": selection["epoch"],
        "id_checkpoint_validation_count": 1600,
        "recomputed_correct_count": primary["recomputed_correct_count"],
        "recomputed_accuracy": primary["recomputed_accuracy"],
        "selected_checkpoint_sha256": checkpoint_hash,
        "training_audit_sha256": sha(training_path),
        "cpu_audit_sha256": sha(cpu_path),
        "cuda_audit_sha256": cuda_hash,
        "raw_id_validation_sha256": sha(primary_path),
        "primary_inference_path": "CUDA" if cuda_hash else "CPU",
        "source_sha256": sha(snapshot),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Consolidated independent ID checkpoint inference only; no official test/OOD data read.",
    }
    (destination / "audit.json").write_text(json.dumps(record, indent=2,
                                                       allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"state": "passed", "class_seed": args.class_seed,
                      "seed": args.seed,
                      "primary_inference_path": record["primary_inference_path"]}),
          flush=True)


if __name__ == "__main__":
    main()
