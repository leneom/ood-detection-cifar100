"""Independent CPU-only schedule, ID-selection and resume-state audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


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


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    require(not torch.cuda.is_available(), "hide CUDA for independent training audit")
    split_folder = BASE / f"split{args.class_seed}"
    folder = split_folder / f"seed{args.seed}"
    destination = split_folder / f"train-audit-seed{args.seed}"
    require(not destination.exists(), "training audit already exists")
    config, split, status, selection, history = [read(folder / name) for name in
        ("config.json", "split.json", "status.json", "final-selection.json", "history.json")]
    launch = read(BASE / "training-launch.json")
    audit = read(split_folder / "partition-audit.json")
    partition = split_folder / "partition.npz"
    evidence = launch["partition_evidence"][str(args.class_seed)]
    amendment_path = BASE / "controller-amendment.json"
    if amendment_path.exists():
        amendment = read(amendment_path)
        controller_bound = (
            amendment["state"] == "controller_resume_key_fix" and
            amendment["immutable_launch_sha256"] == sha(BASE / "training-launch.json") and
            amendment["original_controller_sha256"] == launch["controller_sha256"] ==
            sha(BASE / "controller-original-source.py") and
            amendment["revised_controller_sha256"] ==
            sha(ROOT / "tools/run_class_holdout_replication_training.py") and
            amendment["failure_status_snapshot_sha256"] ==
            sha(BASE / "controller-resume1-failure.json") and
            amendment["source_hashes"] == launch["source_hashes"] and
            amendment["partition_evidence"] == launch["partition_evidence"])
    else:
        controller_bound = (
            launch["controller_sha256"] ==
            sha(ROOT / "tools/run_class_holdout_replication_training.py"))
    require(launch["state"] == "fixed_six_model_schedule_launched" and
            launch["class_seeds"] == [2032, 2033] and
            launch["training_seeds"] == [0, 1, 2] and
            launch["epochs_per_model"] == 100 and
            launch["checkpoint_population"] == "ID_checkpoint_validation_1600" and
            controller_bound and
            launch["base_controller_sha256"] == sha(ROOT / "tools/run_class_holdout_training.py") and
            audit["state"] == "passed" and
            evidence["partition_sha256"] == split["partition_sha256"] == sha(partition) and
            evidence["partition_audit_sha256"] == sha(split_folder / "partition-audit.json") and
            config["class_partition"] == evidence,
            "launch/partition/independent audit binding")
    require(status["state"] == "complete" and status["epoch"] == 100 and
            status["checkpoint_validation_count"] == 1600 and
            len(history) == 100 and
            [row["epoch"] for row in history] == list(range(1, 101)),
            "full fixed 100-epoch schedule")
    require(config["dataset"] == "cifar100_class_holdout" and
            config["model"] == "resnet18" and config["classes"] == 80 and
            config["seed"] == args.seed and config["epochs"] == 100 and
            config["batch_size"] == 128 and config["lr"] == .1 and
            config["momentum"] == .9 and config["weight_decay"] == 5e-4 and
            config["milestones"] == [50, 75, 90] and config["amp"] is True and
            config["deterministic"] is True and config["device"] == "cuda" and
            len(config["id_fine_labels"]) == 80 and
            len(config["ood_fine_labels"]) == 20,
            "fixed trainer configuration")
    require(split["test_used_during_training"] is False and
            split["held_out_training_images_used"] is False and
            [split[key] for key in ("train_count", "validation_count",
                                    "checkpoint_count", "temperature_count",
                                    "threshold_count")] ==
            [36000, 4000, 1600, 1200, 1200], "train/ID role isolation")
    with np.load(partition, allow_pickle=False) as stored, \
         np.load(folder / "split_indices.npz", allow_pickle=False) as saved:
        require(set(saved.files) == {"train", "validation", "checkpoint"} and
                np.array_equal(saved["train"], stored["train_indices"]) and
                np.array_equal(saved["validation"], stored["validation_indices"]) and
                np.array_equal(saved["checkpoint"], stored["checkpoint_indices"]) and
                np.array_equal(config["id_fine_labels"], stored["id_fine_labels"]) and
                np.array_equal(config["ood_fine_labels"], stored["ood_fine_labels"]),
                "training original rows/class map")
    first = max(history, key=lambda row: row["val_accuracy"])
    require(first["epoch"] == selection["epoch"] and
            first["val_accuracy"] == status["best_val_accuracy"] and
            selection["state"] == "frozen" and
            selection["selection_population"] == "ID_checkpoint_validation" and
            selection["count"] == 1600 and
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") ==
            sha(folder / "best.pt"), "first ID-validation maximum selection")
    for row in history:
        expected_lr = .1 * (.1 ** sum(row["epoch"] > step for step in (50, 75, 90)))
        require(np.isclose(row["lr"], expected_lr, atol=1e-12, rtol=1e-12) and
                np.isfinite(row["train_loss"]) and
                0 <= row["train_accuracy"] <= 1 and
                0 <= row["val_accuracy"] <= 1,
                f"epoch {row['epoch']} LR or trace")
    sources = read(folder / "source_hashes.json")
    require(sources == selection["source_hashes"] == launch["source_hashes"],
            "launch/trainer source hashes")
    for name, digest in sources.items():
        require(sha(ROOT / name) == sha(folder / "source_snapshot" / name) == digest,
                "source snapshot differs: " + name)
    final = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
    selected = torch.load(folder / "final-best.pt", map_location="cpu", weights_only=True)
    require(final["epoch"] == 100 and final["history"] == history and
            final["best_accuracy"] == first["val_accuracy"] and
            final["config"] == selected["config"] == config and
            final["split"] == selected["split"] == split and
            final["source_hashes"] == selected["source_hashes"] == sources and
            selected["epoch"] == first["epoch"] and
            selected["val_accuracy"] == first["val_accuracy"] and
            selected["checkpoint_role"] == "ID_checkpoint_validation" and
            selected["checkpoint_count"] == 1600 and
            final["model"]["fc.weight"].shape == (80, 512) and
            final["model"]["fc.bias"].shape == (80,) and
            final["torch_rng"].numel() > 0 and len(final["cuda_rng"]) == 1,
            "full resume state or selected 80-class checkpoint payload")
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    record = {
        "state": "passed", "class_seed": args.class_seed,
        "seed": args.seed, "training_epoch_count": 100,
        "validated_model_tensor_count": len(final["model"]),
        "selected_epoch": first["epoch"],
        "selected_id_checkpoint_accuracy": first["val_accuracy"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "partition_sha256": sha(partition),
        "partition_audit_sha256": sha(split_folder / "partition-audit.json"),
        "trainer_source_sha256": sources["src/ood/class_holdout_train.py"],
        "adapter_source_sha256": sources["tools/train_class_holdout_replication.py"],
        "training_launch_sha256": sha(BASE / "training-launch.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Full schedule/source/resume/ID-checkpoint audit; no official ID-test or OOD test read.",
    }
    (destination / "audit.json").write_text(json.dumps(record, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"state": "passed", "class_seed": args.class_seed,
                      "seed": args.seed, "selected_epoch": first["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
