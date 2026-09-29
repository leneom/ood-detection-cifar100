"""CPU-only source and ID-selection audit of a full 80-class training seed."""
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
RUN = ROOT / "runs" / "cifar100-class-holdout"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    if torch.cuda.is_available():
        raise RuntimeError("hide CUDA for independent training audit")
    folder = RUN / f"seed{seed}"
    destination = RUN / f"train-audit-seed{seed}"
    require(not destination.exists(), "training audit already exists")
    config, split, status, selection, history = [read(folder / name) for name in
        ("config.json", "split.json", "status.json", "final-selection.json", "history.json")]
    launch = read(RUN / "training-launch.json")
    partition = RUN / "partition.npz"
    audit = RUN / "partition-audit.json"
    require(status["state"] == "complete" and status["epoch"] == 100 and
            status["checkpoint_validation_count"] == 1600 and
            len(history) == 100 and [row["epoch"] for row in history] == list(range(1, 101)),
            "complete 100-epoch trace required")
    require(config["dataset"] == "cifar100_class_holdout" and
            config["model"] == "resnet18" and config["classes"] == 80 and
            config["seed"] == seed and config["epochs"] == 100 and
            config["batch_size"] == 128 and config["lr"] == .1 and
            config["momentum"] == .9 and config["weight_decay"] == 5e-4 and
            config["milestones"] == [50, 75, 90] and config["amp"] is True and
            config["deterministic"] is True and config["device"] == "cuda" and
            len(config["id_fine_labels"]) == 80 and len(config["ood_fine_labels"]) == 20,
            "fixed 80-class model or 100-epoch schedule differs")
    require(split["test_used_during_training"] is False and
            split["held_out_training_images_used"] is False and
            [split[key] for key in ("train_count", "validation_count", "checkpoint_count",
                                    "temperature_count", "threshold_count")] ==
            [36000, 4000, 1600, 1200, 1200] and
            split["partition_sha256"] == sha(partition) == launch["partition_sha256"] and
            launch["partition_audit_sha256"] == sha(audit) and
            read(audit)["state"] == "passed",
            "partition and role source differs")
    with np.load(partition, allow_pickle=False) as stored, \
         np.load(folder / "split_indices.npz", allow_pickle=False) as saved:
        require(set(saved.files) == {"train", "validation", "checkpoint"} and
                np.array_equal(saved["train"], stored["train_indices"]) and
                np.array_equal(saved["validation"], stored["validation_indices"]) and
                np.array_equal(saved["checkpoint"], stored["checkpoint_indices"]) and
                np.array_equal(config["id_fine_labels"], stored["id_fine_labels"]) and
                np.array_equal(config["ood_fine_labels"], stored["ood_fine_labels"]),
                "run original rows/class map differ from audited partition")
    first = max(history, key=lambda row: row["val_accuracy"])
    require(first["epoch"] == selection["epoch"] and
            first["val_accuracy"] == status["best_val_accuracy"] and
            selection["state"] == "frozen" and
            selection["selection_population"] == "ID_checkpoint_validation" and
            selection["count"] == 1600 and
            selection["checkpoint_sha256"] == sha(folder / "final-best.pt") ==
            sha(folder / "best.pt"),
            "selected checkpoint is not first maximum on checkpoint-only ID")
    for row in history:
        expected_lr = .1 * (.1 ** sum(row["epoch"] > step for step in (50, 75, 90)))
        require(np.isclose(row["lr"], expected_lr, atol=1e-12, rtol=1e-12) and
                np.isfinite(row["train_loss"]) and
                0 <= row["train_accuracy"] <= 1 and
                0 <= row["val_accuracy"] <= 1,
                f"epoch {row['epoch']} trace or LR differs")
    sources = read(folder / "source_hashes.json")
    require(sources["src/ood/class_holdout_train.py"] ==
            launch["trainer_source_sha256"] == sha(ROOT / "src/ood/class_holdout_train.py") and
            config["class_partition"]["partition_audit_sha256"] == sha(audit),
            "training source/audit launch binding differs")
    for name, digest in sources.items():
        require(sha(ROOT / name) == sha(folder / "source_snapshot" / name) == digest,
                "frozen training source snapshot differs: " + name)
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
            "full resume state or frozen 80-class selected payload differs")
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    record = {"state": "passed", "seed": seed,
              "training_epoch_count": 100,
              "validated_model_tensor_count": len(final["model"]),
              "selected_epoch": first["epoch"],
              "selected_id_checkpoint_accuracy": first["val_accuracy"],
              "selected_checkpoint_sha256": selection["checkpoint_sha256"],
              "partition_sha256": sha(partition),
              "partition_audit_sha256": sha(audit),
              "trainer_source_sha256": sources["src/ood/class_holdout_train.py"],
              "training_launch_sha256": sha(RUN / "training-launch.json"),
              "audit_source_sha256": sha(Path(__file__)),
              "audited_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "100-epoch source/split/resume/ID-checkpoint audit only; no ID-test or OOD score used."}
    (destination / "audit.json").write_text(json.dumps(record, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "selected_epoch": first["epoch"]}), flush=True)


if __name__ == "__main__":
    main()
