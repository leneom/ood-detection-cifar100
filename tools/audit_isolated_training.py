"""CPU-only, source-bound audit of a completed isolated classifier run."""
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
PHASE = ROOT / "runs" / "validation-isolation"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(seed):
    old = ROOT / "runs" / f"cifar100-seed{seed}"
    new = PHASE / f"seed{seed}"
    status, config, selection = (read(new / name) for name in
                                 ("status.json", "config.json", "final-selection.json"))
    old_history, history = read(old / "history.json"), read(new / "history.json")
    require(status["state"] == "complete" and status["epoch"] == 100 and len(history) == 100,
            "complete 100-epoch schedule required")
    require([row["epoch"] for row in history] == list(range(1, 101)), "epoch order changed")
    require(config["seed"] == seed and config["epochs"] == 100 and config["device"] == "cuda" and
            config["amp"] and not config["no_deterministic"], "fixed run configuration differs")
    require(len(old_history) == 100, "original reference schedule incomplete")
    fields = ("lr", "train_loss", "train_accuracy")
    for epoch, (prior, current) in enumerate(zip(old_history, history), 1):
        require(all(prior[key] == current[key] for key in fields),
                f"training trajectory statistic differs at epoch {epoch}")
    for epoch in (50, 51, 75, 76, 90, 91):
        require(history[epoch - 1]["lr"] == old_history[epoch - 1]["lr"],
                f"fixed learning-rate milestone differs at epoch {epoch}")
    first = max(history, key=lambda row: row["val_accuracy"])
    require(first["epoch"] == selection["epoch"] and
            first["val_accuracy"] == status["best_val_accuracy"],
            "checkpoint is not the first ID-checkpoint-validation accuracy maximum")
    require(selection["count"] == 2000 and selection["selection_population"] ==
            "ID_checkpoint_validation", "selection population changed")
    require(selection["checkpoint_sha256"] == digest(new / "final-best.pt") ==
            digest(new / "best.pt"), "immutable selected checkpoint differs")
    prior_indices, indices = arrays(old / "split_indices.npz"), arrays(new / "split_indices.npz")
    partition = arrays(PHASE / "partition.npz")
    require(np.array_equal(indices["train"], prior_indices["train"]) and
            np.array_equal(indices["val"], prior_indices["val"]) and
            np.array_equal(indices["checkpoint"], partition["checkpoint_original_indices"]),
            "train/validation/selection indices differ")
    require(len(indices["train"]) == 45000 and len(indices["checkpoint"]) == 2000 and
            not np.intersect1d(indices["checkpoint"], partition["temperature_original_indices"]).size and
            not np.intersect1d(indices["checkpoint"], partition["threshold_original_indices"]).size,
            "ID validation roles overlap or training count differs")
    sources = read(new / "source_hashes.json")
    launch = read(PHASE / "training-launch.json")
    require(sources["isolated_train.py"] == launch["trainer_source_sha256"] and
            config["validation_partition"]["partition_sha256"] == launch["partition_sha256"] ==
            digest(PHASE / "partition.npz"), "source/partition launch binding changed")
    for name, expected in sources.items():
        require(digest(new / "source_snapshot" / name) == expected, "training source snapshot differs")
    original = torch.load(old / "last.pt", map_location="cpu", weights_only=False)
    final = torch.load(new / "last.pt", map_location="cpu", weights_only=False)
    selected = torch.load(new / "final-best.pt", map_location="cpu", weights_only=True)
    require(original["epoch"] == final["epoch"] == 100 and selected["epoch"] == first["epoch"],
            "full or selected checkpoint epoch differs")
    require(selected["val_accuracy"] == first["val_accuracy"] and
            selected["checkpoint_role"] == "checkpoint" and
            selected["checkpoint_count"] == 2000, "selected checkpoint metadata differs")
    require(original["model"].keys() == final["model"].keys(), "model parameter names differ")
    for name in original["model"]:
        require(torch.equal(original["model"][name], final["model"][name]),
                "final model tensor differs: " + name)
    require(torch.equal(original["torch_rng"], final["torch_rng"]) and
            len(original["cuda_rng"]) == len(final["cuda_rng"]) and
            all(torch.equal(x, y) for x, y in zip(original["cuda_rng"], final["cuda_rng"])),
            "final RNG trajectory differs")
    return {"state": "passed", "seed": seed, "training_epoch_count": 100,
            "exact_training_trace_fields": list(fields),
            "exact_final_model_tensor_count": len(final["model"]),
            "torch_and_cuda_rng_exact": True,
            "new_selection_role": "checkpoint_ID_2000",
            "new_selected_epoch": first["epoch"],
            "new_selected_validation_accuracy": first["val_accuracy"],
            "old_selected_epoch": max(old_history, key=lambda row: row["val_accuracy"])["epoch"],
            "selected_checkpoint_sha256": selection["checkpoint_sha256"],
            "training_source_sha256": sources["isolated_train.py"],
            "partition_sha256": digest(PHASE / "partition.npz"),
            "boundary": "This audits training and checkpoint selection only, not new ID-test/OOD results."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = p.parse_args().seed
    if torch.cuda.is_available():
        raise RuntimeError("mask CUDA for this CPU-only audit")
    output = PHASE / f"train-audit-seed{seed}"
    if output.exists():
        raise FileExistsError("training audit output exists; inspect before another attempt")
    result = audit(seed)
    output.mkdir()
    shutil.copyfile(__file__, output / "executed_audit_source.py")
    result.update({"audit_source_sha256": digest(__file__),
                   "audited_at": datetime.now(timezone.utc).isoformat()})
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed,
                      "selected_epoch": result["new_selected_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
