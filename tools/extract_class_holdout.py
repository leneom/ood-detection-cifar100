"""Extract frozen 80-class CIFAR-100 holdout logits and ID-fitted feature scores.

Run only after all three 100-epoch training schedules and each independent
training audit complete. This stage fits no temperature or threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ood.class_holdout_train import frozen_partition
from src.ood.data import load_cifar, index_hash
from src.ood.evaluate import image_fingerprint
from src.ood.features import extract, KNN, Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import atomic_json, environment, file_hash, seed_all
from src.ood.scores import METHODS as LOGIT_METHODS, score_logits


RUN = ROOT / "runs" / "cifar100-class-holdout"
SOURCE = (
    "tools/extract_class_holdout.py", "src/ood/class_holdout_train.py",
    "src/ood/data.py", "src/ood/evaluate.py", "src/ood/features.py",
    "src/ood/models.py", "src/ood/runtime.py", "src/ood/scores.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def gpu_idle_samples() -> list[dict]:
    """Three two-second-spaced checks before creating any extraction output."""
    samples = []
    for attempt in range(3):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        require(len(rows) == 1, "exactly one verified GPU required")
        used, utilization, memory_utilization, power, total = (
            part.strip() for part in rows[0].split(","))
        samples.append({"used_mib": int(used),
                        "util_percent": int(utilization),
                        "memory_util_percent": int(memory_utilization),
                        "power_w": float(power), "total_mib": int(total)})
        if attempt < 2:
            time.sleep(2)
    return samples


def gpu_is_idle(samples: list[dict]) -> bool:
    return len(samples) == 3 and all(
        row["used_mib"] <= 4096 and row["util_percent"] <= 55 and
        row["memory_util_percent"] <= 10 and row["power_w"] <= 55
        for row in samples)


def preflight(seed: int) -> tuple[dict, dict, dict, dict, Path]:
    phase = read(RUN / "status.json")
    require(phase["state"] == "training_complete_pending_independent_audit" and
            phase["trainer_running"] is False and
            [row["seed"] for row in phase["completed_seeds"]] == [0, 1, 2],
            "all three fixed training schedules must finish before extraction")
    for planned_seed in (0, 1, 2):
        train_audit_path = RUN / f"train-audit-seed{planned_seed}" / "audit.json"
        inference_dir = RUN / f"checkpoint-inference-audit-seed{planned_seed}"
        train_audit = read(train_audit_path)
        inference_audit = read(inference_dir / "audit.json")
        checkpoint_sha = file_hash(RUN / f"seed{planned_seed}" / "final-best.pt")
        require(train_audit["state"] == inference_audit["state"] == "passed" and
                train_audit["seed"] == inference_audit["seed"] == planned_seed and
                train_audit["training_epoch_count"] == 100 and
                train_audit["selected_checkpoint_sha256"] ==
                inference_audit["selected_checkpoint_sha256"] == checkpoint_sha and
                inference_audit["training_audit_sha256"] == file_hash(train_audit_path) and
                inference_audit["raw_id_validation_sha256"] ==
                file_hash(inference_dir / "checkpoint-validation.npz") and
                inference_audit["source_sha256"] == file_hash(ROOT / (
                    "tools/audit_class_holdout_checkpoint_cross_device.py"
                    if planned_seed == 2 else
                    "tools/audit_class_holdout_checkpoint_inference.py")) and
                (planned_seed != 2 or (
                    inference_audit["cpu_cuda_prediction_disagreements"] == 1 and
                    inference_audit["cpu_recomputed_correct_count"] ==
                    inference_audit["recomputed_correct_count"] + 1 and
                    inference_audit["cuda_raw_id_validation_sha256"] ==
                    file_hash(inference_dir / "checkpoint-validation-cuda.npz") and
                    len(inference_audit["gpu_guard_samples"]) == 3 and
                    all(sample["used_mib"] <= 4096 and
                        sample["util_percent"] <= 55 and
                        sample["memory_util_percent"] <= 10 and
                        sample["power_w"] <= 55
                        for sample in inference_audit["gpu_guard_samples"]))),
                f"seed {planned_seed} lacks both independent full-training and ID-inference audits")
    partition, partition_evidence = frozen_partition()
    test_audit = read(RUN / "test-population-audit.json")
    test_manifest = read(RUN / "test-population.json")
    require(test_audit["state"] == "passed" and
            test_audit["test_population_manifest_sha256"] ==
            file_hash(RUN / "test-population.json") and
            test_audit["test_population_sha256"] ==
            test_manifest["test_population_sha256"] ==
            file_hash(RUN / "test-population.npz") and
            test_audit["partition_sha256"] == partition_evidence["partition_sha256"] and
            test_audit["official_test_file_sha256"] ==
            file_hash(ROOT / "data/cifar-100-python/test"),
            "independent 8000/2000 official test population audit missing or changed")
    folder = RUN / f"seed{seed}"
    audit = read(RUN / f"train-audit-seed{seed}" / "audit.json")
    config, selection = read(folder / "config.json"), read(folder / "final-selection.json")
    checkpoint = folder / "final-best.pt"
    require(audit["state"] == "passed" and audit["seed"] == seed and
            audit["training_epoch_count"] == 100 and
            audit["selected_checkpoint_sha256"] ==
            selection["checkpoint_sha256"] == file_hash(checkpoint) and
            audit["training_launch_sha256"] == file_hash(RUN / "training-launch.json") and
            audit["partition_sha256"] == partition_evidence["partition_sha256"] and
            audit["audit_source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_training.py") and
            config["classes"] == 80 and config["seed"] == seed and
            config["id_fine_labels"] == partition["id_fine_labels"].tolist() and
            config["ood_fine_labels"] == partition["ood_fine_labels"].tolist() and
            selection["count"] == 1600 and
            selection["selection_population"] == "ID_checkpoint_validation",
            "independent full-seed ID checkpoint audit missing or changed")
    inference_dir = RUN / f"checkpoint-inference-audit-seed{seed}"
    inference = read(inference_dir / "audit.json")
    require(inference["state"] == "passed" and inference["seed"] == seed and
            inference["selected_epoch"] == selection["epoch"] and
            inference["id_checkpoint_validation_count"] == 1600 and
            inference["recomputed_accuracy"] ==
            audit["selected_id_checkpoint_accuracy"] and
            inference["selected_checkpoint_sha256"] == file_hash(checkpoint) and
            inference["training_audit_sha256"] ==
            file_hash(RUN / f"train-audit-seed{seed}" / "audit.json") and
            inference["raw_id_validation_sha256"] ==
            file_hash(inference_dir / "checkpoint-validation.npz") and
            inference["source_sha256"] == file_hash(ROOT / (
                "tools/audit_class_holdout_checkpoint_cross_device.py"
                if seed == 2 else
                "tools/audit_class_holdout_checkpoint_inference.py")),
            "independent ID-checkpoint inference missing or changed")
    return config, selection, partition, test_manifest, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    require(args.batch_size > 0, "positive batch size required")
    config, selection, partition, test_manifest, checkpoint = preflight(args.seed)
    output = RUN / f"extract-seed{args.seed}"
    require(not output.exists(), "extraction output exists; preserve and inspect it")
    samples = gpu_idle_samples()
    require(gpu_is_idle(samples), f"GPU contention guard deferred extraction: {samples}")
    require(torch.cuda.is_available(), "verified Windows CUDA required")
    sources = {name: file_hash(ROOT / name) for name in SOURCE}
    seed_all(0)
    torch.set_grad_enabled(False)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(selected["epoch"] == selection["epoch"] and
            selected["config"] == config and
            selected["checkpoint_count"] == 1600,
            "checkpoint payload differs from frozen ID selection")
    model = build_model("resnet18", 80).to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(selected["model"])
    model.eval()
    train_images, train_fine = load_cifar(ROOT / "data", "cifar100", True, False)
    test_images, test_fine = load_cifar(ROOT / "data", "cifar100", False, False)
    require(len(train_images) == 50000 and len(test_images) == 10000 and
            np.array_equal(train_fine[partition["train_indices"]].numpy(),
                           partition["train_original_fine_labels"]),
            "training input rows or original labels differ")
    with np.load(RUN / "test-population.npz", allow_pickle=False) as stored:
        test = {name: stored[name] for name in stored.files}
    require(np.array_equal(test_fine[test["id_test_indices"]].numpy(),
                           test["id_original_fine_labels"]) and
            np.array_equal(test_fine[test["ood_test_indices"]].numpy(),
                           test["ood_original_fine_labels"]) and
            len(test["id_test_indices"]) == test_manifest["id_count"] == 8000 and
            len(test["ood_test_indices"]) == test_manifest["ood_count"] == 2000,
            "official test loader differs from audited original rows")
    id_fine = partition["id_fine_labels"]
    lookup = np.full(100, -1, dtype=np.int64)
    lookup[id_fine] = np.arange(80)
    require(np.array_equal(lookup[test["id_original_fine_labels"]],
                           test["id_class_targets"]) and
            np.all(lookup[test["ood_original_fine_labels"]] == -1),
            "test class mapping or held-out isolation differs")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCE:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    atomic_json(output / "status.json", {"state": "extracting", "seed": args.seed,
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "started_at": datetime.now(timezone.utc).isoformat()})
    outputs: dict[str, str] = {}
    inputs: dict[str, dict] = {}

    def saved(path: Path, **data: np.ndarray) -> None:
        save_arrays(path, **data)
        outputs[path.name] = file_hash(path)

    def run_population(name: str, images: torch.Tensor, original_indices: np.ndarray,
                       original_fine: np.ndarray, original_coarse: np.ndarray,
                       targets: np.ndarray | None) -> tuple[torch.Tensor, torch.Tensor]:
        require(len(images) == len(original_indices) == len(original_fine) ==
                len(original_coarse) and (targets is None or len(targets) == len(images)),
                "population array lengths differ")
        inputs[name] = image_fingerprint(images)
        logits, features = extract(model, images, args.batch_size, "cuda")
        require(logits.shape == (len(images), 80) and
                features.shape == (len(images), 512),
                "80-logit or 512-feature extraction shape differs")
        payload = {"logits": logits.numpy(), "features": features.numpy(),
                   "predictions": logits.argmax(1).numpy(),
                   "original_indices": original_indices,
                   "original_fine_labels": original_fine,
                   "original_coarse_labels": original_coarse}
        if targets is not None:
            payload["id_class_targets"] = targets
        saved(output / f"{name}_raw.npz", **payload)
        saved(output / f"{name}_T1_scores.npz",
              **{method: score_logits(logits, method, 1.).numpy()
                 for method in LOGIT_METHODS})
        print(json.dumps({"seed": args.seed, "population": name,
                          "count": len(images)}), flush=True)
        return logits, features

    train_rows = partition["train_indices"]
    train_labels = partition["train_original_fine_labels"]
    _, train_features = run_population(
        "id_train", train_images[train_rows], train_rows, train_labels,
        partition["fine_to_coarse"][train_labels], lookup[train_labels])
    knn = KNN(k=50, device="cuda").fit(train_features)
    maha = Mahalanobis(shrinkage=.01, device="cuda").fit(
        train_features, torch.from_numpy(lookup[train_labels]))

    def feature_scores(name: str, features: torch.Tensor) -> None:
        saved(output / f"{name}_feature_scores.npz",
              knn=knn.score(features).numpy(),
              mahalanobis=maha.score(features).numpy())

    for name in ("checkpoint", "temperature", "threshold"):
        rows = partition[f"{name}_indices"]
        fine = train_fine[rows].numpy()
        _, features = run_population(name + "_id", train_images[rows], rows, fine,
                                     partition["fine_to_coarse"][fine], lookup[fine])
        feature_scores(name + "_id", features)
    id_rows = test["id_test_indices"]
    _, id_features = run_population(
        "id_test", test_images[id_rows], id_rows,
        test["id_original_fine_labels"], test["id_original_coarse_labels"],
        test["id_class_targets"])
    feature_scores("id_test", id_features)
    ood_rows = test["ood_test_indices"]
    _, ood_features = run_population(
        "ood_test", test_images[ood_rows], ood_rows,
        test["ood_original_fine_labels"], test["ood_original_coarse_labels"], None)
    feature_scores("ood_test", ood_features)
    require(file_hash(checkpoint) == selection["checkpoint_sha256"] and
            file_hash(RUN / "test-population.npz") ==
            test_manifest["test_population_sha256"] and
            all(file_hash(ROOT / name) == digest for name, digest in sources.items()),
            "checkpoint, test population or extraction source changed mid-run")
    manifest = {
        "state": "extracted_pending_CPU_calibration_and_audit", "seed": args.seed,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "checkpoint_epoch": selection["epoch"],
        "training_audit_sha256": file_hash(RUN / f"train-audit-seed{args.seed}" / "audit.json"),
        "checkpoint_inference_audit_sha256": file_hash(
            RUN / f"checkpoint-inference-audit-seed{args.seed}" / "audit.json"),
        "partition_sha256": file_hash(RUN / "partition.npz"),
        "test_population_sha256": file_hash(RUN / "test-population.npz"),
        "test_population_audit_sha256": file_hash(RUN / "test-population-audit.json"),
        "train_indices_sha256": index_hash(train_rows),
        "input_pixels": inputs,
        "feature_fit": {"population": "ID optimization training only", "count": 36000,
                        "knn_k": 50, "mahalanobis_shrinkage": .01},
        "environment": environment(), "source_hashes": sources,
        "gpu_guard": {"samples": samples, "max_used_mib": 4096,
                      "max_util_percent": 55, "max_memory_util_percent": 10,
                      "max_power_w": 55, "sample_gap_seconds": 2},
        "outputs": outputs, "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Frozen checkpoint raw outputs and ID-trained feature scores; no temperature or threshold fit.",
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "status.json", {
        "state": manifest["state"], "seed": args.seed,
        "manifest_sha256": file_hash(output / "manifest.json"),
        "completed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"seed": args.seed, "state": manifest["state"],
                      "outputs": len(outputs)}), flush=True)


if __name__ == "__main__":
    main()
