"""Sequential guarded CIFAR-10 inference from frozen 80-class ID-only fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from extract_class_holdout import preflight as frozen_checkpoint_preflight
from src.ood.data import load_cifar
from src.ood.features import extract, KNN, Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import environment, seed_all
from src.ood.scores import METHODS as LOGIT_METHODS, score_logits
from verify_class_holdout_completion import validate as validate_baseline


RUN = ROOT / "runs/class-holdout-cifar10-extension"
BASE = ROOT / "runs/cifar100-class-holdout"
SOURCES = (
    "docs/CLASS_HOLDOUT_CIFAR10_EXTENSION.md",
    "tools/extract_class_holdout_cifar10.py",
    "tools/extract_class_holdout.py",
    "src/ood/data.py", "src/ood/features.py", "src/ood/models.py",
    "src/ood/runtime.py", "src/ood/scores.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def save(path: Path, **data: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **data)


def gpu_guard() -> list[dict]:
    samples = []
    for attempt in range(3):
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        rows = output.splitlines()
        require(len(rows) == 1, "exactly one verified Windows GPU required")
        used, util, memory, power, total = [part.strip() for part in rows[0].split(",")]
        sample = {"used_mib": int(used), "util_percent": int(util),
                  "memory_util_percent": int(memory), "power_w": float(power),
                  "total_mib": int(total)}
        samples.append(sample)
        if attempt < 2:
            time.sleep(2)
    require(all(sample["used_mib"] <= 4096 and sample["util_percent"] <= 55 and
                sample["memory_util_percent"] <= 10 and sample["power_w"] <= 55
                for sample in samples), "GPU contention guard deferred: " + str(samples))
    return samples


def run(seed: int, batch_size: int) -> None:
    require(batch_size > 0, "positive batch size required")
    destination = RUN / f"extract-seed{seed}"
    require(not destination.exists(), "write-once extraction exists; inspect it")
    declaration = read(RUN / "declaration.json")
    audit = read(RUN / "test-population-audit.json")
    population = read(RUN / "test-population.json")
    require(declaration["state"] == "frozen_before_80class_cifar10_inference" and
            declaration["n_id_reference"] == 8000 and
            declaration["n_ood"] == 10000 and
            declaration["seeds"] == [0, 1, 2] and
            declaration["methods"] == ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"] and
            audit["state"] == "passed" and audit["count"] == 10000 and
            audit["declaration_sha256"] == sha(RUN / "declaration.json") and
            audit["test_population_manifest_sha256"] == sha(RUN / "test-population.json") and
            audit["test_population_sha256"] == sha(RUN / "test-population.npz") and
            population["pixel_sha256"] == audit["pixel_sha256"],
            "frozen full CIFAR-10 test population audit required")
    for name, digest in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == digest, "declared input changed: " + name)
    complete = read(ROOT / "report/cifar100-class-holdout/completion.json")
    require(complete["state"] == "complete" and
            complete["inputs_sha256"] == validate_baseline(),
            "original complete three-seed experiment no longer verifies")
    config, selection, partition, _, checkpoint = frozen_checkpoint_preflight(seed)
    frozen_extract = read(BASE / f"extract-seed{seed}" / "manifest.json")
    frozen_scores = read(BASE / f"score-seed{seed}" / "manifest.json")
    score_audit = read(BASE / f"score-audit-seed{seed}" / "audit.json")
    bank_path = BASE / f"extract-seed{seed}" / "id_train_raw.npz"
    require(frozen_extract["outputs"]["id_train_raw.npz"] == sha(bank_path) and
            frozen_extract["checkpoint_sha256"] ==
            frozen_scores["checkpoint_sha256"] == selection["checkpoint_sha256"] and
            score_audit["state"] == "passed" and
            score_audit["scoring_manifest_sha256"] ==
            sha(BASE / f"score-seed{seed}" / "manifest.json") and
            frozen_scores["temperature_fit_role"] == "ID_temperature_1200" and
            frozen_scores["threshold_fit_role"] == "ID_threshold_1200" and
            frozen_scores["targets"] == [.8, .9, .95, .99],
            "original ID-only fit/bank and score audit differ")
    with np.load(RUN / "test-population.npz", allow_pickle=False) as saved_population:
        original_rows = saved_population["original_rows"]
        original_labels = saved_population["original_labels"]
    images, labels = load_cifar(ROOT / "data", "cifar10", False, False)
    require(images.shape == (10000, 32, 32, 3) and
            np.array_equal(original_rows, np.arange(10000)) and
            np.array_equal(labels.numpy(), original_labels) and
            hashlib.sha256(images.numpy().tobytes()).hexdigest() == population["pixel_sha256"],
            "official CIFAR-10 pixels/rows/labels changed")
    samples = gpu_guard()
    require(torch.cuda.is_available(), "verified Windows CUDA required")
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    seed_all(0)
    torch.set_grad_enabled(False)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(selected["epoch"] == selection["epoch"] and
            selected["config"] == config and selected["checkpoint_count"] == 1600,
            "frozen selected checkpoint payload differs")
    model = build_model("resnet18", 80).to("cuda").to(memory_format=torch.channels_last)
    model.load_state_dict(selected["model"])
    model.eval()
    destination.mkdir()
    snapshot = destination / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (destination / "status.json").write_text(json.dumps({
        "state": "extracting", "seed": seed,
        "started_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n",
        encoding="utf-8")
    logits, features = extract(model, images, batch_size, "cuda")
    require(logits.shape == (10000, 80) and features.shape == (10000, 512),
            "complete 80-class CIFAR-10 inference required")
    raw_path = destination / "cifar10_raw.npz"
    t1_path = destination / "cifar10_T1_scores.npz"
    feature_path = destination / "cifar10_feature_scores.npz"
    save(raw_path, logits=logits.numpy(), features=features.numpy(),
         predictions=logits.argmax(1).numpy(), original_rows=original_rows,
         original_labels=original_labels)
    save(t1_path, **{method: score_logits(logits, method, 1.).numpy()
                     for method in LOGIT_METHODS})
    with np.load(bank_path, allow_pickle=False) as stored:
        train_features = torch.from_numpy(stored["features"].copy())
        train_targets = torch.from_numpy(stored["id_class_targets"].copy())
    require(train_features.shape == (36000, 512) and
            train_targets.shape == (36000,) and
            np.array_equal(np.unique(train_targets.numpy()), np.arange(80)) and
            np.array_equal(train_targets.numpy(),
                           np.searchsorted(partition["id_fine_labels"],
                                           partition["train_original_fine_labels"])),
            "original 36k optimization-ID feature bank differs")
    knn = KNN(k=50, device="cuda").fit(train_features)
    maha = Mahalanobis(shrinkage=.01, device="cuda").fit(train_features,
                                                          train_targets)
    save(feature_path, knn=knn.score(features).numpy(),
         mahalanobis=maha.score(features).numpy())
    require(all(sha(ROOT / name) == digest for name, digest in
                declaration["inputs_sha256"].items()) and
            all(sha(ROOT / name) == digest for name, digest in source_hashes.items()) and
            sha(checkpoint) == selection["checkpoint_sha256"] and
            sha(RUN / "test-population.npz") == population["population_sha256"],
            "frozen input/source changed during inference")
    record = {
        "state": "extracted_pending_independent_audit", "seed": seed,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "checkpoint_epoch": selection["epoch"],
        "original_extract_manifest_sha256": sha(BASE / f"extract-seed{seed}" / "manifest.json"),
        "original_score_manifest_sha256": sha(BASE / f"score-seed{seed}" / "manifest.json"),
        "original_score_audit_sha256": sha(BASE / f"score-audit-seed{seed}" / "audit.json"),
        "id_train_bank_sha256": sha(bank_path),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "test_population_sha256": sha(RUN / "test-population.npz"),
        "test_population_audit_sha256": sha(RUN / "test-population-audit.json"),
        "pixel_sha256": population["pixel_sha256"],
        "feature_fit": {"population": "original 36000 optimization-ID only",
                        "knn_k": 50, "mahalanobis_shrinkage": .01},
        "output_sha256": {name: sha(destination / name) for name in
                          (raw_path.name, t1_path.name, feature_path.name)},
        "source_sha256": source_hashes, "gpu_guard_samples": samples,
        "environment": environment(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "All 10000 CIFAR-10 test logits/features and ID-training-only feature scores; no calibration or OOD-based selection.",
    }
    (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n",
                                                encoding="utf-8")
    (destination / "status.json").write_text(json.dumps({
        "state": record["state"], "seed": seed,
        "manifest_sha256": sha(destination / "manifest.json")}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"seed": seed, "state": record["state"],
                      "count": 10000, "outputs": 3}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    lock_path = RUN / "extract.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run(args.seed, args.batch_size)
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
