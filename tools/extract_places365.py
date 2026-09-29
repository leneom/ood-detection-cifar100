"""Frozen three-seed Places365 inference; no OOD-dependent fit or selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
ISOLATED = ROOT / "runs" / "validation-isolation"
sys.path.insert(0, str(ROOT))

from src.ood.features import extract, KNN, Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import file_hash, seed_all, environment
from src.ood.scores import METHODS as LOGIT_METHODS, score_logits


SOURCES = ("tools/extract_places365.py", "src/ood/features.py",
           "src/ood/models.py", "src/ood/runtime.py", "src/ood/data.py",
           "src/ood/scores.py", "tools/prepare_places365.py",
           "tools/audit_places365_cache.py", "tools/freeze_places365.py",
           "docs/PLACES365_EXTENSION.md",
           "docs/PLACES365_AMENDMENT_1.md")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, **content):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **content)
    temp.replace(path)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size <= 0:
        raise RuntimeError("verified CUDA and positive batch size required")
    declaration = read(RUN / "declaration.json")
    amendment = read(RUN / "amendment-1.json")
    cache_manifest = read(RUN / "cache-manifest.json")
    cache_audit = read(RUN / "cache-audit.json")
    cache_status = read(RUN / "cache-status.json")
    cache_path = ROOT / "data" / "places365-validation.npz"
    if (declaration["state"] != "protocol_frozen_before_places365_archive_access" or
            file_hash(ROOT / "docs/PLACES365_EXTENSION.md") != declaration["protocol_sha256"] or
            amendment["original_declaration_sha256"] != file_hash(RUN / "declaration.json") or
            amendment["amendment_sha256"] != file_hash(ROOT / "docs/PLACES365_AMENDMENT_1.md") or
            cache_status["state"] != "verified_source_and_pixels_pending_independent_audit" or
            cache_manifest["state"] != cache_status["state"] or
            cache_audit["cache_sha256"] != cache_manifest["cache_sha256"] or
            cache_status["manifest_sha256"] != file_hash(RUN / "cache-manifest.json") or
            cache_audit["state"] != "passed" or
            cache_audit["manifest_sha256"] != file_hash(RUN / "cache-manifest.json") or
            cache_manifest["cache_sha256"] != file_hash(cache_path)):
        raise ValueError("frozen archive, preprocessing or independent cache audit changed")
    for name, expected in declaration["input_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("frozen model or ID-only input differs: " + name)
    completion = read(ROOT / "report" / "validation-isolation" / "completion.json")
    if completion["state"] != "complete" or completion["three_full_training_seeds"] != [0, 1, 2]:
        raise ValueError("isolated predecessor incomplete")
    phase = read(ISOLATED / "status.json")
    if phase["state"] != "training_complete_pending_evaluation":
        raise ValueError("three isolated models not frozen")
    old = ISOLATED / f"extract-seed{args.seed}"
    old_manifest = read(old / "manifest.json")
    old_status = read(old / "status.json")
    selected = ISOLATED / f"seed{args.seed}" / "final-best.pt"
    if (old_status["manifest_sha256"] != file_hash(old / "manifest.json") or
            old_manifest["checkpoint_sha256"] != file_hash(selected) or
            old_manifest["checkpoint_epoch"] != (92, 96, 98)[args.seed] or
            old_manifest["outputs"]["id_train_raw.npz"] != file_hash(old / "id_train_raw.npz")):
        raise ValueError("frozen checkpoint or training-ID feature bank differs")
    score_manifest = read(ISOLATED / f"score-seed{args.seed}" / "manifest.json")
    score_audit = read(ISOLATED / f"audit-seed{args.seed}" / "audit.json")
    if (score_audit["state"] != "passed" or
            score_audit["scoring_manifest_sha256"] !=
            file_hash(ISOLATED / f"score-seed{args.seed}" / "manifest.json") or
            score_manifest["checkpoint_sha256"] != old_manifest["checkpoint_sha256"]):
        raise ValueError("frozen ID-only score and threshold calibration differs")
    destination = RUN / f"extract-seed{args.seed}"
    if destination.exists():
        raise FileExistsError("Places365 extraction already exists")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    with np.load(cache_path, allow_pickle=False) as z:
        images = z["images"]
        filenames = z["filenames"]
        labels = z["labels"]
    if images.shape != (36500, 32, 32, 3) or images.dtype != np.uint8 or len(filenames) != 36500:
        raise ValueError("frozen uint8 NHWC cache differs")
    with np.load(old / "id_train_raw.npz", allow_pickle=False) as z:
        train_features = torch.from_numpy(z["features"].copy())
        train_labels = torch.from_numpy(z["labels"].copy())
    if train_features.shape != (45000, 512) or train_labels.shape != (45000,):
        raise ValueError("training-ID feature bank differs")
    config = read(ISOLATED / f"seed{args.seed}" / "config.json")
    seed_all(0)
    torch.set_grad_enabled(False)
    weights = torch.load(selected, map_location="cpu", weights_only=True)
    if weights["epoch"] != old_manifest["checkpoint_epoch"] or weights["config"] != config:
        raise ValueError("selected model checkpoint payload changed")
    model = build_model(config["model"], config["classes"]).to("cuda")
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(weights["model"])
    model.eval()
    destination.mkdir()
    snapshots = destination / "source_snapshot"
    snapshots.mkdir()
    for name in SOURCES:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(destination / "status.json", {"state": "extracting", "seed": args.seed,
                                        "started_at": datetime.now(timezone.utc).isoformat()})
    logits, features = extract(model, torch.from_numpy(images), args.batch_size, "cuda")
    knn = KNN(k=50, device="cuda").fit(train_features)
    maha = Mahalanobis(shrinkage=.01, device="cuda").fit(train_features, train_labels)
    t1 = {method: score_logits(logits, method, 1.).numpy() for method in LOGIT_METHODS}
    fit_scores = {"knn": knn.score(features).numpy(),
                  "mahalanobis": maha.score(features).numpy()}
    save(destination / "raw.npz", logits=logits.numpy(), features=features.numpy(),
         predictions=logits.argmax(dim=1).numpy(), source_rows=np.arange(36500),
         filenames=filenames, labels=labels)
    save(destination / "T1_scores.npz", **t1)
    save(destination / "feature_scores.npz", **fit_scores)
    outputs = {name: file_hash(destination / name)
               for name in ("raw.npz", "T1_scores.npz", "feature_scores.npz")}
    if (file_hash(cache_path) != cache_manifest["cache_sha256"] or
            any(file_hash(ROOT / name) != value for name, value in source_hashes.items())):
        raise RuntimeError("cache or executed source changed during extraction")
    manifest = {"state": "extracted_pending_independent_audit_and_scoring",
                "seed": args.seed, "count": 36500,
                "checkpoint_sha256": old_manifest["checkpoint_sha256"],
                "checkpoint_epoch": old_manifest["checkpoint_epoch"],
                "training_id_bank_sha256": old_manifest["outputs"]["id_train_raw.npz"],
                "cache_sha256": cache_manifest["cache_sha256"],
                "cache_audit_sha256": file_hash(RUN / "cache-audit.json"),
                "isolation_completion_sha256": file_hash(ROOT / "report/validation-isolation/completion.json"),
                "id_score_manifest_sha256": file_hash(ISOLATED / f"score-seed{args.seed}" / "manifest.json"),
                "source_hashes": source_hashes, "outputs": outputs,
                "environment": environment(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "boundary": "Frozen classifier/ID-only feature fit; no Places365-dependent fit."}
    write(destination / "manifest.json", manifest)
    write(destination / "status.json", {"state": manifest["state"], "seed": args.seed,
                                        "manifest_sha256": file_hash(destination / "manifest.json")})
    print(json.dumps({"state": manifest["state"], "seed": args.seed, "count": 36500}),
          flush=True)


if __name__ == "__main__":
    main()

