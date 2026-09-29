"""Write-once training declaration for the LogitNorm representation study.

Binds the protocol, the training source and every CE comparator file before any
LogitNorm model exists. Evaluation code is bound separately before OOD scoring.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "logitnorm-representation"
TRAINING_SOURCES = ("docs/LOGITNORM_REPRESENTATION.md", "src/ood/logitnorm_train.py", "src/ood/isolated_train.py",
                    "src/ood/data.py", "src/ood/models.py", "src/ood/runtime.py", "src/ood/train.py",
                    "tools/freeze_logitnorm_representation.py")
LATER = {"tinyimagenet": "tinyimagenet-confirmation", "fashionmnist": "fashionmnist-extension",
         "places365": "places365-extension", "oxford_pets": "oxford-pets-extension"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def comparator_files(seed):
    files = [f"runs/validation-isolation/seed{seed}/final-best.pt",
             f"runs/validation-isolation/seed{seed}/final-selection.json",
             f"runs/validation-isolation/extract-seed{seed}/manifest.json",
             f"runs/validation-isolation/extract-seed{seed}/id_test_raw.npz",
             f"runs/validation-isolation/score-seed{seed}/manifest.json",
             f"runs/validation-isolation/score-seed{seed}/id_test_scores.npz",
             f"runs/validation-isolation/score-seed{seed}/id_validation_scores.npz"]
    files += [f"runs/validation-isolation/score-seed{seed}/{p}_scores.npz"
              for p in ("cifar10", "mnist", "svhn", "textures")]
    for folder in LATER.values():
        files += [f"runs/{folder}/extract-seed{seed}/manifest.json", f"runs/{folder}/score-seed{seed}/scores.npz"]
    return files


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / "declaration.json"
    if target.exists():
        raise FileExistsError("declaration is write-once")
    if any(RUN.glob("seed*")):
        raise RuntimeError("a LogitNorm run already exists; the declaration would not precede training")
    declaration = {
        "state": "frozen_before_training",
        "protocol": "docs/LOGITNORM_REPRESENTATION.md",
        "hypotheses": {"H3": "AUROC_LogitNorm(MSP) - AUROC_CE(MSP) > 0",
                       "H4": "AUROC_LogitNorm(KNN-50) - AUROC_CE(KNN-50) > 0"},
        "populations": ["cifar10", "mnist", "svhn", "textures", *LATER],
        "decision_rule": ("per population: supported iff all three seed-matched differences > 0, contradicted iff "
                          "all < 0, else mixed; confirmed iff supported on all eight, partially confirmed iff "
                          "supported on 1-7 and contradicted on none, else not confirmed"),
        "tau": 0.04,
        "training_sources": {name: sha(ROOT / name) for name in TRAINING_SOURCES},
        "ce_comparators": {f"seed{s}": {name: sha(ROOT / name) for name in comparator_files(s)} for s in (0, 1, 2)},
        "shared_inputs": {name: sha(ROOT / name) for name in
                          ("runs/validation-isolation/partition.npz", "data/cifar-100-python/train",
                           "data/cifar-100-python/test")},
        "created_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": declaration["state"], "declaration_sha256": sha(target)}))


if __name__ == "__main__":
    main()
