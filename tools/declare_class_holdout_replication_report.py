"""Freeze the nine-model report contract before new class-partition test scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-replication"
OUTPUT = RUN / "report-declaration.json"
INPUTS = (
    "docs/CLASS_HOLDOUT_REPLICATION_REPORT.md",
    "docs/CLASS_HOLDOUT_REPLICATION.md",
    "docs/CIFAR100_CLASS_HOLDOUT.md",
    "runs/cifar100-class-holdout-replication/declaration.json",
    "report/cifar100-class-holdout/completion.json",
    "docs/PROTOCOL.md",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once declaration exists: {OUTPUT}")
    for class_seed in (2032, 2033):
        split = RUN / f"split{class_seed}"
        if (split / "test-population.npz").exists() or any(
            (split / f"score-seed{seed}").exists() for seed in (0, 1, 2)
        ):
            raise RuntimeError("new partition test population or score already exists")
    original = json.loads(
        (ROOT / "report/cifar100-class-holdout/completion.json").read_text(
            encoding="utf-8")
    )
    if original.get("state") != "complete" or len(original.get("inputs_sha256", {})) != 160:
        raise RuntimeError("original 160-file completion record differs")
    record = {
        "state": "frozen_before_new_partition_ood_test_scores",
        "declared_at": datetime.now(timezone.utc).isoformat(),
        "class_seeds": [2031, 2032, 2033],
        "training_seeds": [0, 1, 2],
        "variants": ["T1", "calibrated"],
        "methods": ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"],
        "metrics": ["auroc", "aupr_out", "aupr_in", "fpr95_id"],
        "id_threshold_targets": [80, 90, 95, 99],
        "operating_rates": [
            "calibration_acceptance", "id_test_acceptance", "ood_acceptance"
        ],
        "n_id_test_per_partition": 8000,
        "n_ood_test_per_partition": 2000,
        "statistics": "three training-seed points/mean/sample SD per partition; equal-partition grand mean and sample SD of three partition means",
        "inputs_sha256": {name: sha(ROOT / name) for name in INPUTS},
        "boundary": "Descriptive fixed-partition sensitivity; 2031 was viewed before 2032/2033 were selected. No OOD-based selection or independent-class-partition confidence interval.",
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"frozen {OUTPUT.relative_to(ROOT)} sha256={sha(OUTPUT)}")


if __name__ == "__main__":
    main()
