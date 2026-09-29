"""Apply the unchanged six-method ID-only scorer to a new class split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.runtime import file_hash
from tools import score_class_holdout as original
BASE = ROOT / "runs/cifar100-class-holdout-replication"


SOURCE = ("tools/score_class_holdout.py",
          "tools/score_class_holdout_replication.py",
          "tools/calibration_core.py", "src/ood/metrics.py")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    split = BASE / f"split{args.class_seed}"
    feature_folder = split / f"feature-audit-seed{args.seed}"
    feature = read(feature_folder / "audit.json")
    context = read(feature_folder / "class-provenance.json")
    require(feature["state"] == context["state"] == "passed" and
            feature["seed"] == context["training_seed"] == args.seed and
            context["class_rng_seed"] == args.class_seed and
            context["feature_audit_sha256"] == file_hash(feature_folder / "audit.json") and
            context["source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_replication_features.py") and
            context["compatibility_manifest_sha256"] ==
            file_hash(split / "declaration.json") and
            context["extraction_provenance_sha256"] == file_hash(
                split / f"extract-seed{args.seed}/class-provenance.json"),
            "source-bound independent ID optimization feature audit required")
    original.RUN = split
    original.SOURCE = SOURCE
    sys.argv = ["tools/score_class_holdout.py", "--seed", str(args.seed)]
    original.main()
    output = split / f"score-seed{args.seed}"
    manifest = read(output / "manifest.json")
    require(manifest["state"] == "scored_pending_independent_audit" and
            manifest["seed"] == args.seed and
            manifest["methods"] == ["msp", "energy", "maxlogit", "entropy",
                                    "knn", "mahalanobis"] and
            manifest["targets"] == [.80, .90, .95, .99] and
            manifest["temperature_fit_role"] == "ID_temperature_1200" and
            manifest["threshold_fit_role"] == "ID_threshold_1200" and
            manifest["n_id"] == 8000 and manifest["n_ood"] == 2000,
            "all pre-fixed score/ID-only fit outputs")
    record = {
        "state": "source_bound_replication_scoring",
        "class_rng_seed": args.class_seed, "training_seed": args.seed,
        "score_manifest_sha256": file_hash(output / "manifest.json"),
        "feature_audit_context_sha256": file_hash(feature_folder / "class-provenance.json"),
        "partition_sha256": file_hash(split / "partition.npz"),
        "source_sha256": file_hash(Path(__file__)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Unchanged six-method scorer; temperature and four targets from disjoint ID roles only.",
    }
    (output / "class-provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
