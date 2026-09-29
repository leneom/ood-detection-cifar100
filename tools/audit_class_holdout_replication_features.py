"""Run the unchanged independent ID-bank feature audit on a new class split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.runtime import file_hash
from tools import audit_class_holdout_features as original
BASE = ROOT / "runs/cifar100-class-holdout-replication"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-seed", type=int, choices=(2032, 2033), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    split = BASE / f"split{args.class_seed}"
    extracted = split / f"extract-seed{args.seed}"
    require(not (split / f"feature-audit-seed{args.seed}").exists(),
            "write-once feature audit exists")
    manifest = read(extracted / "manifest.json")
    provenance = read(extracted / "class-provenance.json")
    require(manifest["state"] == "extracted_pending_CPU_calibration_and_audit" and
            manifest["seed"] == args.seed and
            provenance["state"] == "source_bound_replication_extraction" and
            provenance["class_rng_seed"] == args.class_seed and
            provenance["training_seed"] == args.seed and
            provenance["extraction_manifest_sha256"] ==
            file_hash(extracted / "manifest.json") and
            provenance["source_sha256"] ==
            file_hash(ROOT / "tools/extract_class_holdout_replication.py"),
            "completed source-bound replication extraction required")
    root_declaration = read(BASE / "declaration.json")
    train_sha = root_declaration["inputs_sha256"]["data/cifar-100-python/train"]
    require(train_sha == file_hash(ROOT / "data/cifar-100-python/train"),
            "frozen official training file hash")
    compatibility = split / "declaration.json"
    record = {
        "state": "compatibility_view_for_unchanged_feature_auditor",
        "class_rng_seed": args.class_seed,
        "root_declaration_sha256": file_hash(BASE / "declaration.json"),
        "partition_sha256": file_hash(split / "partition.npz"),
        "train_file_sha256": train_sha,
        "source_sha256": file_hash(Path(__file__)),
        "boundary": "Provides the frozen train-file hash expected by the unchanged independent feature auditor; no detector fit or new population choice.",
    }
    if compatibility.exists():
        require(read(compatibility) == record,
                "existing feature-audit compatibility manifest differs")
    else:
        compatibility.write_text(json.dumps(record, indent=2) + "\n",
                                 encoding="utf-8")
    original.RUN = split
    sys.argv = ["tools/audit_class_holdout_features.py", "--seed", str(args.seed)]
    original.main()
    folder = split / f"feature-audit-seed{args.seed}"
    audit = read(folder / "audit.json")
    require(audit["state"] == "passed" and audit["seed"] == args.seed and
            audit["extraction_manifest_sha256"] ==
            file_hash(extracted / "manifest.json") and
            audit["source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_features.py"),
            "independent raw-pixel and ID feature fit audit")
    supplement = {
        "state": "passed", "class_rng_seed": args.class_seed,
        "training_seed": args.seed,
        "feature_audit_sha256": file_hash(folder / "audit.json"),
        "compatibility_manifest_sha256": file_hash(compatibility),
        "extraction_provenance_sha256": file_hash(extracted / "class-provenance.json"),
        "source_sha256": file_hash(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Original independent feature audit re-used unchanged with source-bound class context.",
    }
    (folder / "class-provenance.json").write_text(
        json.dumps(supplement, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
