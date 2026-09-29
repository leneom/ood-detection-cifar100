"""Reuse the unchanged independent NumPy/sklearn full-score audit per split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ood.runtime import file_hash
from tools import audit_class_holdout_scores as original
BASE = ROOT / "runs/cifar100-class-holdout-replication"


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
    score_folder = split / f"score-seed{args.seed}"
    require(not (split / f"score-audit-seed{args.seed}").exists(),
            "write-once score audit exists")
    manifest = read(score_folder / "manifest.json")
    context = read(score_folder / "class-provenance.json")
    require(manifest["state"] == "scored_pending_independent_audit" and
            manifest["seed"] == args.seed and
            context["state"] == "source_bound_replication_scoring" and
            context["class_rng_seed"] == args.class_seed and
            context["training_seed"] == args.seed and
            context["score_manifest_sha256"] ==
            file_hash(score_folder / "manifest.json") and
            context["source_sha256"] ==
            file_hash(ROOT / "tools/score_class_holdout_replication.py") and
            context["partition_sha256"] == file_hash(split / "partition.npz") and
            context["feature_audit_context_sha256"] == file_hash(
                split / f"feature-audit-seed{args.seed}/class-provenance.json"),
            "source-bound scorer on the audited class split")
    original.RUN = split
    sys.argv = ["tools/audit_class_holdout_scores.py", "--seed", str(args.seed)]
    original.main()
    folder = split / f"score-audit-seed{args.seed}"
    audit = read(folder / "audit.json")
    require(audit["state"] == "passed" and audit["seed"] == args.seed and
            audit["source_sha256"] ==
            file_hash(ROOT / "tools/audit_class_holdout_scores.py"),
            "all saved logits, scores, metrics, thresholds and masks audited")
    record = {
        "state": "passed", "class_rng_seed": args.class_seed,
        "training_seed": args.seed,
        "score_audit_sha256": file_hash(folder / "audit.json"),
        "scoring_context_sha256": file_hash(score_folder / "class-provenance.json"),
        "source_sha256": file_hash(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Original full-score independent audit re-used unchanged; no OOD fit or selection.",
    }
    (folder / "class-provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
