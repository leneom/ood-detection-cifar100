"""Write once or verify complete local held-out-class descriptive evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_class_holdout_completion import validate as validate_baseline


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "runs/cifar100-class-holdout-classes"
OUT = ROOT / "report/cifar100-class-holdout-classes"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def validate() -> dict[str, str]:
    declaration = read(DIAG / "declaration.json")
    summary = read(OUT / "summary.json")
    audit = read(OUT / "audit.json")
    baseline = read(ROOT / "report/cifar100-class-holdout/completion.json")
    require(baseline["state"] == "complete" and
            baseline["inputs_sha256"] == validate_baseline(),
            "complete 160-file baseline no longer verifies")
    require(declaration["state"] == "frozen_before_per_class_metric_access" and
            declaration["seeds"] == [0, 1, 2] and
            declaration["all_held_out_fine_classes"] == 20 and
            declaration["ood_images_per_class"] == 100 and
            declaration["shared_id_test_count"] == 8000 and
            declaration["ood_test_count"] == 2000 and
            len(declaration["source_sha256"]) == 23 and
            summary["state"] == "descriptive_complete_pending_independent_audit" and
            summary["declaration_sha256"] == sha(DIAG / "declaration.json") and
            summary["baseline_completion_sha256"] ==
            sha(ROOT / "report/cifar100-class-holdout/completion.json") and
            summary["producer_source_sha256"] ==
            sha(ROOT / "tools/build_class_holdout_heterogeneity.py") and
            summary["class_count"] == 20 and
            summary["ood_images_per_class"] == 100 and
            summary["shared_id_test_count"] == 8000,
            "frozen 20-class description differs")
    for item in declaration["source_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "source changed: " + item["path"])
    require(audit["state"] == "passed" and
            audit["declaration_sha256"] == sha(DIAG / "declaration.json") and
            audit["checked"] == {
                "class_auroc": 720, "class_acceptance": 2880,
                "id_acceptance": 144, "aggregate_identities": 180,
                "csv_auroc_rows": 240, "csv_acceptance_rows": 960,
                "distribution_quantiles": 120, "display_rows": 12,
            } and
            audit["class_values_sha256"] ==
            summary["class_values_sha256"] == sha(OUT / "class_values.npz") and
            audit["summary_sha256"] == sha(OUT / "summary.json") and
            audit["class_auroc_csv_sha256"] == sha(OUT / "class_auroc.csv") and
            audit["class_acceptance_csv_sha256"] ==
            sha(OUT / "class_acceptance.csv") and
            audit["report_sha256"] == sha(OUT / "REPORT.md") and
            audit["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_heterogeneity.py"),
            "independent full-class numerical/display audit differs")
    paths = {item["path"] for item in declaration["source_sha256"].values()}
    paths.update({
        "runs/cifar100-class-holdout-classes/declaration.json",
        "runs/cifar100-class-holdout/partition.json",
        "runs/cifar100-class-holdout/test-population.json",
        "report/cifar100-class-holdout-classes/class_values.npz",
        "report/cifar100-class-holdout-classes/class_auroc.csv",
        "report/cifar100-class-holdout-classes/class_acceptance.csv",
        "report/cifar100-class-holdout-classes/summary.json",
        "report/cifar100-class-holdout-classes/REPORT.md",
        "report/cifar100-class-holdout-classes/audit.json",
        "tools/build_class_holdout_heterogeneity.py",
        "tools/audit_class_holdout_heterogeneity.py",
        "tools/verify_class_holdout_heterogeneity.py",
        "tools/verify_class_holdout_completion.py",
    })
    return {name: sha(ROOT / name) for name in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = OUT / "completion.json"
    if args.write:
        require(not path.exists(), "class diagnostic completion exists")
        inputs = validate()
        record = {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "class_count": 20, "images_per_class": 100,
            "seeds": [0, 1, 2],
            "inputs_sha256": inputs,
            "boundary": "Full post-hoc description of the same frozen test, not independent confirmation or OOD-based method selection.",
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record = read(path)
    require(record["state"] == "complete" and
            record["class_count"] == 20 and
            record["images_per_class"] == 100 and
            record["seeds"] == [0, 1, 2], "class diagnostic completion header")
    actual = validate()
    require(record["inputs_sha256"] == actual, "class diagnostic input changed")
    print(json.dumps({"state": "verified_complete", "files": len(actual),
                      "classes": record["class_count"]}), flush=True)


if __name__ == "__main__":
    main()
