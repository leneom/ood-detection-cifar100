"""Write-once/read-only completion gate for the 1,000-draw local bootstrap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_class_holdout_completion import validate as validate_baseline
from verify_class_holdout_heterogeneity import validate as validate_classes


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-bootstrap"
REPORT = ROOT / "report/cifar100-class-holdout-bootstrap"
COMPLETION = REPORT / "completion.json"


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


def validate() -> dict[str, str]:
    baseline_path = ROOT / "report/cifar100-class-holdout/completion.json"
    class_path = ROOT / "report/cifar100-class-holdout-classes/completion.json"
    require(read(baseline_path)["inputs_sha256"] == validate_baseline() and
            read(class_path)["inputs_sha256"] == validate_classes(),
            "underlying completed aggregate and class diagnostics")
    declaration = read(RUN / "declaration.json")
    launch = read(RUN / "launch.json")
    status = read(RUN / "status.json")
    raw = read(RUN / "raw-audit.json")
    summary = read(REPORT / "summary.json")
    audit = read(REPORT / "audit.json")
    require(declaration["state"] == "frozen_before_stratified_draws" and
            declaration["replicates"] == 1000 and
            declaration["rng_seed_base"] == 2035 and
            declaration["id_classes"] == 80 and
            declaration["ood_classes"] == 20 and
            declaration["draws_per_class"] == 100 and
            declaration["fixed_id_target"] == 95 and
            len(declaration["inputs_sha256"]) == 22,
            "frozen 1000-draw fixed-class declaration")
    for item in declaration["inputs_sha256"].values():
        require(sha(ROOT / item["path"]) == item["sha256"],
                "declared source changed: " + item["path"])
    require(launch["state"] == "fixed_1000_draw_schedule_launched" and
            launch["declaration_sha256"] == sha(RUN / "declaration.json") and
            launch["replicates"] == 1000 and launch["block_size"] == 100 and
            status["state"] == "raw_complete_pending_independent_audit" and
            status["completed_blocks"] == list(range(10)) and
            status["completed_replicates"] == 1000 and
            status["launch_sha256"] == sha(RUN / "launch.json"),
            "fixed complete run/launch state")
    for name, expected in launch["source_sha256"].items():
        original = ROOT / name
        snapshot = RUN / "source_snapshot" / original.name
        require(sha(original) == sha(snapshot) == expected,
                "immutable producer source/snapshot changed")
    require(raw["state"] == "passed" and
            raw["declaration_sha256"] == sha(RUN / "declaration.json") and
            raw["launch_sha256"] == sha(RUN / "launch.json") and
            raw["status_sha256"] == sha(RUN / "status.json") and
            raw["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_bootstrap.py") and
            raw["checked"] == {
                "regenerated_weight_vectors": 2000,
                "within_class_weight_totals": 100000,
                "independent_ranking_fields": 144000,
                "independent_acceptance_counts": 72000,
                "expanded_sklearn_ranking_fields": 1440,
                "block_hashes": 30,
            }, "full independent raw bootstrap audit")
    require(len(raw["block_hashes"]) == 30, "ten complete three-file blocks")
    for block in range(10):
        stem = f"block-{block:03d}"
        marker_path = RUN / "blocks" / f"{stem}.json"
        weight_path = RUN / "blocks" / f"{stem}-weights.npz"
        metric_path = RUN / "blocks" / f"{stem}-metrics.npz"
        marker = read(marker_path)
        require(marker["state"] == "committed" and
                marker["start"] == block * 100 and
                marker["end_exclusive"] == (block + 1) * 100 and
                marker["launch_sha256"] == sha(RUN / "launch.json") and
                marker["weights_sha256"] == sha(weight_path) and
                marker["metrics_sha256"] == sha(metric_path),
                "immutable 100-draw block marker")
        for path in (marker_path, weight_path, metric_path):
            require(raw["block_hashes"][path.relative_to(ROOT).as_posix()] ==
                    sha(path), "raw audit block hash")
    require(summary["state"] == "report_pending_independent_audit" and
            summary["replicates"] == 1000 and len(summary["rows"]) == 72 and
            summary["statistics_csv_sha256"] == sha(REPORT / "statistics.csv") and
            summary["source_sha256"] == {
                "declaration": sha(RUN / "declaration.json"),
                "launch": sha(RUN / "launch.json"),
                "raw_audit": sha(RUN / "raw-audit.json"),
                "baseline_summary": sha(ROOT / "report/cifar100-class-holdout/summary.json"),
                "report_builder": sha(ROOT / "tools/build_class_holdout_bootstrap_report.py"),
            }, "source-bound complete 72-row report")
    require(audit["state"] == "passed" and
            audit["checked"] == {
                "original_seed_ranking_fields": 144,
                "original_fixed_mask_fields": 72,
                "bootstrap_quantiles": 216,
                "report_numeric_fields": 576,
                "csv_fields": 576,
                "display_rows": 12,
            } and
            audit["source_sha256"] == {
                "summary": sha(REPORT / "summary.json"),
                "statistics_csv": sha(REPORT / "statistics.csv"),
                "display": sha(REPORT / "REPORT.md"),
                "raw_audit": sha(RUN / "raw-audit.json"),
                "audit_source": sha(ROOT / "tools/audit_class_holdout_bootstrap_report.py"),
            }, "independent original/quantile/CSV/display report audit")
    paths = {item["path"] for item in declaration["inputs_sha256"].values()}
    paths.update({
        "docs/CLASS_HOLDOUT_BOOTSTRAP.md",
        "report/cifar100-class-holdout/completion.json",
        "report/cifar100-class-holdout-classes/completion.json",
        "runs/cifar100-class-holdout-bootstrap/declaration.json",
        "runs/cifar100-class-holdout-bootstrap/launch.json",
        "runs/cifar100-class-holdout-bootstrap/status.json",
        "runs/cifar100-class-holdout-bootstrap/raw-audit.json",
        "tools/audit_class_holdout_bootstrap.py",
        "tools/build_class_holdout_bootstrap_report.py",
        "tools/audit_class_holdout_bootstrap_report.py",
        "tools/verify_class_holdout_bootstrap.py",
        "report/cifar100-class-holdout-bootstrap/statistics.csv",
        "report/cifar100-class-holdout-bootstrap/summary.json",
        "report/cifar100-class-holdout-bootstrap/REPORT.md",
        "report/cifar100-class-holdout-bootstrap/audit.json",
    })
    paths.update(launch["source_sha256"])
    paths.update({"runs/cifar100-class-holdout-bootstrap/source_snapshot/" +
                  Path(name).name for name in launch["source_sha256"]})
    paths.update(raw["block_hashes"])
    return {name: sha(ROOT / name) for name in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    hashes = validate()
    if args.write:
        require(not COMPLETION.exists(), "write-once completion already exists")
        record = {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "replicates": 1000, "source_file_count": len(hashes),
            "inputs_sha256": hashes,
            "boundary": "Conditional within-class image bootstrap of fixed observed classes/models; descriptive intervals, no method selection or external confirmation.",
        }
        COMPLETION.write_text(json.dumps(record, indent=2) + "\n",
                              encoding="utf-8")
    else:
        require(COMPLETION.exists(), "write-once completion not yet created")
        record = read(COMPLETION)
        require(record["state"] == "complete" and
                record["replicates"] == 1000 and
                record["source_file_count"] == len(hashes) and
                record["inputs_sha256"] == hashes,
                "write-once completion source mismatch")
    print(json.dumps({"state": "complete", "source_files": len(hashes),
                      "write": args.write}))


if __name__ == "__main__":
    main()
