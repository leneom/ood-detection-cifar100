"""Write-once/read-only completion gate for fixed-prior exploratory AP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_class_holdout_completion import validate as validate_baseline
from verify_class_holdout_cifar10_extension import validate as validate_cifar10


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-prior-standardization"
OUT = ROOT / "report/class-holdout-prior-standardization"
COMPLETION = OUT / "completion.json"


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
    cifar10_path = ROOT / "report/class-holdout-cifar10-extension/completion.json"
    require(read(baseline_path)["inputs_sha256"] == validate_baseline() and
            read(cifar10_path)["inputs_sha256"] == validate_cifar10(),
            "both complete underlying OOD populations still verify")
    declaration = read(RUN / "declaration.json")
    manifest = read(RUN / "manifest.json")
    raw_audit = read(RUN / "raw-audit.json")
    summary = read(OUT / "summary.json")
    audit = read(OUT / "audit.json")
    require(declaration["state"] == "frozen_before_20pct_standardized_ap_calculation" and
            declaration["seeds"] == [0, 1, 2] and
            declaration["variants"] == ["T1", "calibrated"] and
            declaration["methods"] == ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"] and
            declaration["orientations"] == ["aupr_out", "aupr_in"] and
            declaration["populations"] == ["heldout_cifar100", "cifar10"] and
            declaration["n_id"] == 8000 and
            declaration["n_ood"] == {"heldout_cifar100": 2000, "cifar10": 10000} and
            declaration["target_ood_prior"] == .2 and
            len(declaration["inputs_sha256"]) == 28,
            "complete frozen 20% display-prior declaration")
    for name, digest in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == digest, "declared input changed: " + name)
    require(manifest["state"] == "raw_complete_pending_independent_audit" and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            manifest["raw_values_sha256"] == sha(RUN / "raw-values.npz") and
            manifest["producer_source_sha256"] ==
            sha(ROOT / "tools/run_class_holdout_prior_standardization.py") and
            manifest["shape"] == [3, 2, 6, 2, 2] and
            manifest["target_ood_prior"] == .2 and
            manifest["id_row_weight"] == .8 / 8000 and
            manifest["ood_row_weights"] == {"heldout_cifar100": .2 / 2000,
                                            "cifar10": .2 / 10000} and
            raw_audit["state"] == "passed" and
            raw_audit["checked"] == {"sklearn_weighted_ap": 144,
                                     "natural_ap": 144,
                                     "heldout_identity": 72,
                                     "weight_mass_checks": 72} and
            raw_audit["declaration_sha256"] == sha(RUN / "declaration.json") and
            raw_audit["run_manifest_sha256"] == sha(RUN / "manifest.json") and
            raw_audit["raw_values_sha256"] == sha(RUN / "raw-values.npz") and
            raw_audit["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_prior_standardization.py"),
            "complete independently audited 144 standardized AP values")
    require(summary["state"] == "report_pending_independent_audit" and
            summary["target_ood_prior"] == .2 and
            summary["seeds"] == [0, 1, 2] and
            len(summary["rows"]) == 48 and
            summary["statistics_csv_sha256"] == sha(OUT / "statistics.csv") and
            summary["source_sha256"] == {
                "declaration": sha(RUN / "declaration.json"),
                "run_manifest": sha(RUN / "manifest.json"),
                "raw_values": sha(RUN / "raw-values.npz"),
                "raw_audit": sha(RUN / "raw-audit.json"),
                "builder": sha(ROOT / "tools/build_class_holdout_prior_report.py"),
            } and
            audit["state"] == "passed" and
            audit["checked"] == {"standardized_seed_values": 144,
                                 "natural_seed_values": 144,
                                 "statistical_fields": 384,
                                 "csv_fields": 672,
                                 "display_rows": 12,
                                 "heldout_identities": 72} and
            audit["source_sha256"] == {
                "summary": sha(OUT / "summary.json"),
                "statistics_csv": sha(OUT / "statistics.csv"),
                "report": sha(OUT / "REPORT.md"),
                "raw_audit": sha(RUN / "raw-audit.json"),
                "audit_source": sha(ROOT / "tools/audit_class_holdout_prior_report.py"),
            }, "complete independent statistical/CSV/display report audit")
    regression = (OUT / "regression.log").read_text(encoding="utf-8")
    require("91 passed" in regression and "failed" not in regression,
            "full CPU regression log")
    paths = set(declaration["inputs_sha256"])
    paths.update({
        "runs/class-holdout-prior-standardization/declaration.json",
        "runs/class-holdout-prior-standardization/manifest.json",
        "runs/class-holdout-prior-standardization/raw-values.npz",
        "runs/class-holdout-prior-standardization/raw-audit.json",
        "tools/run_class_holdout_prior_standardization.py",
        "tools/audit_class_holdout_prior_standardization.py",
        "tools/build_class_holdout_prior_report.py",
        "tools/audit_class_holdout_prior_report.py",
        "tools/verify_class_holdout_prior_standardization.py",
        "report/class-holdout-prior-standardization/summary.json",
        "report/class-holdout-prior-standardization/statistics.csv",
        "report/class-holdout-prior-standardization/REPORT.md",
        "report/class-holdout-prior-standardization/audit.json",
        "report/class-holdout-prior-standardization/regression.log",
    })
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
            "target_ood_prior": .2,
            "source_file_count": len(hashes),
            "inputs_sha256": hashes,
            "boundary": "Exploratory hypothetical prevalence standardization of frozen scores; no detector selection or new validation.",
        }
        COMPLETION.write_text(json.dumps(record, indent=2) + "\n",
                              encoding="utf-8")
    else:
        require(COMPLETION.exists(), "completion not yet written")
        record = read(COMPLETION)
        require(record["state"] == "complete" and
                record["target_ood_prior"] == .2 and
                record["source_file_count"] == len(hashes) and
                record["inputs_sha256"] == hashes,
                "write-once evidence differs")
    print(json.dumps({"state": "complete", "source_files": len(hashes),
                      "write": args.write}))


if __name__ == "__main__":
    main()
