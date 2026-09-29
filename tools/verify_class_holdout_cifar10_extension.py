"""Write-once/read-only completion evidence for exploratory 80-class CIFAR-10."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_class_holdout_completion import validate as validate_baseline
from verify_class_holdout_heterogeneity import validate as validate_classes
from verify_class_holdout_bootstrap import validate as validate_bootstrap


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
OUT = ROOT / "report/class-holdout-cifar10-extension"
COMPLETION = OUT / "completion.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
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
    prior = (
        ("report/cifar100-class-holdout/completion.json", validate_baseline),
        ("report/cifar100-class-holdout-classes/completion.json", validate_classes),
        ("report/cifar100-class-holdout-bootstrap/completion.json", validate_bootstrap),
    )
    for name, verifier in prior:
        require(read(ROOT / name)["inputs_sha256"] == verifier(),
                "prior immutable complete phase changed: " + name)
    declaration = read(RUN / "declaration.json")
    population = read(RUN / "test-population.json")
    test_audit = read(RUN / "test-population-audit.json")
    require(declaration["state"] == "frozen_before_80class_cifar10_inference" and
            declaration["n_id_reference"] == 8000 and
            declaration["n_ood"] == 10000 and
            declaration["seeds"] == [0, 1, 2] and
            declaration["variants"] == ["T1", "calibrated"] and
            declaration["methods"] == ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"] and
            declaration["targets"] == [80, 90, 95, 99] and
            len(declaration["inputs_sha256"]) == 37 and
            declaration["cifar10_archive_official_md5"] ==
            md5(ROOT / "data/cifar-10-python.tar.gz") and
            declaration["cifar10_test_official_md5"] ==
            md5(ROOT / "data/cifar-10-batches-py/test_batch"),
            "frozen official-source three-seed CIFAR-10 declaration")
    for name, digest in declaration["inputs_sha256"].items():
        require(sha(ROOT / name) == digest, "declared original source changed: " + name)
    require(population["state"] == "population_pending_independent_audit" and
            population["count"] == 10000 and
            population["class_counts"] == [1000] * 10 and
            population["declaration_sha256"] == sha(RUN / "declaration.json") and
            population["population_sha256"] == sha(RUN / "test-population.npz") and
            population["producer_source_sha256"] ==
            sha(ROOT / "tools/prepare_class_holdout_cifar10.py") and
            test_audit["state"] == "passed" and test_audit["count"] == 10000 and
            test_audit["class_counts"] == [1000] * 10 and
            test_audit["pixel_sha256"] == population["pixel_sha256"] and
            test_audit["test_population_manifest_sha256"] ==
            sha(RUN / "test-population.json") and
            test_audit["test_population_sha256"] == sha(RUN / "test-population.npz") and
            test_audit["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_cifar10_test.py"),
            "all-row independent official population audit")
    files = set(declaration["inputs_sha256"])
    files.update({
        "runs/class-holdout-cifar10-extension/declaration.json",
        "runs/class-holdout-cifar10-extension/test-population.npz",
        "runs/class-holdout-cifar10-extension/test-population.json",
        "runs/class-holdout-cifar10-extension/test-population-audit.json",
        "tools/prepare_class_holdout_cifar10.py",
        "tools/audit_class_holdout_cifar10_test.py",
        "tools/extract_class_holdout_cifar10.py",
        "tools/audit_class_holdout_cifar10_raw.py",
        "tools/score_class_holdout_cifar10.py",
        "tools/audit_class_holdout_cifar10_scores.py",
        "tools/build_class_holdout_cifar10_report.py",
        "tools/audit_class_holdout_cifar10_report.py",
        "tools/verify_class_holdout_cifar10_extension.py",
        "report/class-holdout-cifar10-extension/summary.json",
        "report/class-holdout-cifar10-extension/metrics.csv",
        "report/class-holdout-cifar10-extension/operating.csv",
        "report/class-holdout-cifar10-extension/REPORT.md",
        "report/class-holdout-cifar10-extension/audit.json",
        "report/class-holdout-cifar10-extension/regression.log",
    })
    for seed in range(3):
        extraction_dir = RUN / f"extract-seed{seed}"
        scoring_dir = RUN / f"score-seed{seed}"
        extraction = read(extraction_dir / "manifest.json")
        extraction_status = read(extraction_dir / "status.json")
        raw_audit_path = RUN / f"raw-audit-seed{seed}.json"
        raw_audit = read(raw_audit_path)
        scoring = read(scoring_dir / "manifest.json")
        scoring_status = read(scoring_dir / "status.json")
        score_audit_path = RUN / f"score-audit-seed{seed}.json"
        score_audit = read(score_audit_path)
        require(extraction["state"] == extraction_status["state"] ==
                "extracted_pending_independent_audit" and
                extraction["seed"] == extraction_status["seed"] == seed and
                extraction_status["manifest_sha256"] == sha(extraction_dir / "manifest.json") and
                extraction["declaration_sha256"] == sha(RUN / "declaration.json") and
                extraction["test_population_audit_sha256"] ==
                sha(RUN / "test-population-audit.json") and
                extraction["checkpoint_sha256"] ==
                sha(ROOT / f"runs/cifar100-class-holdout/seed{seed}/final-best.pt") and
                raw_audit["state"] == "passed" and raw_audit["seed"] == seed and
                raw_audit["count"] == 10000 and
                raw_audit["checked_t1_score_vectors"] == 4 and
                len(raw_audit["feature_probe_rows"]) == 15 and
                raw_audit["manifest_sha256"] == sha(extraction_dir / "manifest.json") and
                raw_audit["audit_source_sha256"] ==
                sha(ROOT / "tools/audit_class_holdout_cifar10_raw.py"),
                "full frozen checkpoint/raw/feature independent audit")
        require(len(extraction["gpu_guard_samples"]) == 3 and
                all(sample["used_mib"] <= 4096 and sample["util_percent"] <= 55 and
                    sample["memory_util_percent"] <= 10 and sample["power_w"] <= 55
                    for sample in extraction["gpu_guard_samples"]),
                "sequential GPU contention guard")
        for name, digest in extraction["output_sha256"].items():
            require(sha(extraction_dir / name) == digest,
                    "raw extraction output changed: " + name)
        for name, digest in extraction["source_sha256"].items():
            require(sha(ROOT / name) == digest ==
                    sha(extraction_dir / "source_snapshot" / name),
                    "executed extraction source changed: " + name)
            files.add((extraction_dir / "source_snapshot" / name).relative_to(ROOT).as_posix())
        require(scoring["state"] == scoring_status["state"] ==
                "scored_pending_independent_audit" and
                scoring["seed"] == scoring_status["seed"] == seed and
                scoring_status["manifest_sha256"] == sha(scoring_dir / "manifest.json") and
                scoring["extraction_manifest_sha256"] ==
                sha(extraction_dir / "manifest.json") and
                scoring["raw_audit_sha256"] == sha(raw_audit_path) and
                scoring["n_id"] == 8000 and scoring["n_ood"] == 10000 and
                scoring["ood_prevalence"] == 10000 / 18000 and
                score_audit["state"] == "passed" and score_audit["seed"] == seed and
                score_audit["scoring_manifest_sha256"] == sha(scoring_dir / "manifest.json") and
                score_audit["audit_source_sha256"] ==
                sha(ROOT / "tools/audit_class_holdout_cifar10_scores.py") and
                score_audit["checked"] == {"score_vectors": 12,
                                           "metric_fields": 156,
                                           "operating_fields": 528,
                                           "ood_mask_vectors": 48,
                                           "frozen_id_mask_vectors": 48},
                "all six methods, two variants, four thresholds independently audited")
        for name, digest in scoring["output_sha256"].items():
            require(sha(scoring_dir / name) == digest,
                    "scored output changed: " + name)
        for name, digest in scoring["source_sha256"].items():
            require(sha(ROOT / name) == digest ==
                    sha(scoring_dir / "source_snapshot" / name),
                    "executed scoring source changed: " + name)
            files.add((scoring_dir / "source_snapshot" / name).relative_to(ROOT).as_posix())
        for path in (extraction_dir / "manifest.json", extraction_dir / "status.json",
                     raw_audit_path, scoring_dir / "manifest.json",
                     scoring_dir / "status.json", score_audit_path):
            files.add(path.relative_to(ROOT).as_posix())
        for name in extraction["output_sha256"]:
            files.add((extraction_dir / name).relative_to(ROOT).as_posix())
        for name in scoring["output_sha256"]:
            files.add((scoring_dir / name).relative_to(ROOT).as_posix())
    summary, audit = read(OUT / "summary.json"), read(OUT / "audit.json")
    require(summary["state"] == "report_pending_independent_audit" and
            summary["n_id"] == 8000 and summary["n_ood"] == 10000 and
            len(summary["metric_rows"]) == 48 and
            len(summary["operating_rows"]) == 144 and
            summary["csv_sha256"] == {"metrics": sha(OUT / "metrics.csv"),
                                      "operating": sha(OUT / "operating.csv")} and
            audit["state"] == "passed" and
            audit["checked"] == {"metric_seed_values": 144,
                                 "metric_statistics": 240,
                                 "operating_seed_values": 432,
                                 "operating_counts": 432,
                                 "operating_statistics": 432,
                                 "csv_rows": 192,
                                 "display_rows": 12} and
            audit["source_sha256"] == {
                "summary": sha(OUT / "summary.json"),
                "metric_csv": sha(OUT / "metrics.csv"),
                "operating_csv": sha(OUT / "operating.csv"),
                "report": sha(OUT / "REPORT.md"),
                "audit_source": sha(ROOT / "tools/audit_class_holdout_cifar10_report.py"),
            }, "complete 3-seed report and independent display/statistical audit")
    regression = (OUT / "regression.log").read_text(encoding="utf-8")
    require("91 passed" in regression and "failed" not in regression,
            "passing 91-test regression evidence")
    return {name: sha(ROOT / name) for name in sorted(files)}


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
            "three_full_frozen_seeds": [0, 1, 2],
            "official_ood_test_count": 10000,
            "source_file_count": len(hashes),
            "inputs_sha256": hashes,
            "boundary": "Exploratory complete CIFAR-10 test extension; prior OOD exposure, unequal AUPR prevalence and no OOD-based selection.",
        }
        COMPLETION.write_text(json.dumps(record, indent=2) + "\n",
                              encoding="utf-8")
    else:
        require(COMPLETION.exists(), "completion not yet created")
        record = read(COMPLETION)
        require(record["state"] == "complete" and
                record["three_full_frozen_seeds"] == [0, 1, 2] and
                record["official_ood_test_count"] == 10000 and
                record["source_file_count"] == len(hashes) and
                record["inputs_sha256"] == hashes,
                "write-once completion evidence differs")
    print(json.dumps({"state": "complete", "source_files": len(hashes),
                      "write": args.write}))


if __name__ == "__main__":
    main()
