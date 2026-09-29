"""Write once, then read-only verify the complete nine-model sensitivity phase."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/cifar100-class-holdout-replication"
REPORT = ROOT / "report/cifar100-class-holdout-replication"
PARTITIONS = (2032, 2033)
SEEDS = (0, 1, 2)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def validate() -> dict[str, str]:
    inputs: dict[str, str] = {}
    def bind(path: Path, expected: str | None = None) -> str:
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"source/output changed: {path}")
        inputs[path.relative_to(ROOT).as_posix()] = digest
        return digest

    import sys
    sys.path.insert(0, str(ROOT))
    from tools.verify_class_holdout_completion import validate as old_validate
    old_path = ROOT / "report/cifar100-class-holdout/completion.json"
    old = read(old_path)
    require(old["state"] == "complete" and
            len(old["inputs_sha256"]) == 160 and
            old["inputs_sha256"] == old_validate(),
            "original three-model 160-file baseline")
    bind(old_path)
    declaration_path = BASE / "report-declaration.json"
    declaration = read(declaration_path)
    require(declaration["state"] == "frozen_before_new_partition_ood_test_scores" and
            declaration["class_seeds"] == [2031, 2032, 2033] and
            declaration["training_seeds"] == list(SEEDS),
            "predeclared nine-model report")
    bind(declaration_path)
    for name, digest in declaration["inputs_sha256"].items():
        bind(ROOT / name, digest)
    status_path = BASE / "status.json"
    status = read(status_path)
    require(status["state"] == "training_complete_pending_independent_audit" and
            status["trainer_running"] is False and
            {(r["class_seed"], r["training_seed"]) for r in status["completed"]} ==
            {(p, s) for p in PARTITIONS for s in SEEDS} and
            all(r["epochs"] == 100 for r in status["completed"]),
            "six full schedules")
    bind(status_path)

    summary_path = REPORT / "summary.json"
    audit_path = REPORT / "audit.json"
    summary, audit = read(summary_path), read(audit_path)
    require(summary["state"] == "report_pending_independent_audit" and
            audit["state"] == "passed" and
            summary["class_seeds"] == audit["class_seeds"] ==
            [2031, 2032, 2033] and
            summary["training_seeds"] == audit["training_seeds"] == list(SEEDS) and
            audit["nine_score_manifests"] == 9 and
            audit["metric_csv_rows"] == 48 and
            audit["operating_csv_rows"] == 144 and
            audit["displayed_method_rows"] == 12 and
            audit["report_spec_sha256"] ==
            sha(ROOT / "docs/CLASS_HOLDOUT_REPLICATION_REPORT.md") and
            audit["report_declaration_sha256"] == sha(declaration_path) and
            summary["source_sha256"] == audit["report_source_sha256"] ==
            sha(ROOT / "tools/build_class_holdout_replication_report.py") ==
            sha(REPORT / "executed_report_source.py") and
            audit["audit_source_sha256"] ==
            sha(ROOT / "tools/audit_class_holdout_replication_report.py") ==
            sha(REPORT / "audit-source.py"),
            "independent full-grid numerical/display report audit")
    bind(summary_path)
    bind(audit_path)
    for name, digest in summary["source_inputs_sha256"].items():
        bind(ROOT / name, digest)
    for name, digest in audit["outputs_sha256"].items():
        bind(ROOT / name, digest)
    for path in (REPORT / "metrics.csv", REPORT / "operating.csv",
                 REPORT / "SUMMARY.md", REPORT / "executed_report_source.py",
                 REPORT / "audit-source.py"):
        bind(path)

    # Source and raw-output validation remains live after the report audit.
    for partition in PARTITIONS:
        split = BASE / f"split{partition}"
        pa_path = split / "partition-audit.json"
        ta_path = split / "test-population-audit.json"
        pa, ta = read(pa_path), read(ta_path)
        require(pa["state"] == ta["state"] == "passed" and
                pa["class_rng_seed"] == ta["class_rng_seed"] == partition and
                pa["partition_sha256"] == ta["partition_sha256"] ==
                sha(split / "partition.npz") and
                ta["test_population_sha256"] ==
                sha(split / "test-population.npz"),
                f"partition {partition} full official test population")
        bind(pa_path)
        bind(ta_path)
        for seed in SEEDS:
            train_path = split / f"train-audit-seed{seed}/audit.json"
            infer_path = split / f"checkpoint-inference-audit-seed{seed}/audit.json"
            feature_path = split / f"feature-audit-seed{seed}/audit.json"
            score_path = split / f"score-audit-seed{seed}/audit.json"
            checkpoint = split / f"seed{seed}/final-best.pt"
            train, infer, feature, scored = (
                read(path) for path in
                (train_path, infer_path, feature_path, score_path))
            require(train["state"] == infer["state"] == feature["state"] ==
                    scored["state"] == "passed" and
                    train["class_seed"] == infer["class_seed"] == partition and
                    train["seed"] == infer["seed"] == feature["seed"] ==
                    scored["seed"] == seed and
                    train["training_epoch_count"] == 100 and
                    train["selected_checkpoint_sha256"] ==
                    infer["selected_checkpoint_sha256"] == sha(checkpoint) and
                    infer["training_audit_sha256"] == sha(train_path) and
                    infer["recomputed_accuracy"] ==
                    train["selected_id_checkpoint_accuracy"],
                    f"partition {partition} seed {seed} audited ID checkpoint")
            for path in (train_path, infer_path, feature_path, score_path,
                         checkpoint):
                bind(path)
            for stage in ("extract", "score"):
                folder = split / f"{stage}-seed{seed}"
                manifest_path = folder / "manifest.json"
                manifest = read(manifest_path)
                state = read(folder / "status.json")
                require(state["state"] == manifest["state"] and
                        state["manifest_sha256"] == sha(manifest_path) and
                        manifest["seed"] == seed and
                        manifest["checkpoint_sha256"] == sha(checkpoint),
                        f"partition {partition} seed {seed} {stage} stage")
                bind(manifest_path)
                bind(folder / "status.json")
                for filename, digest in manifest["outputs"].items():
                    bind(folder / filename, digest)
                for name, digest in manifest["source_hashes"].items():
                    bind(ROOT / name, digest)
                    bind(folder / "source_snapshot" / name, digest)
            raw_manifest = split / f"extract-seed{seed}/manifest.json"
            scored_manifest = split / f"score-seed{seed}/manifest.json"
            require(feature["extraction_manifest_sha256"] ==
                    scored["extraction_manifest_sha256"] == sha(raw_manifest) and
                    scored["scoring_manifest_sha256"] == sha(scored_manifest) and
                    scored["feature_audit_sha256"] == sha(feature_path),
                    f"partition {partition} seed {seed} raw/score audit binding")
            for stage in ("extract", "feature-audit", "score", "score-audit"):
                bind(split / f"{stage}-seed{seed}/class-provenance.json")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    destination = REPORT / "completion.json"
    hashes = validate()
    if args.write:
        require(not destination.exists(), "write-once completion exists")
        result = {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "class_seeds": [2031, 2032, 2033],
            "training_seeds": list(SEEDS),
            "id_test_count_per_partition": 8000,
            "ood_test_count_per_partition": 2000,
            "source_sha256": sha(Path(__file__)),
            "inputs_sha256": hashes,
            "boundary": "Descriptive three-fixed-partition nine-model sensitivity, fully source/raw/report audited; no OOD-based selection or partition-sampling CI.",
        }
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) +
                               "\n", encoding="utf-8")
    record = read(destination)
    require(record["state"] == "complete" and
            record["class_seeds"] == [2031, 2032, 2033] and
            record["training_seeds"] == list(SEEDS) and
            record["id_test_count_per_partition"] == 8000 and
            record["ood_test_count_per_partition"] == 2000 and
            record["source_sha256"] == sha(Path(__file__)) and
            record["inputs_sha256"] == hashes,
            "write-once completed phase hashes changed")
    print(json.dumps({"state": "verified_complete", "files": len(hashes),
                      "models": 9}))


if __name__ == "__main__":
    main()
