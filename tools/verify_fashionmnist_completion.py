"""Write once or read-only verify the complete local Fashion-MNIST extension."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "fashionmnist-extension"
OUT = ROOT / "report" / "fashionmnist-extension"


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


def inputs() -> list[str]:
    names = [
        "docs/FASHIONMNIST_EXTENSION.md",
        "docs/FASHIONMNIST_RUNBOOK.md",
        "report/validation-isolation/completion.json",
        "report/tinyimagenet-confirmation/completion.json",
        "report/tinyimagenet-class-heterogeneity/completion.json",
        "data/fashionmnist-test/t10k-images-idx3-ubyte.gz",
        "data/fashionmnist-test/t10k-labels-idx1-ubyte.gz",
        "data/fashionmnist-test-32x32.npz",
        "runs/fashionmnist-extension/declaration.json",
        "runs/fashionmnist-extension/download-status.json",
        "runs/fashionmnist-extension/cache-manifest.json",
        "runs/fashionmnist-extension/cache-audit.json",
        "runs/fashionmnist-extension/extraction-launch.json",
        "runs/fashionmnist-extension/extraction-status.json",
        "runs/fashionmnist-extension/recovery-launch.json",
        "runs/fashionmnist-extension/recovery-status.json",
        "runs/fashionmnist-extension/cpu-preflight.json",
        "runs/fashionmnist-extension/interrupted-attempt-1/extract-seed0/status.json",
        "report/fashionmnist-extension/summary.json",
        "report/fashionmnist-extension/REPORT.md",
        "report/fashionmnist-extension/auroc.png",
        "report/fashionmnist-extension/auroc.svg",
        "report/fashionmnist-extension/fixed95-acceptance.png",
        "report/fashionmnist-extension/fixed95-acceptance.svg",
        "report/fashionmnist-extension/display-audit.json",
        "report/fashionmnist-extension/figure-inspection.json",
        "report/fashionmnist-extension/regression.log",
        "tools/download_fashionmnist.py",
        "tools/prepare_fashionmnist.py",
        "tools/audit_fashionmnist_cache.py",
        "tools/extract_fashionmnist.py",
        "tools/run_fashionmnist_extraction.py",
        "tools/run_fashionmnist_extraction_recovery.py",
        "tools/preflight_fashionmnist_extraction.py",
        "tools/audit_fashionmnist_features.py",
        "tools/score_fashionmnist.py",
        "tools/audit_fashionmnist_scores.py",
        "tools/build_fashionmnist_report.py",
        "tools/audit_fashionmnist_report.py",
        "tools/verify_fashionmnist_completion.py",
    ]
    for seed in range(3):
        names += [
            f"runs/fashionmnist-extension/extract-seed{seed}/status.json",
            f"runs/fashionmnist-extension/extract-seed{seed}/manifest.json",
            f"runs/fashionmnist-extension/extract-seed{seed}/raw.npz",
            f"runs/fashionmnist-extension/extract-seed{seed}/T1_scores.npz",
            f"runs/fashionmnist-extension/extract-seed{seed}/feature_scores.npz",
            f"runs/fashionmnist-extension/feature-audit-seed{seed}/audit.json",
            f"runs/fashionmnist-extension/score-seed{seed}/status.json",
            f"runs/fashionmnist-extension/score-seed{seed}/manifest.json",
            f"runs/fashionmnist-extension/score-seed{seed}/scores.npz",
            f"runs/fashionmnist-extension/score-seed{seed}/acceptance_masks.npz",
            f"runs/fashionmnist-extension/score-audit-seed{seed}/audit.json",
            f"runs/validation-isolation/score-seed{seed}/manifest.json",
        ]
    return names


def validate() -> dict[str, str]:
    declaration = read(RUN / "declaration.json")
    download = read(RUN / "download-status.json")
    cache = read(RUN / "cache-audit.json")
    preflight = read(RUN / "cpu-preflight.json")
    original = read(RUN / "extraction-status.json")
    recovery_launch = read(RUN / "recovery-launch.json")
    recovery = read(RUN / "recovery-status.json")
    summary = read(OUT / "summary.json")
    display = read(OUT / "display-audit.json")
    visual = read(OUT / "figure-inspection.json")
    require(declaration["state"] == "frozen_before_any_fashion_data_access" and
            declaration["protocol_sha256"] == sha(ROOT / "docs/FASHIONMNIST_EXTENSION.md") and
            declaration["previous_aggregate_completion_sha256"] ==
            sha(ROOT / "report/tinyimagenet-confirmation/completion.json") and
            declaration["previous_diagnostic_completion_sha256"] ==
            sha(ROOT / "report/tinyimagenet-class-heterogeneity/completion.json"),
            "pre-access frozen declaration")
    require(download["state"] == "verified_test_archives_downloaded" and
            download["protocol_sha256"] == declaration["protocol_sha256"],
            "official test download declaration")
    for name, record in download["files"].items():
        require(record["sha256"] == sha(ROOT / "data/fashionmnist-test" / name),
                "official archive changed: " + name)
    require(cache["state"] == "passed" and cache["count"] == 10000 and
            cache["independent_numpy_pixels_checked"] == 10000 and
            cache["cifar100_exact_pixel_duplicates"] == 0 and
            cache["cache_sha256"] == sha(ROOT / "data/fashionmnist-test-32x32.npz") and
            cache["manifest_sha256"] == sha(RUN / "cache-manifest.json"),
            "independent full image cache audit")
    require(preflight["state"] == "passed_cpu_only" and
            preflight["source_sha256"] == sha(ROOT / "tools/preflight_fashionmnist_extraction.py") and
            preflight["original_extraction_launch_sha256"] ==
            sha(RUN / "extraction-launch.json") and
            preflight["fashion_cache_sha256"] == cache["cache_sha256"] and
            [entry["seed"] for entry in preflight["seeds"]] == [0, 1, 2],
            "three-seed frozen input preflight")
    require(original["state"] == "extraction_failed_preserve_partial" and
            recovery_launch["original_launch_sha256"] == sha(RUN / "extraction-launch.json") and
            recovery_launch["original_failure_sha256"] == sha(RUN / "extraction-status.json") and
            recovery_launch["archived_partial_status_sha256"] ==
            sha(RUN / "interrupted-attempt-1/extract-seed0/status.json") and
            recovery_launch["extract_source_sha256"] == sha(ROOT / "tools/extract_fashionmnist.py") and
            recovery["state"] == "three_seeds_extracted_pending_audit_and_scoring" and
            recovery["completed_seeds"] == [0, 1, 2] and
            recovery["recovery_launch_sha256"] == sha(RUN / "recovery-launch.json"),
            "single guarded recovery and preserved failed attempt")
    require(summary["state"] == "three_seed_sequential_extension_scored_and_audited" and
            summary["id_test_count"] == summary["ood_count"] == 10000 and
            summary["protocol_sha256"] == declaration["protocol_sha256"] and
            summary["cache_sha256"] == cache["cache_sha256"] and
            display["state"] == visual["state"] == "passed" and
            display["summary_sha256"] == sha(OUT / "summary.json") and
            display["report_sha256"] == sha(OUT / "REPORT.md"),
            "three-seed audited report")
    require(set(visual["images"]) == {"auroc.png", "fixed95-acceptance.png"},
            "both actual figures inspected")
    for name, details in visual["images"].items():
        require(details["sha256"] == display["figure_sha256"][name] ==
                sha(OUT / name), "inspected figure changed: " + name)
    for name, digest in display["figure_sha256"].items():
        require(digest == sha(OUT / name), "display-audited figure changed: " + name)
    regression = (OUT / "regression.log").read_text(encoding="utf-8")
    require("Ran 91 tests" in regression and "OK (skipped=2)" in regression and
            "FAILED" not in regression, "full regression evidence")
    for seed in range(3):
        extracted = RUN / f"extract-seed{seed}"
        scored = RUN / f"score-seed{seed}"
        e, s = read(extracted / "manifest.json"), read(scored / "manifest.json")
        fa = read(RUN / f"feature-audit-seed{seed}" / "audit.json")
        sa = read(RUN / f"score-audit-seed{seed}" / "audit.json")
        frozen = preflight["seeds"][seed]
        require(frozen["seed"] == seed and
                frozen["checkpoint_epoch"] == e["checkpoint_epoch"] and
                frozen["checkpoint_sha256"] == sha(
                    ROOT / f"runs/validation-isolation/seed{seed}/final-best.pt") and
                frozen["id_train_bank_sha256"] == sha(
                    ROOT / f"runs/validation-isolation/extract-seed{seed}/id_train_raw.npz") and
                frozen["id_score_manifest_sha256"] == sha(
                    ROOT / f"runs/validation-isolation/score-seed{seed}/manifest.json") and
                frozen["id_score_audit_sha256"] == sha(
                    ROOT / f"runs/validation-isolation/audit-seed{seed}/audit.json"),
                f"seed {seed} preflight source changed")
        require(e["seed"] == s["seed"] == seed and
                e["count"] == s["n_ood"] == s["n_id_test"] == 10000 and
                e["checkpoint_sha256"] == s["checkpoint_sha256"] and
                e["cache_sha256"] == cache["cache_sha256"] and
                e["isolation_completion_sha256"] ==
                sha(ROOT / "report/validation-isolation/completion.json") and
                e["prior_tiny_completion_sha256"] ==
                sha(ROOT / "report/tinyimagenet-confirmation/completion.json") and
                e["id_score_manifest_sha256"] ==
                sha(ROOT / f"runs/validation-isolation/score-seed{seed}/manifest.json") and
                fa["state"] == sa["state"] == "passed" and
                fa["extraction_manifest_sha256"] == sha(extracted / "manifest.json") and
                sa["extraction_manifest_sha256"] == sha(extracted / "manifest.json") and
                sa["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
                s["prior_id_score_manifest_sha256"] ==
                sha(ROOT / f"runs/validation-isolation/score-seed{seed}/manifest.json"),
                f"seed {seed} independent source bindings")
    return {name: sha(ROOT / name) for name in inputs()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = OUT / "completion.json"
    if args.write:
        require(not path.exists(), "completion manifest already exists")
        hashes = validate()
        record = {
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "population": "Fashion-MNIST official test, all 10000 images",
            "seeds": [0, 1, 2],
            "primary_cells": 12,
            "regression": {"total": 91, "passed": 89, "skipped": 2, "exit_code": 0},
            "inputs_sha256": hashes,
            "boundary": "Sequentially chosen extra OOD population; no OOD fitting, method selection, leaderboard or multi-population confirmation claim.",
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record = read(path)
    require(record["state"] == "complete" and record["seeds"] == [0, 1, 2],
            "completion header")
    actual = validate()
    require(record["inputs_sha256"] == actual, "completion hash-bound inputs changed")
    print(json.dumps({"state": "verified_complete", "files": len(actual),
                      "seeds": record["seeds"]}))


if __name__ == "__main__":
    main()
