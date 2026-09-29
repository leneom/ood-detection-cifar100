"""Create or read-only verify the local Tiny ImageNet completion manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "tinyimagenet-confirmation"
OUT = ROOT / "report" / "tinyimagenet-confirmation"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def inputs():
    names = [
        "docs/TINYIMAGENET_CONFIRMATION.md",
        "docs/TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md",
        "report/validation-isolation/completion.json",
        "data/tiny-imagenet-200.zip",
        "data/tinyimagenet-validation.npz",
        "runs/tinyimagenet-confirmation/declaration.json",
        "runs/tinyimagenet-confirmation/amendment-1.json",
        "runs/tinyimagenet-confirmation/download-status.json",
        "runs/tinyimagenet-confirmation/cache-status.json",
        "runs/tinyimagenet-confirmation/cache-manifest.json",
        "runs/tinyimagenet-confirmation/cache-audit.json",
        "runs/tinyimagenet-confirmation/extraction-launch.json",
        "runs/tinyimagenet-confirmation/extraction-status.json",
        "runs/tinyimagenet-confirmation/study-status.json",
        "report/tinyimagenet-confirmation/summary.json",
        "report/tinyimagenet-confirmation/REPORT.md",
        "report/tinyimagenet-confirmation/auroc.png",
        "report/tinyimagenet-confirmation/auroc.svg",
        "report/tinyimagenet-confirmation/fixed95-acceptance.png",
        "report/tinyimagenet-confirmation/fixed95-acceptance.svg",
        "report/tinyimagenet-confirmation/display-audit.json",
        "report/tinyimagenet-confirmation/figure-inspection.json",
        "report/tinyimagenet-confirmation/regression.log",
        "tools/download_tinyimagenet.py",
        "tools/prepare_tinyimagenet.py",
        "tools/audit_tinyimagenet_cache.py",
        "tools/extract_tinyimagenet.py",
        "tools/run_tinyimagenet_extraction.py",
        "tools/audit_tinyimagenet_features.py",
        "tools/score_tinyimagenet.py",
        "tools/audit_tinyimagenet_scores.py",
        "tools/build_tinyimagenet_report.py",
        "tools/audit_tinyimagenet_report.py",
        "tools/verify_tinyimagenet_completion.py",
    ]
    for seed in range(3):
        names += [
            f"runs/tinyimagenet-confirmation/extract-seed{seed}/status.json",
            f"runs/tinyimagenet-confirmation/extract-seed{seed}/manifest.json",
            f"runs/tinyimagenet-confirmation/extract-seed{seed}/raw.npz",
            f"runs/tinyimagenet-confirmation/extract-seed{seed}/T1_scores.npz",
            f"runs/tinyimagenet-confirmation/extract-seed{seed}/feature_scores.npz",
            f"runs/tinyimagenet-confirmation/feature-audit-seed{seed}/audit.json",
            f"runs/tinyimagenet-confirmation/score-seed{seed}/status.json",
            f"runs/tinyimagenet-confirmation/score-seed{seed}/manifest.json",
            f"runs/tinyimagenet-confirmation/score-seed{seed}/scores.npz",
            f"runs/tinyimagenet-confirmation/score-seed{seed}/acceptance_masks.npz",
            f"runs/tinyimagenet-confirmation/score-audit-seed{seed}/audit.json",
            f"runs/validation-isolation/score-seed{seed}/manifest.json",
        ]
    return names


def validate():
    declaration = read(RUN / "declaration.json")
    amendment = read(RUN / "amendment-1.json")
    cache = read(RUN / "cache-audit.json")
    summary = read(OUT / "summary.json")
    display = read(OUT / "display-audit.json")
    visual = read(OUT / "figure-inspection.json")
    study = read(RUN / "study-status.json")
    require(declaration["protocol_sha256"] == sha(ROOT / "docs/TINYIMAGENET_CONFIRMATION.md") and
            declaration["prior_completion_sha256"] == sha(ROOT / "report/validation-isolation/completion.json"),
            "frozen prior declaration")
    require(amendment["original_declaration_sha256"] == sha(RUN / "declaration.json") and
            amendment["amendment_sha256"] == sha(ROOT / "docs/TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md"),
            "frozen predecode amendment")
    require(cache["state"] == "passed" and cache["count"] == 10000 and
            cache["cache_sha256"] == sha(ROOT / "data/tinyimagenet-validation.npz"),
            "independent full image cache audit")
    require(summary["state"] == "three_seed_new_population_scored_and_audited" and
            display["state"] == visual["state"] == "passed" and
            display["summary_sha256"] == sha(OUT / "summary.json") and
            display["report_sha256"] == sha(OUT / "REPORT.md") and
            study["state"] == "complete" and study["completed_seeds"] == [0, 1, 2],
            "three-seed display/study status")
    for name, details in visual["images"].items():
        require(details["sha256"] == sha(OUT / name), "visually inspected figure changed")
    regression = (OUT / "regression.log").read_text(encoding="utf-8")
    require("Ran 91 tests" in regression and "OK (skipped=2)" in regression and
            "FAILED" not in regression, "full regression evidence")
    for seed in range(3):
        extracted = RUN / f"extract-seed{seed}"
        scored = RUN / f"score-seed{seed}"
        e, s = read(extracted / "manifest.json"), read(scored / "manifest.json")
        fa = read(RUN / f"feature-audit-seed{seed}" / "audit.json")
        sa = read(RUN / f"score-audit-seed{seed}" / "audit.json")
        require(e["seed"] == s["seed"] == seed and e["count"] == s["n_ood"] == 10000 and
                s["n_id_test"] == 10000 and s["checkpoint_sha256"] == e["checkpoint_sha256"] and
                fa["state"] == sa["state"] == "passed" and
                fa["extraction_manifest_sha256"] == sha(extracted / "manifest.json") and
                sa["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
                s["prior_id_score_manifest_sha256"] ==
                sha(ROOT / f"runs/validation-isolation/score-seed{seed}/manifest.json"),
                f"seed {seed} independent source bindings")
    return {name: sha(ROOT / name) for name in inputs()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = OUT / "completion.json"
    if args.write:
        require(not path.exists(), "completion manifest already exists")
        hashes = validate()
        record = {"state": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "population": "Tiny ImageNet-200 official validation, 10000 images",
                  "seeds": [0, 1, 2], "primary_cells": 12,
                  "independent_score_metric_fields": 468,
                  "independent_score_masks": 432,
                  "independent_summary_seed_values": 828,
                  "regression": {"total": 91, "passed": 89, "skipped": 2, "exit_code": 0},
                  "inputs_sha256": hashes,
                  "boundary": "One previously unscored OOD population; no OOD-based fitting/selection; no leaderboard or universal-generalization claim."}
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

