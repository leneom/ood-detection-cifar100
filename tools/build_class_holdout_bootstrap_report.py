"""Summarize the frozen, independently audited class-holdout image bootstrap."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cifar100-class-holdout-bootstrap"
REPORT = ROOT / "report/cifar100-class-holdout-bootstrap"
BASELINE = ROOT / "report/cifar100-class-holdout/summary.json"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
FIELDS = ("auroc", "aupr_out", "aupr_in", "fpr95_id",
          "id_test_acceptance_target95", "ood_acceptance_target95")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(not REPORT.exists(), "write-once bootstrap report already exists")
    declaration = read(RUN / "declaration.json")
    audit = read(RUN / "raw-audit.json")
    require(audit["state"] == "passed" and
            audit["declaration_sha256"] == sha(RUN / "declaration.json") and
            audit["launch_sha256"] == sha(RUN / "launch.json") and
            audit["status_sha256"] == sha(RUN / "status.json") and
            declaration["replicates"] == 1000,
            "complete source-bound independent bootstrap audit required")
    status = read(RUN / "status.json")
    require(status["state"] == "raw_complete_pending_independent_audit" and
            status["completed_replicates"] == 1000,
            "all 1000 draws required")
    baseline = read(BASELINE)
    require(baseline["seeds"] == [0, 1, 2] and
            baseline["variants"] == list(VARIANTS) and
            baseline["methods"] == list(METHODS),
            "complete original baseline required")
    cells = {(cell["variant"], cell["method"]): cell
             for cell in baseline["cells"]}
    require(len(cells) == 12, "all fixed baseline cells required")
    ranking = []
    ids = []
    oods = []
    for block in range(10):
        stem = f"block-{block:03d}"
        path = RUN / "blocks" / f"{stem}-metrics.npz"
        marker = read(RUN / "blocks" / f"{stem}.json")
        require(marker["metrics_sha256"] == sha(path) and
                audit["block_hashes"][path.relative_to(ROOT).as_posix()] == sha(path),
                "independently audited block hash changed")
        with np.load(path, allow_pickle=False) as data:
            require(np.array_equal(data["replicates"],
                                   np.arange(block * 100, (block + 1) * 100)),
                    "replicate sequence")
            ranking.append(data["ranking"])
            ids.append(data["id_accepted_counts"])
            oods.append(data["ood_accepted_counts"])
    all_ranking = np.concatenate(ranking)
    all_id = np.concatenate(ids) / 8000.0
    all_ood = np.concatenate(oods) / 2000.0
    require(all_ranking.shape == (1000, 3, 2, 6, 4) and
            all_id.shape == all_ood.shape == (1000, 3, 2, 6) and
            np.isfinite(all_ranking).all() and
            np.isfinite(all_id).all() and np.isfinite(all_ood).all(),
            "complete finite 1000-replicate arrays")
    rows: list[dict] = []
    for vi, variant in enumerate(VARIANTS):
        for mi, method in enumerate(METHODS):
            original = cells[variant, method]
            for fi, field in enumerate(FIELDS):
                if fi < 4:
                    draws = all_ranking[:, :, vi, mi, fi]
                    reference = original["metrics"][field]
                else:
                    draws = (all_id if fi == 4 else all_ood)[:, :, vi, mi]
                    source = "id_test_acceptance" if fi == 4 else "ood_acceptance"
                    reference = original["operating"]["target95"][source]
                points = np.asarray(reference["per_seed"], dtype=np.float64)
                require(points.shape == (3,) and np.isfinite(points).all() and
                        abs(points.mean() - reference["mean"]) < 1e-12 and
                        abs(points.std(ddof=1) - reference["sample_sd"]) < 1e-12,
                        "original seed statistics")
                paired_model_mean = draws.mean(axis=1)
                q = np.quantile(paired_model_mean, [0.025, 0.5, 0.975],
                                method="linear")
                rows.append({
                    "variant": variant, "method": method, "field": field,
                    "original_seed0": float(points[0]),
                    "original_seed1": float(points[1]),
                    "original_seed2": float(points[2]),
                    "original_mean": float(points.mean()),
                    "training_seed_sample_sd": float(points.std(ddof=1)),
                    "paired_image_bootstrap_q025": float(q[0]),
                    "paired_image_bootstrap_median": float(q[1]),
                    "paired_image_bootstrap_q975": float(q[2]),
                    "bootstrap_draws": 1000,
                })
    require(len(rows) == 72, "six fields for every fixed method and variant")
    REPORT.mkdir(parents=True)
    with (REPORT / "statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details = {
        "state": "report_pending_independent_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Post-hoc paired, within-class image sampling conditional on the fixed observed classes and split; no formal coverage claim or test-based model selection.",
        "seeds": [0, 1, 2], "variants": list(VARIANTS),
        "methods": list(METHODS), "fields": list(FIELDS),
        "replicates": 1000, "id_images_per_draw": 8000,
        "ood_images_per_draw": 2000, "ood_prevalence": 0.2,
        "rows": rows,
        "source_sha256": {
            "declaration": sha(RUN / "declaration.json"),
            "launch": sha(RUN / "launch.json"),
            "raw_audit": sha(RUN / "raw-audit.json"),
            "baseline_summary": sha(BASELINE),
            "report_builder": sha(Path(__file__)),
        },
        "statistics_csv_sha256": sha(REPORT / "statistics.csv"),
    }
    (REPORT / "summary.json").write_text(json.dumps(details, indent=2,
                                                  allow_nan=False) + "\n",
                                         encoding="utf-8")
    lines = [
        "# CIFAR-100 same-source class holdout: paired image bootstrap",
        "",
        "Post-hoc description of the completed three-seed 80-ID/20-OOD-class study.",
        "The 1,000 paired draws resample 100 test images within each of the 100 observed",
        "fine classes. The same draw is shared by all models and methods. Every draw",
        "retains 8,000 ID and 2,000 OOD images. This interval describes image sampling",
        "conditional on these classes, this class split and these three trained models;",
        "it is not a confidence interval for unseen classes or new training seeds.",
        "Training-seed sample SD and paired image bootstrap quantiles are separate.",
        "All original thresholds and scores are frozen. No method winner or test-based",
        "setting is selected. Complete numerical values are in `statistics.csv`;",
        "all 1,000 raw draws and metric arrays remain in the immutable run blocks.",
        "",
        "| Variant | Method | AUROC original mean ± seed SD | Paired image q2.5–q97.5 | OOD acceptance at original ID-val 95% target: mean ± seed SD | Paired image q2.5–q97.5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    indexed = {(row["variant"], row["method"], row["field"]): row for row in rows}
    for variant in VARIANTS:
        for method in METHODS:
            auroc = indexed[variant, method, "auroc"]
            accept = indexed[variant, method, "ood_acceptance_target95"]
            lines.append(
                f"| {variant} | {method} | {100*auroc['original_mean']:.2f} ± {100*auroc['training_seed_sample_sd']:.2f}% | "
                f"{100*auroc['paired_image_bootstrap_q025']:.2f}–{100*auroc['paired_image_bootstrap_q975']:.2f}% | "
                f"{100*accept['original_mean']:.2f} ± {100*accept['training_seed_sample_sd']:.2f}% | "
                f"{100*accept['paired_image_bootstrap_q025']:.2f}–{100*accept['paired_image_bootstrap_q975']:.2f}% |"
            )
    lines += ["", "The original AUROC and OOD acceptance refer to the full official test",
              "population; the bootstrap quantiles refer to the fixed three-model mean.",
              "FPR95-ID is recomputed from each draw's ID test order statistic, while",
              "fixed95 acceptance always uses the original disjoint ID-validation threshold.",
              "The same images underlie the held-out-class heterogeneity analysis; these",
              "descriptive analyses are not independent external confirmations.", ""]
    (REPORT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": details["state"], "rows": len(rows),
                      "report": str(REPORT)}))


if __name__ == "__main__":
    main()
