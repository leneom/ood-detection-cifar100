"""Build the frozen Fashion-MNIST three-seed extension without tuning."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "fashionmnist-extension"
OUT = ROOT / "report" / "fashionmnist-extension"
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
TARGETS = (80, 90, 95, 99)
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id", "id_tpr_at_fpr95",
           "benchmark_threshold", "validation_threshold",
           "id_test_acceptance_at_validation_threshold",
           "ood_acceptance_at_validation_threshold", "calibration_id_acceptance",
           "ood_prevalence")
RATES = ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")
SOURCE = "tools/build_fashionmnist_report.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    points = [float(v) for v in values]
    if len(points) != 3 or not np.isfinite(points).all():
        raise ValueError("exactly three finite seed points required")
    return {"seeds": points, "mean": float(np.mean(points)),
            "sample_sd": float(np.std(points, ddof=1))}


def percent(record):
    return f"{100*record['mean']:.2f} ± {100*record['sample_sd']:.2f}"


def plot(summary, metric, filename, title, xlabel):
    fig, ax = plt.subplots(figsize=(9.5, 5.1), constrained_layout=True)
    ys = np.arange(len(METHODS))
    names = {"msp": "MSP", "energy": "Energy", "maxlogit": "MaxLogit",
             "entropy": "Entropy", "knn": "KNN-50", "mahalanobis": "Pooled Mahalanobis"}
    colors = {"T1": "#247ba0", "calibrated": "#dc5f3b"}
    for variant, offset in (("T1", -.12), ("calibrated", .12)):
        records = [summary["results"][variant][method] for method in METHODS]
        if metric == "auroc":
            vals = [record["metrics"][metric] for record in records]
        else:
            vals = [record["operating_grid"]["target95"][metric] for record in records]
        means = [100*value["mean"] for value in vals]
        errors = [100*value["sample_sd"] for value in vals]
        ax.errorbar(means, ys + offset, xerr=errors, fmt="o", ms=6, capsize=3,
                    color=colors[variant], label=variant, linewidth=1.5)
    ax.set_yticks(ys, [names[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    ax.set_xlim(0, 100)
    fig.savefig(OUT / f"{filename}.png", dpi=180)
    fig.savefig(OUT / f"{filename}.svg")
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError("completed Fashion-MNIST report directory exists")
    manifests = []
    seed_records = []
    for seed in range(3):
        scored = RUN / f"score-seed{seed}"
        manifest = read(scored / "manifest.json")
        audit = read(RUN / f"score-audit-seed{seed}" / "audit.json")
        extracted = read(RUN / f"extract-seed{seed}" / "manifest.json")
        if not (audit["state"] == "passed" and audit["seed"] == seed and
                audit["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
                manifest["seed"] == seed and manifest["n_id_test"] == 10000 and
                manifest["n_ood"] == 10000 and extracted["checkpoint_sha256"] ==
                manifest["checkpoint_sha256"]):
            raise ValueError("all three scored seeds need bound passing audits")
        manifests.append(manifest)
        seed_records.append({"seed": seed, "checkpoint_epoch": extracted["checkpoint_epoch"],
                             "checkpoint_sha256": manifest["checkpoint_sha256"],
                             "temperature": manifest["temperature"],
                             "scoring_manifest_sha256": sha(scored / "manifest.json"),
                             "score_audit_sha256": sha(RUN / f"score-audit-seed{seed}" / "audit.json"),
                             "raw_logits_features": f"runs/fashionmnist-extension/extract-seed{seed}/raw.npz",
                             "per_image_scores": f"runs/fashionmnist-extension/score-seed{seed}/scores.npz",
                             "acceptance_masks": f"runs/fashionmnist-extension/score-seed{seed}/acceptance_masks.npz"})
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for method in METHODS:
            per_seed = [m["results"][variant][method] for m in manifests]
            metrics = {name: stats([record["metrics"][name] for record in per_seed])
                       for name in METRICS}
            operating = {}
            for target in TARGETS:
                key = f"target{target}"
                operating[key] = {name: stats([record["operating_grid"][key][name]
                                               for record in per_seed]) for name in RATES}
                operating[key]["thresholds"] = [float(record["operating_grid"][key]["threshold"])
                                                  for record in per_seed]
                operating[key]["calibration_accepted"] = [int(record["operating_grid"][key]["calibration_accepted"])
                                                         for record in per_seed]
                operating[key]["id_test_accepted"] = [int(record["operating_grid"][key]["id_test_accepted"])
                                                     for record in per_seed]
                operating[key]["ood_accepted"] = [int(record["operating_grid"][key]["ood_accepted"])
                                                 for record in per_seed]
                operating[key]["calibration_ties_at_threshold"] = [int(record["operating_grid"][key]["calibration_ties_at_threshold"])
                                                                  for record in per_seed]
            results[variant][method] = {"metrics": metrics, "operating_grid": operating}
    OUT.mkdir(parents=True)
    summary = {"state": "three_seed_sequential_extension_scored_and_audited",
               "dataset": "Fashion-MNIST official test, full 10000, 28-to-32 bilinear gray-to-RGB",
               "id_test_count": 10000, "ood_count": 10000, "ood_prevalence": .5,
               "method_order": METHODS, "variants": VARIANTS, "targets": TARGETS,
               "seed_records": seed_records, "results": results,
               "protocol_sha256": sha(ROOT / "docs/FASHIONMNIST_EXTENSION.md"),
               "cache_sha256": sha(ROOT / "data/fashionmnist-test-32x32.npz"),
               "source_sha256": sha(ROOT / SOURCE),
               "created_at": datetime.now(timezone.utc).isoformat(),
               "boundary": "Sequentially selected additional OOD population; methods, models, ID fits and thresholds frozen before its data access; no method selection."}
    write(OUT / "summary.json", summary)
    lines = ["# Fashion-MNIST frozen population extension", "",
             "All 10,000 official Fashion-MNIST test images were scored against the same 10,000 CIFAR-100 ID-test images. This population was selected sequentially after viewing the Tiny ImageNet results, then its full protocol was frozen before any Fashion image access. Three isolated classifiers (selected epochs 92/96/98), their ID-only temperatures and thresholds, and all six original methods were reused. No Fashion image was used to select a model, fit a detector or choose a threshold.", "",
             "Each table value is the mean ± sample SD across three training seeds, in percent. Individual seed points and exact threshold counts are in [summary.json](summary.json). AUROC and AUPR use 10,000 ID + 10,000 OOD samples, so OOD prevalence is 50%. FPR95-ID uses the ID **test** order statistic as a benchmark operating point; fixed95 OOD/ID acceptance uses the pre-existing 1,500-row ID-validation threshold.", ""]
    display = {"msp": "MSP", "energy": "Energy", "maxlogit": "MaxLogit",
               "entropy": "Entropy", "knn": "KNN-50", "mahalanobis": "Pooled Mahalanobis"}
    for variant in VARIANTS:
        lines += [f"## {variant}", "", "| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            row = results[variant][method]
            vals = [row["metrics"][name] for name in ("auroc", "aupr_out", "aupr_in", "fpr95_id")]
            vals += [row["operating_grid"]["target95"][name] for name in ("ood_acceptance", "id_test_acceptance")]
            lines.append("| " + display[method] + " | " + " | ".join(percent(v) for v in vals) + " |")
        lines += ["", "Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):", "",
                  "| Method | 80% | 90% | 95% | 99% |", "| --- | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            vals = [results[variant][method]["operating_grid"][f"target{t}"]["ood_acceptance"]
                    for t in TARGETS]
            lines.append("| " + display[method] + " | " + " | ".join(percent(v) for v in vals) + " |")
        lines.append("")
    lines += ["## Evidence and scope", "",
              "Per-image raw logits/features, predictions, official labels and rows, twelve score vectors and all 80/90/95/99 ID/OOD acceptance masks are retained locally for each seed. The image cache and feature scores were independently audited against all decoded IDX pixels and refitted training-ID feature banks. Score audits independently recomputed all formulas, 468 metric fields, 432 masks and fixed threshold counts across three seeds. [Completion manifest](completion.json) binds the report, figures, audits and regression evidence.", "",
              "Fashion-MNIST is grayscale 28×28 product imagery converted to 32×32 RGB. Its source, color and resolution differ sharply from CIFAR-100, so these measurements cannot isolate semantic novelty, establish universal generalization or reproduce an OpenOOD leaderboard. This sequential extension is not added to the previously fixed four-population macro. All six methods and both temperature variants remain in frozen order, with no post-test winner selection. Sample SD reflects three trained classifier seeds, not image-level confidence intervals.", "",
              "Figures: [AUROC](auroc.png) and [fixed95 OOD acceptance](fixed95-acceptance.png).", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    plot(summary, "auroc", "auroc", "Fashion-MNIST OOD discrimination (3 seeds)", "AUROC (%)")
    plot(summary, "ood_acceptance", "fixed95-acceptance",
         "OOD accepted at frozen 95% ID-validation target (3 seeds)", "OOD acceptance (%)")
    print(json.dumps({"state": summary["state"], "cells": 12, "seed_points": 36,
                      "report": str(OUT / "REPORT.md")}))


if __name__ == "__main__":
    main()
