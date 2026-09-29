"""Report every method's fixed-20%-prior AP, keeping natural AP separately."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-prior-standardization"
OUT = ROOT / "report/class-holdout-prior-standardization"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
POPS = ("heldout_cifar100", "cifar10")
ORIENTATIONS = ("aupr_out", "aupr_in")


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


def main() -> None:
    require(not OUT.exists(), "write-once prior-standardized report exists")
    declaration = read(RUN / "declaration.json")
    manifest = read(RUN / "manifest.json")
    audit = read(RUN / "raw-audit.json")
    require(declaration["state"] == "frozen_before_20pct_standardized_ap_calculation" and
            manifest["state"] == "raw_complete_pending_independent_audit" and
            manifest["declaration_sha256"] == sha(RUN / "declaration.json") and
            manifest["raw_values_sha256"] == sha(RUN / "raw-values.npz") and
            audit["state"] == "passed" and
            audit["run_manifest_sha256"] == sha(RUN / "manifest.json") and
            audit["raw_values_sha256"] == sha(RUN / "raw-values.npz") and
            audit["checked"] == {"sklearn_weighted_ap": 144,
                                 "natural_ap": 144,
                                 "heldout_identity": 72,
                                 "weight_mass_checks": 72},
            "complete independently audited 20% prior AP values")
    with np.load(RUN / "raw-values.npz", allow_pickle=False) as source:
        fixed = source["standardized"]
        natural = source["original_natural"]
    require(fixed.shape == natural.shape == (3, 2, 6, 2, 2),
            "all three seeds, methods, populations and orientations")
    rows = []
    for vi, variant in enumerate(VARIANTS):
        for mi, method in enumerate(METHODS):
            for pi, population in enumerate(POPS):
                for oi, orientation in enumerate(ORIENTATIONS):
                    values = fixed[:, vi, mi, pi, oi]
                    original = natural[:, vi, mi, pi, oi]
                    rows.append({
                        "variant": variant, "method": method,
                        "population": population, "orientation": orientation,
                        "target_ood_prior": .2,
                        "n_id": 8000, "n_ood": 2000 if pi == 0 else 10000,
                        "seed0": float(values[0]), "seed1": float(values[1]),
                        "seed2": float(values[2]),
                        "mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)),
                        "original_natural_prior": .2 if pi == 0 else 10000 / 18000,
                        "original_natural_seed0": float(original[0]),
                        "original_natural_seed1": float(original[1]),
                        "original_natural_seed2": float(original[2]),
                        "original_natural_mean": float(original.mean()),
                        "original_natural_sample_sd": float(original.std(ddof=1)),
                    })
    require(len(rows) == 48, "all fixed prior AP rows")
    OUT.mkdir(parents=True)
    with (OUT / "statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "state": "report_pending_independent_audit",
        "target_ood_prior": .2,
        "seeds": [0, 1, 2], "variants": list(VARIANTS),
        "methods": list(METHODS), "populations": list(POPS),
        "orientations": list(ORIENTATIONS),
        "rows": rows,
        "source_sha256": {
            "declaration": sha(RUN / "declaration.json"),
            "run_manifest": sha(RUN / "manifest.json"),
            "raw_values": sha(RUN / "raw-values.npz"),
            "raw_audit": sha(RUN / "raw-audit.json"),
            "builder": sha(Path(__file__)),
        },
        "statistics_csv_sha256": sha(OUT / "statistics.csv"),
        "boundary": "Hypothetical 20% class-prior reweighting of all observed images; exploratory post-hoc, no method selection or unseen-population inference.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                               allow_nan=False) + "\n",
                                      encoding="utf-8")
    indexed = {(r["variant"], r["method"], r["population"], r["orientation"]): r
               for r in rows}
    energy = indexed["T1", "energy", "cifar10", "aupr_out"]
    lines = [
        "# Fixed-20%-prior precision-recall description",
        "",
        "All three already completed 80-class models, both temperature variants and",
        "all six unchanged scores are included. Every observed ID and OOD score is",
        "retained: the 8,000 ID rows each receive weight 0.8/8,000, while the 2,000",
        "held-out-class or 10,000 CIFAR-10 rows each receive 0.2/n_OOD. Grouped score",
        "ties contribute together to noninterpolated average precision. AUPR-in",
        "reverses the positive label and score sign while retaining these weights.",
        "",
        "This 20% OOD prior matches the observed held-out-class test mixture; it is a",
        "**hypothetical display prior**, not an estimated deployment prevalence.",
        "The full CIFAR-10 test population originally had 55.56% OOD prevalence.",
        f"For T=1 Energy, its natural-prior CIFAR-10 AUPR-out was {100*energy['original_natural_mean']:.2f}%",
        f"(seed SD {100*energy['original_natural_sample_sd']:.2f} pp); at the fixed 20% prior",
        f"it is {100*energy['mean']:.2f}% (seed SD {100*energy['sample_sd']:.2f} pp).",
        "Complete per-seed natural and standardized values are in `statistics.csv`.",
        "",
        "| Variant | Method | Held-out AP-out @20% | CIFAR-10 AP-out @20% | Held-out AP-in @20% | CIFAR-10 AP-in @20% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for method in METHODS:
            values = [indexed[variant, method, population, orientation]
                      for population, orientation in ((POPS[0], ORIENTATIONS[0]),
                                                      (POPS[1], ORIENTATIONS[0]),
                                                      (POPS[0], ORIENTATIONS[1]),
                                                      (POPS[1], ORIENTATIONS[1]))]
            shown = " | ".join(f"{100*r['mean']:.2f} ± {100*r['sample_sd']:.2f}%"
                                 for r in values)
            lines.append(f"| {variant} | {method} | {shown} |")
    lines += ["", "The held-out-class 20%-prior values reproduce its original natural",
              "AUPR. Reweighting changes only the precision-recall mixture arithmetic:",
              "it does not align semantic difficulty, data source, class splits or score",
              "distributions. Both OOD tests had already been seen, and CIFAR-10 was",
              "chosen after the held-out-class result. This is exploratory description,",
              "not independent confirmation, a detector ranking, a confidence interval,",
              "or a causal comparison of dataset shifts.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": summary["state"], "rows": len(rows),
                      "energy_c10_aupr_out_20pct": energy["mean"]}))


if __name__ == "__main__":
    main()
