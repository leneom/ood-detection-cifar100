"""Independent full-value/CSV/display audit of fixed-prior AP report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
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


def close(actual: float, expected: float, message: str) -> None:
    require(math.isfinite(float(actual)) and
            abs(float(actual) - float(expected)) < 1e-12,
            f"{message}: {actual} vs {expected}")


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    points = [float(value) for value in values]
    mean = math.fsum(points) / 3
    sample_sd = math.sqrt(math.fsum((point - mean) ** 2 for point in points) / 2)
    return mean, sample_sd


def main() -> None:
    output = OUT / "audit.json"
    require(not output.exists(), "write-once prior-standardized report audit exists")
    summary = read(OUT / "summary.json")
    raw_audit = read(RUN / "raw-audit.json")
    require(summary["state"] == "report_pending_independent_audit" and
            summary["target_ood_prior"] == .2 and
            summary["seeds"] == [0, 1, 2] and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            summary["populations"] == list(POPS) and
            summary["orientations"] == list(ORIENTATIONS) and
            raw_audit["state"] == "passed" and
            raw_audit["checked"] == {"sklearn_weighted_ap": 144,
                                     "natural_ap": 144,
                                     "heldout_identity": 72,
                                     "weight_mass_checks": 72},
            "complete independently audited 20% AP report schema")
    sources = {
        "declaration": RUN / "declaration.json",
        "run_manifest": RUN / "manifest.json",
        "raw_values": RUN / "raw-values.npz",
        "raw_audit": RUN / "raw-audit.json",
        "builder": ROOT / "tools/build_class_holdout_prior_report.py",
    }
    require(summary["source_sha256"] ==
            {name: sha(path) for name, path in sources.items()} and
            summary["statistics_csv_sha256"] == sha(OUT / "statistics.csv"),
            "report source/CSV hashes")
    with np.load(RUN / "raw-values.npz", allow_pickle=False) as source:
        fixed = source["standardized"]
        natural = source["original_natural"]
    require(fixed.shape == natural.shape == (3, 2, 6, 2, 2),
            "all frozen raw AP values")
    with (OUT / "statistics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    rows = summary["rows"]
    require(len(rows) == len(csv_rows) == 48,
            "both populations/orientations, all methods/variants")
    checked = {"standardized_seed_values": 0, "natural_seed_values": 0,
               "statistical_fields": 0, "csv_fields": 0,
               "display_rows": 0, "heldout_identities": 0}
    index = 0
    for vi, variant in enumerate(VARIANTS):
        for mi, method in enumerate(METHODS):
            for pi, population in enumerate(POPS):
                for oi, orientation in enumerate(ORIENTATIONS):
                    row, exported = rows[index], csv_rows[index]
                    index += 1
                    require((row["variant"], row["method"], row["population"],
                             row["orientation"]) ==
                            (variant, method, population, orientation) and
                            set(row) == set(exported) and
                            [exported[k] for k in ("variant", "method", "population", "orientation")] ==
                            [variant, method, population, orientation],
                            "complete ordered all-method report and CSV")
                    weights = fixed[:, vi, mi, pi, oi]
                    original = natural[:, vi, mi, pi, oi]
                    for seed in range(3):
                        close(row[f"seed{seed}"], weights[seed],
                              "standardized seed raw")
                        close(exported[f"seed{seed}"], weights[seed],
                              "standardized CSV seed raw")
                        close(row[f"original_natural_seed{seed}"], original[seed],
                              "natural seed raw")
                        close(exported[f"original_natural_seed{seed}"], original[seed],
                              "natural CSV seed raw")
                        checked["standardized_seed_values"] += 1
                        checked["natural_seed_values"] += 1
                        checked["csv_fields"] += 2
                    fixed_mean, fixed_sd = mean_sd(weights)
                    natural_mean, natural_sd = mean_sd(original)
                    expected = {"mean": fixed_mean, "sample_sd": fixed_sd,
                                "original_natural_mean": natural_mean,
                                "original_natural_sample_sd": natural_sd,
                                "target_ood_prior": .2,
                                "original_natural_prior": .2 if pi == 0 else 10000 / 18000,
                                "n_id": 8000, "n_ood": 2000 if pi == 0 else 10000}
                    for name, value in expected.items():
                        close(row[name], value, "all report statistics " + name)
                        close(exported[name], value, "all CSV statistics " + name)
                        checked["statistical_fields"] += 1
                        checked["csv_fields"] += 1
                    if pi == 0:
                        require(np.allclose(weights, original, rtol=0, atol=1e-12),
                                "natural 20%-prior heldout identity")
                        checked["heldout_identities"] += 3
    display = (OUT / "REPORT.md").read_text(encoding="utf-8")
    display_rows = [line for line in display.splitlines() if
                    line.startswith("| T1 |") or line.startswith("| calibrated |")]
    require(len(display_rows) == 12, "12 method/variant display rows")
    indexed = {(row["variant"], row["method"], row["population"],
                row["orientation"]): row for row in rows}
    for variant in VARIANTS:
        for method in METHODS:
            values = [indexed[variant, method, pop, orientation]
                      for pop, orientation in ((POPS[0], ORIENTATIONS[0]),
                                               (POPS[1], ORIENTATIONS[0]),
                                               (POPS[0], ORIENTATIONS[1]),
                                               (POPS[1], ORIENTATIONS[1]))]
            expected = " | ".join(f"{100*row['mean']:.2f} ± {100*row['sample_sd']:.2f}%"
                                   for row in values)
            require(f"| {variant} | {method} | {expected} |" in display_rows,
                    "display row differs from all-method report values")
            checked["display_rows"] += 1
    example = indexed["T1", "energy", "cifar10", "aupr_out"]
    require(f"{100*example['original_natural_mean']:.2f}%" in display and
            f"{100*example['mean']:.2f}%" in display and
            "hypothetical display prior" in display and
            "not independent confirmation" in display and
            "not an estimated deployment prevalence" in display,
            "natural/standardized example and interpretation limits")
    record = {
        "state": "passed", "checked": checked,
        "source_sha256": {"summary": sha(OUT / "summary.json"),
                          "statistics_csv": sha(OUT / "statistics.csv"),
                          "report": sha(OUT / "REPORT.md"),
                          "raw_audit": sha(RUN / "raw-audit.json"),
                          "audit_source": sha(Path(__file__))},
        "boundary": "All 48 report rows, 144 standardized and 144 natural seed values, CSV and all-method display independently checked.",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": checked}))


if __name__ == "__main__":
    main()
