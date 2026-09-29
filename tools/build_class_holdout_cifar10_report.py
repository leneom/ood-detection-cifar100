"""Complete fixed six-method, three-seed exploratory CIFAR-10 extension report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/class-holdout-cifar10-extension"
OUT = ROOT / "report/class-holdout-cifar10-extension"
BASELINE = ROOT / "report/cifar100-class-holdout/summary.json"
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")
TARGETS = (80, 90, 95, 99)
OPERATING = ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")


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


def stats(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (3,) and np.isfinite(x).all(), "three finite model seed values")
    return {"per_seed": x.tolist(), "mean": float(x.mean()),
            "sample_sd": float(x.std(ddof=1))}


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    require(not OUT.exists(), "write-once CIFAR-10 extension report exists")
    declaration = read(RUN / "declaration.json")
    test_audit = read(RUN / "test-population-audit.json")
    require(declaration["state"] == "frozen_before_80class_cifar10_inference" and
            test_audit["state"] == "passed" and
            test_audit["declaration_sha256"] == sha(RUN / "declaration.json"),
            "declared and independently audited full test population")
    seed_records = []
    inputs = {"declaration": sha(RUN / "declaration.json"),
              "test_population_audit": sha(RUN / "test-population-audit.json"),
              "baseline_summary": sha(BASELINE),
              "report_builder": sha(Path(__file__))}
    for seed in range(3):
        path = RUN / f"score-seed{seed}" / "manifest.json"
        audit_path = RUN / f"score-audit-seed{seed}.json"
        record, audit = read(path), read(audit_path)
        require(record["state"] == "scored_pending_independent_audit" and
                record["seed"] == seed and
                audit["state"] == "passed" and audit["seed"] == seed and
                audit["scoring_manifest_sha256"] == sha(path) and
                audit["checked"] == {"score_vectors": 12, "metric_fields": 156,
                                     "operating_fields": 528,
                                     "ood_mask_vectors": 48,
                                     "frozen_id_mask_vectors": 48} and
                record["n_id"] == 8000 and record["n_ood"] == 10000 and
                record["ood_prevalence"] == 10000 / 18000,
                "all three complete score audits required")
        seed_records.append(record)
        inputs[f"score_seed{seed}"] = sha(path)
        inputs[f"score_audit_seed{seed}"] = sha(audit_path)
    baseline = read(BASELINE)
    baseline_cells = {(cell["variant"], cell["method"]): cell
                      for cell in baseline["cells"]}
    require(len(baseline_cells) == 12, "original same-source OOD summary")
    metric_rows, operating_rows = [], []
    for variant in VARIANTS:
        for method in METHODS:
            for metric in METRICS:
                values = [seed_records[seed]["method_results"][variant][method]
                          ["metrics"][metric] for seed in range(3)]
                result = stats(values)
                metric_rows.append({"variant": variant, "method": method,
                                    "metric": metric,
                                    "seed0": result["per_seed"][0],
                                    "seed1": result["per_seed"][1],
                                    "seed2": result["per_seed"][2],
                                    "mean": result["mean"],
                                    "sample_sd": result["sample_sd"],
                                    "n_id": 8000, "n_ood": 10000,
                                    "ood_prevalence": 10000 / 18000})
            for target in TARGETS:
                for population in OPERATING:
                    key = ("calibration" if population == "calibration_acceptance"
                           else "id_test" if population == "id_test_acceptance"
                           else "ood")
                    denominator = 1200 if key == "calibration" else 8000 if key == "id_test" else 10000
                    records = [seed_records[seed]["method_results"][variant][method]
                               ["operating_grid"][f"target{target}"] for seed in range(3)]
                    counts = [record[f"{key}_accepted"] for record in records]
                    values = [record[population] for record in records]
                    require(all(record[f"{key}_count"] == denominator for record in records) and
                            all(abs(value - count / denominator) < 1e-12
                                for value, count in zip(values, counts)),
                            "operating count/denominator agreement")
                    result = stats(values)
                    operating_rows.append({"variant": variant, "method": method,
                                           "target": target, "population": population,
                                           "accepted0": counts[0], "accepted1": counts[1],
                                           "accepted2": counts[2],
                                           "denominator": denominator,
                                           "seed0": result["per_seed"][0],
                                           "seed1": result["per_seed"][1],
                                           "seed2": result["per_seed"][2],
                                           "mean": result["mean"],
                                           "sample_sd": result["sample_sd"]})
    require(len(metric_rows) == 48 and len(operating_rows) == 144,
            "all fixed metrics/operating settings")
    OUT.mkdir(parents=True)
    csv_write(OUT / "metrics.csv", metric_rows)
    csv_write(OUT / "operating.csv", operating_rows)
    summary = {
        "state": "report_pending_independent_audit",
        "study": "Frozen 80-class CIFAR-100 holdout models versus complete official CIFAR-10 test",
        "seeds": [0, 1, 2], "variants": list(VARIANTS),
        "methods": list(METHODS), "metrics": list(METRICS),
        "targets": list(TARGETS), "n_id": 8000, "n_ood": 10000,
        "ood_prevalence": 10000 / 18000,
        "metric_rows": metric_rows, "operating_rows": operating_rows,
        "inputs_sha256": inputs,
        "csv_sha256": {"metrics": sha(OUT / "metrics.csv"),
                       "operating": sha(OUT / "operating.csv")},
        "boundary": "Exploratory post-hoc extension; CIFAR-10 appeared earlier in project; no method selection or cross-population AUPR comparison.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                               allow_nan=False) + "\n",
                                      encoding="utf-8")
    metrics_by = {(r["variant"], r["method"], r["metric"]): r for r in metric_rows}
    operating_by = {(r["variant"], r["method"], r["target"], r["population"]): r
                    for r in operating_rows}
    lines = [
        "# Frozen 80-class holdout models on full CIFAR-10 test",
        "",
        "Exploratory post-hoc extension selected after the same-source held-out-class",
        "results had been viewed. CIFAR-10 was used in earlier project work. All three",
        "original 100-epoch selected checkpoints, their 36,000 optimization-ID feature",
        "banks, ID-only temperatures and validation thresholds were reused unchanged.",
        "All 10,000 official CIFAR-10 test rows were scored against the same 8,000",
        "CIFAR-100 ID test rows. No CIFAR-10 image was used for fit or selection.",
        "",
        "CIFAR-10 OOD prevalence is 55.56% (10,000/18,000), while the original",
        "held-out-class OOD prevalence was 20% (2,000/10,000). Their AUPR values",
        "must not be directly compared as though class proportions were identical.",
        "Full AUROC, AUPR-out/in, FPR95-ID and four original ID-validation operating",
        "targets for all methods/variants/seeds are in `metrics.csv` and `operating.csv`.",
        "Raw 80-class logits, 512-dimensional features, per-image scores and masks",
        "remain in the local run folders. The table is descriptive, not a selected winner.",
        "",
        "| Variant | Method | CIFAR-10 AUROC mean ± seed SD | Held-out-class AUROC mean ± seed SD | CIFAR-10 OOD acceptance at original ID-val 95% | Held-out-class OOD acceptance at original ID-val 95% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for method in METHODS:
            current_a = metrics_by[variant, method, "auroc"]
            current_o = operating_by[variant, method, 95, "ood_acceptance"]
            original = baseline_cells[variant, method]
            old_a = original["metrics"]["auroc"]
            old_o = original["operating"]["target95"]["ood_acceptance"]
            lines.append(
                f"| {variant} | {method} | {100*current_a['mean']:.2f} ± {100*current_a['sample_sd']:.2f}% | "
                f"{100*old_a['mean']:.2f} ± {100*old_a['sample_sd']:.2f}% | "
                f"{100*current_o['mean']:.2f} ± {100*current_o['sample_sd']:.2f}% | "
                f"{100*old_o['mean']:.2f} ± {100*old_o['sample_sd']:.2f}% |"
            )
    lines += ["", "Both populations share the same ID test scores and ID-validation",
              "thresholds. FPR95-ID uses the ID test order statistic instead and is",
              "separate from deployable calibration. Different semantic label sets,",
              "the earlier use of CIFAR-10, and this post-hoc population choice limit",
              "any cross-population interpretation. No external confirmation or causal",
              "source-matching conclusion is claimed.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": summary["state"], "metric_rows": 48,
                      "operating_rows": 144}))


if __name__ == "__main__":
    main()
