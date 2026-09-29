"""Independent reconstruction of the Places365 three-seed summary and display."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
OUT = ROOT / "report" / "places365-extension"
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
TARGETS = (80, 90, 95, 99)
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id", "id_tpr_at_fpr95",
           "benchmark_threshold", "validation_threshold",
           "id_test_acceptance_at_validation_threshold",
           "ood_acceptance_at_validation_threshold", "calibration_id_acceptance",
           "ood_prevalence")
RATES = ("calibration_acceptance", "id_test_acceptance", "ood_acceptance")
DISPLAY = {"msp": "MSP", "energy": "Energy", "maxlogit": "MaxLogit",
           "entropy": "Entropy", "knn": "KNN-50", "mahalanobis": "Pooled Mahalanobis"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def group(record, points, label):
    expected = [float(v) for v in points]
    check(len(expected) == 3 and record["seeds"] == expected, label + " seed points")
    check(abs(record["mean"] - statistics.fmean(expected)) <= 1e-12,
          label + " mean")
    check(abs(record["sample_sd"] - statistics.stdev(expected)) <= 1e-12,
          label + " sample SD")
    return 3


def pct(record):
    return f"{record['mean']*100:.2f} ± {record['sample_sd']*100:.2f}"


def main():
    summary = read(OUT / "summary.json")
    check(summary["state"] == "three_seed_places365_scored_and_audited" and
          summary["id_test_count"] == 10000 and summary["ood_count"] == 36500 and
          summary["ood_prevalence"] == 36500/46500 and
          tuple(summary["method_order"]) == METHODS and
          tuple(summary["variants"]) == VARIANTS and
          tuple(summary["targets"]) == TARGETS, "fixed summary design")
    for name, path in (("protocol_sha256", "docs/PLACES365_EXTENSION.md"),
                       ("amendment_sha256", "docs/PLACES365_AMENDMENT_1.md"),
                       ("cache_sha256", "data/places365-validation.npz"),
                       ("source_sha256", "tools/build_places365_report.py")):
        check(summary[name] == sha(ROOT / path), "source binding " + path)
    manifests = []
    for seed in range(3):
        scored = RUN / f"score-seed{seed}"
        audit = RUN / f"score-audit-seed{seed}" / "audit.json"
        extract = RUN / f"extract-seed{seed}" / "manifest.json"
        manifest, audit_json, raw = read(scored / "manifest.json"), read(audit), read(extract)
        sr = summary["seed_records"][seed]
        check(audit_json["state"] == "passed" and
              audit_json["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
              sr["seed"] == seed and sr["checkpoint_epoch"] == raw["checkpoint_epoch"] and
              sr["checkpoint_sha256"] == raw["checkpoint_sha256"] == manifest["checkpoint_sha256"] and
              sr["temperature"] == manifest["temperature"] and
              sr["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
              sr["score_audit_sha256"] == sha(audit), "seed lineage")
        for path_key, filepath in (("raw_logits_features", RUN / f"extract-seed{seed}/raw.npz"),
                                  ("per_image_scores", scored / "scores.npz"),
                                  ("acceptance_masks", scored / "acceptance_masks.npz")):
            check((ROOT / sr[path_key]).resolve() == filepath.resolve() and filepath.is_file(),
                  "raw artifact pointer " + path_key)
        manifests.append(manifest)
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    count = {"seed_values": 0, "table_rows": 0, "metric_groups": 0,
             "operating_groups": 0, "integer_vectors": 0, "figures": 0}
    for variant in VARIANTS:
        check(f"## {variant}\n" in report, variant + " report section")
        section = report.split(f"## {variant}\n", 1)[1].split("## ", 1)[0]
        for method in METHODS:
            result = summary["results"][variant][method]
            originals = [m["results"][variant][method] for m in manifests]
            for key in METRICS:
                count["seed_values"] += group(result["metrics"][key],
                    [r["metrics"][key] for r in originals], variant + method + key)
                count["metric_groups"] += 1
            for target in TARGETS:
                t = f"target{target}"
                for key in RATES:
                    count["seed_values"] += group(result["operating_grid"][t][key],
                        [r["operating_grid"][t][key] for r in originals],
                        variant + method + t + key)
                    count["operating_groups"] += 1
                for key, rawkey in (("thresholds", "threshold"),
                                    ("calibration_accepted", "calibration_accepted"),
                                    ("id_test_accepted", "id_test_accepted"),
                                    ("ood_accepted", "ood_accepted"),
                                    ("calibration_ties_at_threshold", "calibration_ties_at_threshold")):
                    expected = [r["operating_grid"][t][rawkey] for r in originals]
                    check(result["operating_grid"][t][key] == expected,
                          variant + method + t + key)
                    count["integer_vectors"] += 1
            values = [result["metrics"][k] for k in
                      ("auroc", "aupr_out", "aupr_in", "fpr95_id")]
            values += [result["operating_grid"]["target95"][k]
                       for k in ("ood_acceptance", "id_test_acceptance")]
            row = "| " + DISPLAY[method] + " | " + " | ".join(pct(v) for v in values) + " |"
            check(section.count(row) == 1, variant + method + " main report row")
            count["table_rows"] += 1
            values = [result["operating_grid"][f"target{target}"]["ood_acceptance"]
                      for target in TARGETS]
            row = "| " + DISPLAY[method] + " | " + " | ".join(pct(v) for v in values) + " |"
            check(section.count(row) == 1, variant + method + " operating report row")
            count["table_rows"] += 1
    for figure in ("auroc", "fixed95-acceptance"):
        png, svg = OUT / f"{figure}.png", OUT / f"{figure}.svg"
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            check(im.width >= 1500 and im.height >= 800, figure + " legible canvas")
        check("<svg" in svg.read_text(encoding="utf-8")[:500], figure + " vector figure")
        check(f"[{('AUROC' if figure == 'auroc' else 'fixed95 OOD acceptance')}]({figure}.png)" in report,
              figure + " report link")
        count["figures"] += 2
    record = {"state": "passed", "checked": count,
              "summary_sha256": sha(OUT / "summary.json"),
              "report_sha256": sha(OUT / "REPORT.md"),
              "figure_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*.png")) + sorted(OUT.glob("*.svg"))},
              "audit_source_sha256": sha(__file__),
              "created_at": datetime.now(timezone.utc).isoformat()}
    path = OUT / "display-audit.json"
    check(not path.exists(), "display audit already exists")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "checked": count}))


if __name__ == "__main__":
    main()

