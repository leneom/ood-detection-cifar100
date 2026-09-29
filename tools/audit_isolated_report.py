"""Audit isolation report text, source bindings and plotted numeric coordinates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "validation-isolation"
NATURAL = ("cifar10", "mnist", "svhn", "textures")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
VARIANTS = ("T1", "calibrated")
LABELS = {"msp": "MSP", "energy": "Energy", "maxlogit": "MaxLogit",
          "entropy": "Entropy", "knn": "KNN-50", "mahalanobis": "Pooled Maha"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def eq(actual, expected, label):
    require(math.isclose(float(actual), float(expected), abs_tol=1e-14, rel_tol=1e-14), label)


def formatted(item, signed=False):
    if signed:
        return f'{100*item["mean"]:+.2f} ± {100*item["sample_sd"]:.2f}'
    return f'{100*item["mean"]:.2f} ± {100*item["sample_sd"]:.2f}'


def main():
    summary_path = REPORT / "summary.json"
    audit_path = REPORT / "summary-audit.json"
    status_path = REPORT / "render-status.json"
    summary, audit, status = read(summary_path), read(audit_path), read(status_path)
    require(audit["state"] == "passed" and audit["summary_sha256"] == sha(summary_path),
            "summary audit changed")
    require(status["state"] == "rendered_pending_independent_visual_audit" and
            status["summary_sha256"] == sha(summary_path) and
            status["summary_audit_sha256"] == sha(audit_path), "render status")
    require(status["render_source_sha256"] ==
            sha(REPORT / "executed_render_source.py") ==
            sha(ROOT / "tools/render_isolated_report.py"), "render source")
    for name, expected in status["outputs_sha256"].items():
        require(sha(REPORT / name) == expected, "changed report output: " + name)
    plot = read(REPORT / "plot-data.json")
    require(plot["summary_sha256"] == sha(summary_path) and
            plot["audit_sha256"] == sha(audit_path), "plot source binding")
    require(tuple(plot["methods"]) == METHODS and tuple(plot["natural"]) == NATURAL and
            tuple(plot["variants"]) == VARIANTS, "plot axes")
    cells = {(r["variant"], r["method"], r["population"]): r for r in summary["cells"]}
    require(len(cells) == 72, "complete metric grid")
    coordinate_count = 0
    for variant in VARIANTS:
        for mi, method in enumerate(METHODS):
            for pi, population in enumerate(NATURAL):
                expected = cells[(variant, method, population)]["metrics"]["auroc"][
                    "paired_new_minus_old"]["mean"]
                eq(plot["paired_auroc_delta"][variant][mi][pi], expected,
                   "AUROC coordinate")
                coordinate_count += 1
            for side, key in (("old", "old_ood_acceptance"), ("new", "ood_acceptance")):
                actual = plot["fixed95_macro_ood_acceptance"][variant][method][side]
                expected = cells[(variant, method, "natural_macro")]["operating"]["target95"][
                    key]["per_seed"]
                require(len(actual) == len(expected) == 3, "acceptance seeds")
                for x, y in zip(actual, expected):
                    eq(x, y, "acceptance coordinate")
                    coordinate_count += 1

    text = (REPORT / "REPORT.md").read_text(encoding="utf-8")
    require("previously inspected" in text and "Noise is a separate synthetic diagnostic" in text
            and "cannot be attributed solely" in text, "interpretation boundary")
    for variant in VARIANTS:
        require(f"## {variant} natural-OOD macro" in text, "variant heading")
        for method in METHODS:
            row = cells[(variant, method, "natural_macro")]
            a, f = row["metrics"]["auroc"], row["metrics"]["fpr95_id"]
            o = row["operating"]["target95"]
            expected = "| " + " | ".join((
                LABELS[method], formatted(a["old"]), formatted(a["new"]),
                formatted(a["paired_new_minus_old"], True),
                formatted(f["old"]), formatted(f["new"]),
                formatted(f["paired_new_minus_old"], True),
                formatted(o["old_ood_acceptance"]), formatted(o["ood_acceptance"]),
                formatted(o["paired_new_minus_old_ood_acceptance"], True))) + " |"
            require(text.count(expected) == 1, "displayed macro table row: " + variant + "/" + method)

    sizes = {}
    for name in ("paired-auroc", "fixed95-acceptance"):
        with Image.open(REPORT / (name + ".png")) as im:
            require(im.width >= 1600 and im.height >= 500, "figure resolution")
            sizes[name] = [im.width, im.height]
        root = ET.parse(REPORT / (name + ".svg")).getroot()
        require(root.tag.endswith("svg") and len(list(root.iter())) > 100, "SVG content")
    result_path = REPORT / "visual-audit.json"
    require(not result_path.exists(), "visual audit already exists")
    result = {"state": "passed_pending_human_figure_inspection",
              "render_status_sha256": sha(status_path),
              "summary_sha256": sha(summary_path),
              "audit_source_sha256": sha(__file__),
              "plot_coordinates_checked": coordinate_count,
              "displayed_macro_rows_checked": len(VARIANTS)*len(METHODS),
              "png_sizes": sizes}
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

