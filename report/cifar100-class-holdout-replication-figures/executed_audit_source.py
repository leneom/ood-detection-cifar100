"""Independently audit fixed all-method figure data and rendered file hashes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/cifar100-class-holdout-replication"
FIGURES = ROOT / "report/cifar100-class-holdout-replication-figures"
PARTITIONS = (2031, 2032, 2033)
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
PANELS = ("auroc", "id_test_acceptance", "ood_acceptance")


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


def close(a: float, b: float, label: str) -> None:
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {a} versus {b}")


def main() -> None:
    audit_path = FIGURES / "audit.json"
    snapshot = FIGURES / "executed_audit_source.py"
    require(not audit_path.exists() and not snapshot.exists(),
            "write-once figure audit already exists")
    summary_path = REPORT / "summary.json"
    report_audit_path = REPORT / "audit.json"
    completion_path = REPORT / "completion.json"
    manifest_path = FIGURES / "manifest.json"
    data_path = FIGURES / "figure-data.json"
    require(manifest_path.exists() and completion_path.exists(),
            "completed report and rendered figure required")
    plot_source = ROOT / "tools/plot_class_holdout_replication.py"
    executed_source = FIGURES / "executed_plot_source.py"
    plan = ROOT / "docs/CLASS_HOLDOUT_REPLICATION_FIGURES.md"
    summary, report_audit, completion = (
        read(path) for path in (summary_path, report_audit_path, completion_path))
    manifest, data = read(manifest_path), read(data_path)
    require(report_audit["state"] == "passed" and
            completion["state"] == "complete" and
            summary["state"] == "report_pending_independent_audit" and
            manifest["state"] == "rendered_pending_visual_audit" and
            manifest["figure_spec_sha256"] == sha(plan) and
            manifest["report_summary_sha256"] == sha(summary_path) and
            manifest["report_audit_sha256"] == sha(report_audit_path) and
            manifest["report_completion_sha256"] == sha(completion_path) and
            manifest["source_sha256"] == sha(plot_source) ==
            sha(executed_source) and
            data["state"] == "descriptive_figure_data" and
            data["class_seeds"] == list(PARTITIONS) and
            data["training_seeds"] == [0, 1, 2] and
            data["variants"] == list(VARIANTS) and
            data["methods"] == list(METHODS) and
            data["panels"] == list(PANELS) and
            data["n_rows"] == len(data["rows"]) == 108,
            "fixed complete audited figure provenance and 108-row grid")
    require(set(manifest["outputs_sha256"]) == {
        "all-methods.png", "all-methods.pdf", "figure-data.json",
        "executed_plot_source.py"}, "four immutable figure outputs")
    for filename, expected in manifest["outputs_sha256"].items():
        require(sha(FIGURES / filename) == expected,
                "figure output changed: " + filename)

    position = 0
    for variant_index, variant in enumerate(VARIANTS):
        for panel in PANELS:
            for method_index, method in enumerate(METHODS):
                cell = summary["cells"][variant_index * len(METHODS) +
                                        method_index]
                require(cell["variant"] == variant and cell["method"] == method,
                        "summary cell ordering")
                values = (cell["metrics"]["auroc"] if panel == "auroc" else
                          cell["operating"]["target95"][panel])
                for partition in PARTITIONS:
                    saved = data["rows"][position]
                    position += 1
                    require(saved["variant"] == variant and
                            saved["panel"] == panel and
                            saved["method"] == method and
                            saved["class_seed"] == partition,
                            "figure-data row ordering")
                    source = values["partitions"][str(partition)]
                    points = np.asarray(source["per_seed"], dtype=float)
                    actual = np.asarray(saved["training_seed_values"], dtype=float)
                    require(points.shape == actual.shape == (3,) and
                            np.isfinite(points).all() and np.isfinite(actual).all(),
                            "complete finite three-model observations")
                    for seed in range(3):
                        close(actual[seed], points[seed],
                              f"{variant}/{panel}/{method}/{partition}/{seed}")
                    mean = float(np.mean(points))
                    sd = float(np.std(points, ddof=1))
                    close(source["mean"], mean, "source partition mean")
                    close(source["sample_sd"], sd, "source within-partition SD")
                    close(saved["mean"], mean, "plotted partition mean")
                    close(saved["sample_sd"], sd, "plotted partition SD")
    require(position == 108, "all fixed figure rows checked")

    png = FIGURES / "all-methods.png"
    with Image.open(png) as image:
        require(image.format == "PNG" and image.size == (3000, 2200),
                "six-panel 15x11 inch 200-dpi PNG size")
        rgb = np.asarray(image.convert("RGB"))
    nonwhite = int(np.count_nonzero(np.any(rgb < 245, axis=2)))
    require(nonwhite > 6000, "rendered PNG appears blank")
    pdf = FIGURES / "all-methods.pdf"
    require(pdf.stat().st_size > 20_000 and
            pdf.open("rb").read(5) == b"%PDF-", "rendered vector PDF")
    shutil.copyfile(__file__, snapshot)
    record = {
        "state": "numeric_and_file_audit_passed_pending_visual_review",
        "figure_rows": position,
        "seed_values": position * 3,
        "summary_means_checked": position,
        "summary_sample_sds_checked": position,
        "png_size_pixels": [3000, 2200],
        "png_nonwhite_pixels": nonwhite,
        "figure_spec_sha256": sha(plan),
        "report_completion_sha256": sha(completion_path),
        "manifest_sha256": sha(manifest_path),
        "audit_source_sha256": sha(Path(__file__)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Independent numeric and rendered-file audit only; visual inspection is still required before presentation completion.",
    }
    audit_path.write_text(json.dumps(record, indent=2, allow_nan=False) +
                          "\n", encoding="utf-8")
    print(f"figure data/file audit passed; {position} rows; visual review pending")


if __name__ == "__main__":
    main()
