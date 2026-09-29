"""Render the predeclared all-method, three-partition sensitivity figure."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/cifar100-class-holdout-replication"
OUTPUT = ROOT / "report/cifar100-class-holdout-replication-figures-v2"
SPEC = ROOT / "docs/CLASS_HOLDOUT_REPLICATION_FIGURES.md"
PARTITIONS = (2031, 2032, 2033)
VARIANTS = ("T1", "calibrated")
METHODS = ("msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis")
COLORS = ("#3B6FB6", "#E08D3C", "#38936B")
ROWS = (
    ("AUROC", "auroc", "AUROC (%)"),
    ("ID accepted at fixed 95%", "id_test_acceptance", "ID acceptance (%)"),
    ("OOD accepted at fixed 95%", "ood_acceptance", "OOD acceptance (%)"),
)


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
        raise ValueError(why)


def main() -> None:
    require(not OUTPUT.exists(), "write-once figure directory already exists")
    completion_path = REPORT / "completion.json"
    require(completion_path.exists(),
            "complete audited nine-model report required before plotting")
    completion = read(completion_path)
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.verify_class_holdout_replication_completion import validate
    require(completion["state"] == "complete" and
            completion["inputs_sha256"] == validate(),
            "complete nine-model source/raw/report audit required")
    summary_path = REPORT / "summary.json"
    audit_path = REPORT / "audit.json"
    summary, audit = read(summary_path), read(audit_path)
    require(summary["state"] == "report_pending_independent_audit" and
            audit["state"] == "passed" and
            summary["class_seeds"] == list(PARTITIONS) and
            summary["variants"] == list(VARIANTS) and
            summary["methods"] == list(METHODS) and
            len(summary["cells"]) == 12,
            "fixed complete audited all-method report")
    data_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True,
                             constrained_layout=True)
    x = np.arange(len(METHODS), dtype=float)
    offsets = (-.24, 0., .24)
    for column, variant in enumerate(VARIANTS):
        for row_index, (title, field, ylabel) in enumerate(ROWS):
            ax = axes[row_index, column]
            for method_index, method in enumerate(METHODS):
                cell = summary["cells"][column * len(METHODS) + method_index]
                require(cell["variant"] == variant and cell["method"] == method,
                        "fixed cell ordering")
                values = (cell["metrics"]["auroc"] if field == "auroc" else
                          cell["operating"]["target95"][field])
                for part_index, partition in enumerate(PARTITIONS):
                    item = values["partitions"][str(partition)]
                    points = np.asarray(item["per_seed"], dtype=float)
                    mean = float(item["mean"])
                    sd = float(item["sample_sd"])
                    require(points.shape == (3,) and np.isfinite(points).all() and
                            0 <= points.min() <= points.max() <= 1 and
                            0 <= mean <= 1 and sd >= 0,
                            "finite three-seed percentage observations")
                    center = x[method_index] + offsets[part_index]
                    ax.errorbar(center, 100 * mean, yerr=100 * sd, fmt="o",
                                color=COLORS[part_index], capsize=3,
                                markersize=5, elinewidth=1.5,
                                label=f"Partition {partition}" if
                                method_index == 0 and row_index == 0 and
                                column == 0 else None)
                    ax.scatter(center + np.asarray((-.045, 0., .045)),
                               100 * points, s=13, color=COLORS[part_index],
                               alpha=.50, linewidths=0)
                    data_rows.append({
                        "variant": variant, "panel": field, "method": method,
                        "class_seed": partition, "training_seed_values":
                            points.tolist(), "mean": mean, "sample_sd": sd,
                    })
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.set_xlim(-.65, len(METHODS) - .35)
            ax.grid(axis="y", alpha=.20, linewidth=.7)
            ax.set_title(f"{title} · {'T=1' if variant == 'T1' else 'ID-only temperature'}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, [m.upper() if m in ("msp", "knn") else
                              ("MaxLogit" if m == "maxlogit" else
                               ("Mahalanobis" if m == "mahalanobis" else m.title()))
                              for m in METHODS], rotation=18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", ncol=3,
                      frameon=False, fontsize=10)
    fig.suptitle("Fixed CIFAR-100 class-partition sensitivity · three model seeds each",
                 fontsize=15)
    fig.supxlabel(
        "All six fixed methods · points are model seeds; circles/error bars are "
        "partition mean ± within-partition sample SD\n"
        "Full 0–100% scale. Original partition 2031 was previously viewed; "
        "these three partitions do not form a confidence interval.",
        fontsize=9)
    OUTPUT.mkdir(parents=True)
    png = OUTPUT / "all-methods.png"
    pdf = OUTPUT / "all-methods.pdf"
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    data_path = OUTPUT / "figure-data.json"
    data_path.write_text(json.dumps({
        "state": "descriptive_figure_data",
        "class_seeds": list(PARTITIONS),
        "training_seeds": [0, 1, 2],
        "variants": list(VARIANTS),
        "methods": list(METHODS),
        "panels": [r[1] for r in ROWS],
        "n_rows": len(data_rows),
        "rows": data_rows,
        "boundary": "All methods and fixed ID-validation 95% target; no OOD-based choice or partition CI.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    source_copy = OUTPUT / "executed_plot_source.py"
    shutil.copyfile(__file__, source_copy)
    manifest = {
        "state": "rendered_pending_visual_audit",
        "figure_spec_sha256": sha(SPEC),
        "report_summary_sha256": sha(summary_path),
        "report_audit_sha256": sha(audit_path),
        "report_completion_sha256": sha(completion_path),
        "source_sha256": sha(Path(__file__)),
        "outputs_sha256": {
            item.name: sha(item) for item in
            (png, pdf, data_path, source_copy)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "Presentation of the independently audited nine-model grid; complete numeric CSV/JSON remains authoritative.",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"rendered {len(data_rows)} fixed panel/method/partition rows")


if __name__ == "__main__":
    main()
