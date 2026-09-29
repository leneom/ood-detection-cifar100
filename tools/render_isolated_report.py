"""Render the fixed three-seed validation-isolation description after summary audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def points(value):
    return f'{100*value["mean"]:.2f} ± {100*value["sample_sd"]:.2f}'


def signed(value):
    return f'{100*value["mean"]:+.2f} ± {100*value["sample_sd"]:.2f}'


def main():
    summary_path = REPORT / "summary.json"
    audit_path = REPORT / "summary-audit.json"
    summary = read(summary_path)
    audit = read(audit_path)
    if (audit["state"] != "passed" or audit["summary_sha256"] != sha(summary_path) or
            audit["summary_source_sha256"] != summary["summary_source_sha256"] or
            audit["audit_source_sha256"] != sha(REPORT / "executed_summary_audit_source.py")):
        raise ValueError("independent summary audit missing or stale")
    for name, expected in summary["source_inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source input changed after summary: " + name)
    if (REPORT / "REPORT.md").exists() or (REPORT / "plot-data.json").exists():
        raise FileExistsError("report already exists; preserve it")
    cells = {(r["variant"], r["method"], r["population"]): r for r in summary["cells"]}
    if len(cells) != 72:
        raise ValueError("incomplete primary detector grid")

    lines = [
        "# Three-way ID-validation isolation replication",
        "",
        "Exploratory paired three-training-seed result on the previously inspected CIFAR-10, "
        "MNIST, SVHN and Textures OOD tests. Noise is a separate synthetic diagnostic. "
        "No detector or checkpoint was selected from OOD performance, and these data "
        "do not provide independent confirmation.",
        "",
        "The original 5,000 ID-validation examples selected checkpoints. Old T=1 "
        "operating thresholds used all 5,000; old calibrated temperature and "
        "threshold fits used disjoint 2,500/2,500 subsets of those same 5,000. "
        "In the new predeclared run, each class contributes 20 checkpoint rows, "
        "15 temperature-fit rows and 15 operational-threshold rows "
        "(2,000/1,500/1,500 mutually disjoint total). All seeds completed the same 100-epoch "
        "training schedule. Temperature and four fixed operating thresholds were fit "
        "from their designated ID rows only. The old/new comparison also changes "
        "selection and fitting sample sizes; a difference cannot be attributed solely "
        "to validation reuse.",
        "",
        f'Old selected epochs: {summary["selected_epochs_old"]}; new: '
        f'{summary["selected_epochs_new"]}. ID-test accuracy old '
        f'{points(summary["id_test_accuracy"]["old"])}%, new '
        f'{points(summary["id_test_accuracy"]["new"])}%, paired change '
        f'{signed(summary["id_test_accuracy"]["paired_new_minus_old"])} percentage points. '
        f'ID-fitted temperature old {summary["temperature"]["old"]["mean"]:.4f}, '
        f'new {summary["temperature"]["new"]["mean"]:.4f} (three seeds each).',
        "",
        "Each cell below is a three-seed mean ± sample SD in percent. Δ is the paired "
        "new-minus-old mean ± sample SD in percentage points. FPR95-ID is a benchmark "
        "test-derived threshold; acceptance95 uses the fixed ID-validation threshold.",
        "",
    ]
    for variant in VARIANTS:
        lines += [f"## {variant} natural-OOD macro", "",
                  "| Method | AUROC old | AUROC new | Δ AUROC | FPR95-ID old | FPR95-ID new | Δ FPR95-ID | Acceptance95 old | Acceptance95 new | Δ acceptance95 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            c = cells[(variant, method, "natural_macro")]
            a, f = c["metrics"]["auroc"], c["metrics"]["fpr95_id"]
            o = c["operating"]["target95"]
            lines.append("| " + " | ".join([
                LABELS[method], points(a["old"]), points(a["new"]),
                signed(a["paired_new_minus_old"]),
                points(f["old"]), points(f["new"]), signed(f["paired_new_minus_old"]),
                points(o["old_ood_acceptance"]), points(o["ood_acceptance"]),
                signed(o["paired_new_minus_old_ood_acceptance"])]) + " |")
        lines += ["", "Natural macro is the equal-weight mean of four named OOD populations "
                  "within each seed, followed by a three-seed mean and sample SD.", ""]
    lines += [
        "## Interpretation and provenance", "",
        "Raw per-image logits, features, six detector scores, correctness and all "
        "80/90/95/99 acceptance masks are retained in each seed's local extraction "
        "and scoring folders. The machine-readable summary includes every method, "
        "variant and natural OOD population, plus separate noise. Independent "
        "feature, score and summary audits bind the frozen inputs and outputs.",
        "",
        "AUROC, AUPR-out, AUPR-in, FPR95-ID, every operating rate and all three "
        "paired seed values are in [summary.json](summary.json). AUPR depends on "
        "the saved ID/OOD population counts there. The sample SD describes "
        "three training seeds, not a confidence interval over images. The four "
        "OOD test populations had already informed earlier method development; "
        "no best-method, causal or deployment claim follows from these comparisons.",
        "",
        "Figures: [paired AUROC changes](paired-auroc.png) and "
        "[fixed95 natural macro OOD acceptance](fixed95-acceptance.png).",
        "",
    ]

    auroc_delta = {v: [[cells[(v, m, p)]["metrics"]["auroc"]
                         ["paired_new_minus_old"]["mean"] for p in NATURAL]
                        for m in METHODS] for v in VARIANTS}
    acceptance = {v: {m: {
        "old": cells[(v, m, "natural_macro")]["operating"]["target95"]
               ["old_ood_acceptance"]["per_seed"],
        "new": cells[(v, m, "natural_macro")]["operating"]["target95"]
               ["ood_acceptance"]["per_seed"]}
        for m in METHODS} for v in VARIANTS}
    plot_data = {"summary_sha256": sha(summary_path), "audit_sha256": sha(audit_path),
                 "methods": METHODS, "natural": NATURAL, "variants": VARIANTS,
                 "paired_auroc_delta": auroc_delta, "fixed95_macro_ood_acceptance": acceptance}
    (REPORT / "plot-data.json").write_text(json.dumps(plot_data, indent=2) + "\n",
                                            encoding="utf-8")
    (REPORT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "savefig.dpi": 180})
    matrices = [100*np.asarray(auroc_delta[v], dtype=float) for v in VARIANTS]
    bound = max(1., max(float(np.max(np.abs(x))) for x in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.4), constrained_layout=True)
    for ax, v, matrix in zip(axes, VARIANTS, matrices):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(4), [p.upper() if p != "textures" else "Textures" for p in NATURAL])
        ax.set_yticks(range(6), [LABELS[m] for m in METHODS])
        ax.set_title(f"{v}: paired new − old AUROC (pp)")
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center",
                        color="white" if abs(matrix[i,j]) > .58*bound else "black", fontsize=8)
    fig.colorbar(im, ax=axes, label="Percentage points", shrink=.84)
    fig.savefig(REPORT / "paired-auroc.png")
    fig.savefig(REPORT / "paired-auroc.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.3), sharey=True, constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for ax, v in zip(axes, VARIANTS):
        for i, m in enumerate(METHODS):
            old_points = 100*np.asarray(acceptance[v][m]["old"], dtype=float)
            new_points = 100*np.asarray(acceptance[v][m]["new"], dtype=float)
            ax.plot([i-.12, i+.12], [old_points.mean(), new_points.mean()],
                    color=colors[i], marker="o", linewidth=1.8)
            for a, b in zip(old_points, new_points):
                ax.plot([i-.12, i+.12], [a, b], color=colors[i], alpha=.33, linewidth=.8)
        ax.set_xticks(range(6), [LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_xlim(-.55, 5.55)
        ax.set_title(f"{v}: old → new, fixed95 ID threshold")
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Natural macro OOD acceptance (%)")
    fig.savefig(REPORT / "fixed95-acceptance.png")
    fig.savefig(REPORT / "fixed95-acceptance.svg")
    plt.close(fig)

    outputs = ("REPORT.md", "plot-data.json", "paired-auroc.png", "paired-auroc.svg",
               "fixed95-acceptance.png", "fixed95-acceptance.svg")
    shutil.copyfile(__file__, REPORT / "executed_render_source.py")
    result = {"state": "rendered_pending_independent_visual_audit",
              "summary_sha256": sha(summary_path), "summary_audit_sha256": sha(audit_path),
              "render_source_sha256": sha(__file__),
              "outputs_sha256": {name: sha(REPORT / name) for name in outputs}}
    (REPORT / "render-status.json").write_text(json.dumps(result, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"state": result["state"], "outputs": len(outputs)}), flush=True)


if __name__ == "__main__":
    main()

