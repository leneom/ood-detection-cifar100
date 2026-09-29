"""Build the LogitNorm-vs-CE report from the three frozen seed manifests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_logitnorm_representation import (METHODS, OBJECTIVES, PHASE, POPULATIONS, TARGETS, read, sha, write)

REPORT = ROOT / "report" / "logitnorm-representation"
SEEDS = (0, 1, 2)
LABELS = {"cifar10": "CIFAR-10", "mnist": "MNIST", "svhn": "SVHN", "textures": "Textures",
          "tinyimagenet": "Tiny ImageNet", "fashionmnist": "Fashion-MNIST", "places365": "Places365",
          "oxford_pets": "Oxford Pets"}
METHOD_LABELS = {"msp": "MSP", "energy": "Energy", "maxlogit": "MaxLogit", "entropy": "Entropy",
                 "knn": "KNN-50", "mahalanobis": "Pooled Mahalanobis"}
HYPOTHESES = {"H3": "msp", "H4": "knn"}
METRICS = ("auroc", "aupr_out", "aupr_in", "fpr95_id")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"seeds": values.tolist(), "mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def decide(deltas):
    if all(d > 0 for d in deltas):
        return "supported"
    if all(d < 0 for d in deltas):
        return "contradicted"
    return "mixed"


def overall(decisions):
    supported = sum(d == "supported" for d in decisions)
    if supported == len(decisions):
        return "confirmed"
    if supported and "contradicted" not in decisions:
        return "partially confirmed"
    return "not confirmed"


def pp(entry):
    return f"{100 * entry['mean']:+.2f} ± {100 * entry['sd']:.2f}"


def pct(entry):
    return f"{100 * entry['mean']:.2f} ± {100 * entry['sd']:.2f}"


def load():
    manifests = {}
    evaluation_sha = sha(PHASE / "evaluation-declaration.json")
    for seed in SEEDS:
        folder = PHASE / f"score-seed{seed}"
        if read(folder / "status.json")["manifest_sha256"] != sha(folder / "manifest.json"):
            raise ValueError("score manifest changed")
        manifest = read(folder / "manifest.json")
        if manifest["evaluation_declaration_sha256"] != evaluation_sha or manifest["seed"] != seed:
            raise ValueError("score manifest not bound to the evaluation declaration")
        manifests[seed] = manifest
    return evaluation_sha, manifests


def summarize(manifests):
    value = lambda s, o, p, m: manifests[s]["results"][o][p][m]
    metrics = {o: {p: {m: {k: stats([value(s, o, p, m)["metrics"][k] for s in SEEDS]) for k in METRICS}
                       for m in METHODS} for p in POPULATIONS} for o in OBJECTIVES}
    operating = {o: {p: {m: {f: stats([value(s, o, p, m)["operating_grid"]["target95"][f] for s in SEEDS])
                             for f in ("id_test_acceptance", "ood_acceptance")}
                         for m in METHODS} for p in POPULATIONS} for o in OBJECTIVES}
    contrasts = {}
    for m in METHODS:
        contrasts[m] = {}
        for p in POPULATIONS:
            auroc = [value(s, "logitnorm", p, m)["metrics"]["auroc"] - value(s, "ce", p, m)["metrics"]["auroc"]
                     for s in SEEDS]
            grid = lambda s, o, f: value(s, o, p, m)["operating_grid"]["target95"][f]
            ood = [grid(s, "logitnorm", "ood_acceptance") - grid(s, "ce", "ood_acceptance") for s in SEEDS]
            idt = [grid(s, "logitnorm", "id_test_acceptance") - grid(s, "ce", "id_test_acceptance") for s in SEEDS]
            contrasts[m][p] = {"auroc_difference": stats(auroc), "sign_pattern": decide(auroc),
                               "fixed95_ood_acceptance_difference": stats(ood),
                               "fixed95_id_test_acceptance_difference": stats(idt)}
    decisions = {}
    for name, m in HYPOTHESES.items():
        per = {p: contrasts[m][p]["sign_pattern"] for p in POPULATIONS}
        decisions[name] = {"method": m, "per_population": per, "overall": overall(list(per.values()))}
    accuracy = {o: stats([manifests[s]["id_test_accuracy"][o] for s in SEEDS]) for o in OBJECTIVES}
    accuracy["difference"] = stats([manifests[s]["id_test_accuracy"]["logitnorm"] -
                                    manifests[s]["id_test_accuracy"]["ce"] for s in SEEDS])
    floor = {o: {p: min(((operating[o][p][m]["ood_acceptance"]["mean"], m) for m in METHODS))
                 for p in POPULATIONS} for o in OBJECTIVES}
    floor = {o: {p: {"method": m, "mean_ood_acceptance": v} for p, (v, m) in per.items()} for o, per in floor.items()}
    return metrics, operating, contrasts, decisions, accuracy, floor


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def tables(manifests):
    rows = []
    for o in OBJECTIVES:
        for p in POPULATIONS:
            for m in METHODS:
                for s in SEEDS:
                    r = manifests[s]["results"][o][p][m]
                    g = r["operating_grid"]
                    rows.append([o, p, m, s, *[repr(r["metrics"][k]) for k in METRICS],
                                 *[repr(g[f"target{t}"]["id_test_acceptance"]) for t in TARGETS],
                                 *[repr(g[f"target{t}"]["ood_acceptance"]) for t in TARGETS]])
    write_csv(REPORT / "metrics.csv", ["objective", "population", "method", "seed", *METRICS,
                                       *[f"id_test_acceptance_target{t}" for t in TARGETS],
                                       *[f"ood_acceptance_target{t}" for t in TARGETS]], rows)
    contrast_rows = []
    for m in METHODS:
        for p in POPULATIONS:
            for s in SEEDS:
                ln, ce = (manifests[s]["results"][o][p][m] for o in ("logitnorm", "ce"))
                contrast_rows.append([m, p, s, repr(ln["metrics"]["auroc"] - ce["metrics"]["auroc"]),
                                      repr(ln["operating_grid"]["target95"]["ood_acceptance"] -
                                           ce["operating_grid"]["target95"]["ood_acceptance"])])
    write_csv(REPORT / "contrasts.csv", ["method", "population", "seed", "auroc_difference_logitnorm_minus_ce",
                                         "fixed95_ood_acceptance_difference"], contrast_rows)
    return len(rows), len(contrast_rows)


def figure(contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=surface, sharey=True)
    titles = {"H3": "H3: MSP, LogitNorm − CE", "H4": "H4: KNN-50, LogitNorm − CE"}
    for ax, (name, m) in zip(axes, HYPOTHESES.items()):
        ax.set_facecolor(surface)
        ax.axvline(0, color=muted, linewidth=1)
        for y, p in enumerate(POPULATIONS):
            entry = contrasts[m][p]["auroc_difference"]
            ax.plot([100 * entry["mean"]] * 2, [y - 0.3, y + 0.3], color=ink, linewidth=2, zorder=2,
                    solid_capstyle="round")
            ax.scatter(100 * np.asarray(entry["seeds"]), [y] * 3, s=52, color=series, edgecolors=surface,
                       linewidths=1.5, zorder=3)
        ax.set_title(titles[name], color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Seed-matched AUROC difference (percentage points)", color=muted, fontsize=9)
        ax.grid(axis="x", color=grid, linewidth=1)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(POPULATIONS)), [LABELS[p] for p in POPULATIONS], color=ink)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, "Dots: the three seed-matched model pairs. Black tick: mean. Right of zero means LogitNorm "
             "separates OOD better.", color=muted, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REPORT / "logitnorm_auroc_differences.png", dpi=160, facecolor=surface)
    plt.close(fig)
    return "logitnorm_auroc_differences.png"


def report_text(evaluation_sha, metrics, operating, contrasts, decisions, accuracy, floor, figure_name):
    lines = ["# LogitNorm representation study", "",
             "Three new ResNet-18 classifiers trained with LogitNorm (τ = 0.04), with every other setting identical to "
             "the three ID-isolated cross-entropy (CE) models, which serve as seed-matched comparators. The "
             "[protocol](../../docs/LOGITNORM_REPRESENTATION.md) was frozen before training, and the evaluation code "
             f"before any LogitNorm OOD output existed (evaluation declaration `{evaluation_sha}`).", "",
             f"ID-test accuracy: CE {pct(accuracy['ce'])}%, LogitNorm {pct(accuracy['logitnorm'])}%, "
             f"seed-matched difference {pp(accuracy['difference'])} pp.", "",
             "## Decisions", "", "| Hypothesis | " + " | ".join(LABELS[p] for p in POPULATIONS) + " | Overall |",
             "|---|" + "---|" * (len(POPULATIONS) + 1)]
    names = {"H3": "H3 MSP improves", "H4": "H4 KNN-50 improves"}
    for name in HYPOTHESES:
        per = decisions[name]["per_population"]
        lines.append(f"| {names[name]} | " + " | ".join(per[p] for p in POPULATIONS) +
                     f" | **{decisions[name]['overall']}** |")
    lines += ["", "The sign rule was fixed before training: \"supported\" means all three seed-matched differences "
              "are positive, \"contradicted\" means all are negative, and \"mixed\" covers everything else.", "",
              "## AUROC difference, LogitNorm − CE", "",
              "Mean ± sample SD over the three seed-matched pairs, in percentage points.", "",
              "| Population | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
              "|---|" + "---:|" * len(METHODS)]
    for p in POPULATIONS:
        lines.append(f"| {LABELS[p]} | " + " | ".join(pp(contrasts[m][p]["auroc_difference"]) for m in METHODS) + " |")
    lines += ["", f"![AUROC differences]({figure_name})", "",
              "## Absolute AUROC", "", "Mean ± sample SD over three seeds, in percent.", "",
              "| Population | Objective | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
              "|---|---|" + "---:|" * len(METHODS)]
    for p in POPULATIONS:
        for o, label in (("ce", "CE"), ("logitnorm", "LogitNorm")):
            lines.append(f"| {LABELS[p]} | {label} | " +
                         " | ".join(pct(metrics[o][p][m]["auroc"]) for m in METHODS) + " |")
    lines += ["", "## Fixed 95% ID-validation operating point", "",
              "OOD acceptance (lower is better) at each model's own frozen 1,500-row ID threshold, mean ± sample SD, "
              "in percent. The last two columns give the lowest-acceptance method among the six for each "
              "objective.", "",
              "| Population | CE MSP | LogitNorm MSP | CE KNN-50 | LogitNorm KNN-50 | Lowest, CE | Lowest, LogitNorm |",
              "|---|---:|---:|---:|---:|---|---|"]
    for p in POPULATIONS:
        cell = lambda o, m: pct(operating[o][p][m]["ood_acceptance"])
        low = lambda o: f"{METHOD_LABELS[floor[o][p]['method']]} {100 * floor[o][p]['mean_ood_acceptance']:.2f}"
        lines.append(f"| {LABELS[p]} | {cell('ce', 'msp')} | {cell('logitnorm', 'msp')} | {cell('ce', 'knn')} | "
                     f"{cell('logitnorm', 'knn')} | {low('ce')} | {low('logitnorm')} |")
    lines += ["", "Achieved ID-test acceptance for every method and objective, all four targets, AUPR and FPR95-ID are "
              "in [metrics.csv](metrics.csv) and [summary.json](summary.json).", ""]
    return lines


def main():
    evaluation_sha, manifests = load()
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics, operating, contrasts, decisions, accuracy, floor = summarize(manifests)
    metric_rows, contrast_rows = tables(manifests)
    figure_name = figure(contrasts)
    summary = {"state": "built_pending_independent_audit", "evaluation_declaration_sha256": evaluation_sha,
               "seed_manifests": {f"seed{s}": sha(PHASE / f"score-seed{s}" / "manifest.json") for s in SEEDS},
               "decisions": decisions, "id_test_accuracy": accuracy, "contrasts": contrasts,
               "metrics": metrics, "operating": operating, "lowest_fixed95_ood_acceptance": floor,
               "csv_rows": {"metrics": metric_rows, "contrasts": contrast_rows}}
    write(REPORT / "summary.json", summary)
    lines = report_text(evaluation_sha, metrics, operating, contrasts, decisions, accuracy, floor, figure_name)
    interpretation = REPORT / "INTERPRETATION.md"
    if interpretation.exists():
        lines += interpretation.read_text(encoding="utf-8").splitlines()
    (REPORT / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({name: d["overall"] for name, d in decisions.items()}))


if __name__ == "__main__":
    main()
