"""Build the ASH-S / RMD confirmation report from the three frozen seed manifests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_posthoc_confirmation import METHODS, PRIMARY, RUN, SECONDARY, TARGETS, read, sha, write

REPORT = ROOT / "report" / "posthoc-confirmation"
SEEDS = (0, 1, 2)
POPULATIONS = (*PRIMARY, *SECONDARY)
LABELS = {"tinyimagenet": "Tiny ImageNet", "fashionmnist": "Fashion-MNIST", "places365": "Places365",
          "oxford_pets": "Oxford Pets", "cifar10": "CIFAR-10", "mnist": "MNIST", "svhn": "SVHN",
          "textures": "Textures"}
METHOD_LABELS = {"ash_p_energy": "ASH-P", "ash_b_energy": "ASH-B", "ash_s_energy": "ASH-S", "rmd": "RMD",
                 "matched_cpu_md": "Matched CPU MD", "energy": "Energy (T=1)",
                 "mahalanobis": "Pooled Mahalanobis (saved)", "knn": "KNN-50"}
CONTRASTS = {"H1": ("ash_s_energy", "energy"), "H2": ("rmd", "matched_cpu_md"),
             "ASH-S - KNN": ("ash_s_energy", "knn"), "RMD - KNN": ("rmd", "knn"),
             "RMD - Energy": ("rmd", "energy")}
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


def pp(entry, digits=2):
    return f"{100 * entry['mean']:+.{digits}f} ± {100 * entry['sd']:.{digits}f}"


def pct(entry, digits=2):
    return f"{100 * entry['mean']:.{digits}f} ± {100 * entry['sd']:.{digits}f}"


def load():
    declaration_sha = sha(RUN / "declaration.json")
    manifests = {}
    for seed in SEEDS:
        folder = RUN / f"score-seed{seed}"
        status = read(folder / "status.json")
        if status["manifest_sha256"] != sha(folder / "manifest.json"):
            raise ValueError("seed manifest changed after scoring")
        manifest = read(folder / "manifest.json")
        if manifest["declaration_sha256"] != declaration_sha or manifest["seed"] != seed:
            raise ValueError("seed manifest is not bound to the frozen declaration")
        for name, expected in manifest["outputs"].items():
            if sha(folder / name) != expected:
                raise ValueError("scored output changed: " + name)
        manifests[seed] = manifest
    return declaration_sha, manifests


def summarize(manifests):
    metrics, operating, contrasts = {}, {}, {}
    for population in POPULATIONS:
        metrics[population], operating[population] = {}, {}
        for method in METHODS:
            rows = [manifests[s]["results"][population][method] for s in SEEDS]
            metrics[population][method] = {m: stats([r["metrics"][m] for r in rows]) for m in METRICS}
            operating[population][method] = {
                f"target{t}": {field: stats([r["operating_grid"][f"target{t}"][field] for r in rows])
                               for field in ("id_test_acceptance", "ood_acceptance")} for t in TARGETS}
    for name, (new, reference) in CONTRASTS.items():
        contrasts[name] = {}
        for population in POPULATIONS:
            result = lambda s, m: manifests[s]["results"][population][m]
            auroc = [result(s, new)["metrics"]["auroc"] - result(s, reference)["metrics"]["auroc"] for s in SEEDS]
            grid = lambda s, m, f: result(s, m)["operating_grid"]["target95"][f]
            ood = [grid(s, new, "ood_acceptance") - grid(s, reference, "ood_acceptance") for s in SEEDS]
            idt = [grid(s, new, "id_test_acceptance") - grid(s, reference, "id_test_acceptance") for s in SEEDS]
            entry = {"role": "primary" if population in PRIMARY else "secondary_model_replication",
                     "auroc_difference": stats(auroc), "fixed95_ood_acceptance_difference": stats(ood),
                     "fixed95_id_test_acceptance_difference": stats(idt)}
            if name in ("H1", "H2"):
                entry["sign_pattern"] = decide(auroc)
            contrasts[name][population] = entry
    decisions = {}
    for name in ("H1", "H2"):
        per = {p: contrasts[name][p]["sign_pattern"] for p in PRIMARY}
        decisions[name] = {"per_primary_population": per, "overall": overall(list(per.values())),
                           "secondary_replication_patterns": {p: contrasts[name][p]["sign_pattern"]
                                                              for p in SECONDARY}}
    side = {key: stats([manifests[s]["id_side_effects"][key] for s in SEEDS])
            for key in manifests[0]["id_side_effects"]}
    return metrics, operating, contrasts, decisions, side


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def tables(manifests, contrasts):
    metric_rows = []
    for population in POPULATIONS:
        for method in METHODS:
            for seed in SEEDS:
                result = manifests[seed]["results"][population][method]
                grid = result["operating_grid"]
                metric_rows.append([population, "primary" if population in PRIMARY else "secondary", method, seed,
                                    *[repr(result["metrics"][m]) for m in METRICS],
                                    *[repr(grid[f"target{t}"]["id_test_acceptance"]) for t in TARGETS],
                                    *[repr(grid[f"target{t}"]["ood_acceptance"]) for t in TARGETS]])
    write_csv(REPORT / "metrics.csv",
              ["population", "role", "method", "seed", *METRICS,
               *[f"id_test_acceptance_target{t}" for t in TARGETS],
               *[f"ood_acceptance_target{t}" for t in TARGETS]], metric_rows)
    contrast_rows = []
    for name, per in contrasts.items():
        for population, entry in per.items():
            for i, seed in enumerate(SEEDS):
                contrast_rows.append([name, population, entry["role"], seed,
                                      repr(entry["auroc_difference"]["seeds"][i]),
                                      repr(entry["fixed95_ood_acceptance_difference"]["seeds"][i]),
                                      repr(entry["fixed95_id_test_acceptance_difference"]["seeds"][i])])
    write_csv(REPORT / "contrasts.csv",
              ["contrast", "population", "role", "seed", "auroc_difference",
               "fixed95_ood_acceptance_difference", "fixed95_id_test_acceptance_difference"], contrast_rows)
    return len(metric_rows), len(contrast_rows)


def figure(contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    order = list(PRIMARY) + list(SECONDARY)
    positions = {p: i + (0.8 if p in SECONDARY else 0) for i, p in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=surface, sharey=True)
    titles = {"H1": "H1: ASH-S − Energy", "H2": "H2: RMD − matched MD"}
    for ax, name in zip(axes, ("H1", "H2")):
        ax.set_facecolor(surface)
        ax.axvline(0, color=muted, linewidth=1)
        for population in order:
            entry = contrasts[name][population]["auroc_difference"]
            y = positions[population]
            values = 100 * np.asarray(entry["seeds"])
            primary = population in PRIMARY
            ax.plot([100 * entry["mean"]] * 2, [y - 0.3, y + 0.3], color=ink, linewidth=2, zorder=2,
                    solid_capstyle="round")
            if primary:
                ax.scatter(values, [y] * 3, s=52, color=series, edgecolors=surface, linewidths=1.5, zorder=3)
            else:
                ax.scatter(values, [y] * 3, s=46, color=surface, edgecolors=series, linewidths=1.6, zorder=3)
        ax.set_title(titles[name], color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Paired AUROC difference (percentage points)", color=muted, fontsize=9)
        ax.grid(axis="x", color=grid, linewidth=1)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        lo, hi = ax.get_xlim()
        span = max(abs(lo), abs(hi))
        ax.set_xlim(min(lo, -0.08 * span), hi)
    axes[0].set_yticks([positions[p] for p in order], [LABELS[p] for p in order], color=ink)
    axes[0].invert_yaxis()
    divider = (positions[list(PRIMARY)[-1]] + positions[SECONDARY[0]]) / 2
    for ax in axes:
        ax.axhline(divider, color=grid, linewidth=1)
    axes[0].text(-0.02, positions[list(PRIMARY)[0]] - 0.75, "Primary (predeclared test)", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", color=muted, fontsize=8.5)
    axes[0].text(-0.02, positions[SECONDARY[0]] - 0.75, "Secondary (model replication)", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", color=muted, fontsize=8.5)
    fig.text(0.01, 0.01, "Filled dots: three seeds on predeclared populations. Hollow dots: populations used to develop the "
             "method. Black tick: seed mean.", color=muted, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = REPORT / "paired_auroc_differences.png"
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return path.name


def report_text(declaration_sha, metrics, operating, contrasts, decisions, side, figure_name):
    lines = ["# ASH-S / RMD contrast confirmation", "",
             f"Predeclared test of two exploratory post-hoc methods on four populations where these method × population "
             f"pairs had never been scored. The [frozen protocol](../../docs/POSTHOC_CONFIRMATION.md) and declaration "
             f"(SHA-256 `{declaration_sha}`) were committed before scoring. The same three ID-isolated CIFAR-100 "
             f"classifiers, their 45,000 training-ID features and their 1,500-row ID threshold role were reused. "
             f"No OOD row fitted anything.", "",
             "## Decisions", "",
             "| Hypothesis | Tiny ImageNet | Fashion-MNIST | Places365 | Oxford Pets | Overall |",
             "|---|---|---|---|---|---|"]
    names = {"H1": "H1 ASH-S > Energy", "H2": "H2 RMD > matched MD"}
    for name in ("H1", "H2"):
        per = decisions[name]["per_primary_population"]
        lines.append(f"| {names[name]} | " + " | ".join(per[p] for p in PRIMARY) +
                     f" | **{decisions[name]['overall']}** |")
    lines += ["", "\"Supported\" means all three seed differences are positive, \"contradicted\" means all are negative, "
              "and \"mixed\" covers everything else. The rule was fixed before scoring.", "",
              "## Paired AUROC differences", "",
              "Each cell is the mean ± sample SD over three training seeds, in percentage points. Seed points are in "
              "[contrasts.csv](contrasts.csv).", "",
              "| Population | Role | ASH-S − Energy | RMD − matched MD | ASH-S − KNN | RMD − KNN | RMD − Energy |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for population in POPULATIONS:
        role = "primary" if population in PRIMARY else "replication"
        lines.append(f"| {LABELS[population]} | {role} | " +
                     " | ".join(pp(contrasts[c][population]["auroc_difference"]) for c in CONTRASTS) + " |")
    lines += ["", f"![Paired AUROC differences]({figure_name})", "",
              "## Fixed 95% ID-validation operating point", "",
              "OOD acceptance difference at each method's own frozen 1,500-row ID threshold (lower is better), with "
              "the accompanying ID-test acceptance difference. These are not equal-coverage comparisons.", "",
              "| Population | ASH-S − Energy: OOD | ASH-S − Energy: ID test | RMD − MD: OOD | RMD − MD: ID test |",
              "|---|---:|---:|---:|---:|"]
    for population in POPULATIONS:
        h1, h2 = contrasts["H1"][population], contrasts["H2"][population]
        lines.append(f"| {LABELS[population]} | {pp(h1['fixed95_ood_acceptance_difference'])} | "
                     f"{pp(h1['fixed95_id_test_acceptance_difference'])} | "
                     f"{pp(h2['fixed95_ood_acceptance_difference'])} | "
                     f"{pp(h2['fixed95_id_test_acceptance_difference'])} |")
    lines += ["", "## AUROC and fixed-95% OOD acceptance by method", "",
              "Mean ± sample SD over three seeds, in percent. The full AUPR, FPR95-ID and all four targets are in "
              "[metrics.csv](metrics.csv) and [summary.json](summary.json).", ""]
    for population in POPULATIONS:
        role = "primary" if population in PRIMARY else "secondary model replication"
        lines += [f"### {LABELS[population]} ({role})", "",
                  "| Method | AUROC | Fixed95 OOD accepted | Fixed95 ID-test accepted |", "|---|---:|---:|---:|"]
        for method in METHODS:
            grid = operating[population][method]["target95"]
            lines.append(f"| {METHOD_LABELS[method]} | {pct(metrics[population][method]['auroc'])} | "
                         f"{pct(grid['ood_acceptance'])} | {pct(grid['id_test_acceptance'])} |")
        lines.append("")
    lines += ["## ID side effects of ASH", "",
              f"Original classifier ID-test accuracy {pct(side['original_accuracy'])}%. Intermediate shaped-head "
              f"accuracy: ASH-P {pct(side['ash_p_accuracy'])}%, ASH-B {pct(side['ash_b_accuracy'])}%, ASH-S "
              f"{pct(side['ash_s_accuracy'])}%. Changed ID-test argmax predictions per seed: ASH-P "
              f"{[int(x) for x in side['ash_p_changed_predictions']['seeds']]}, ASH-B {[int(x) for x in side['ash_b_changed_predictions']['seeds']]}, "
              f"ASH-S {[int(x) for x in side['ash_s_changed_predictions']['seeds']]}. Detection is always paired with the original "
              f"classifier's predictions.", ""]
    return lines


def main():
    declaration_sha, manifests = load()
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics, operating, contrasts, decisions, side = summarize(manifests)
    metric_count, contrast_count = tables(manifests, contrasts)
    figure_name = figure(contrasts)
    summary = {"state": "built_pending_independent_audit", "declaration_sha256": declaration_sha,
               "seed_manifests": {f"seed{s}": sha(RUN / f"score-seed{s}" / "manifest.json") for s in SEEDS},
               "decisions": decisions, "contrasts": contrasts, "metrics": metrics, "operating": operating,
               "id_side_effects": side, "csv_rows": {"metrics": metric_count, "contrasts": contrast_count}}
    write(REPORT / "summary.json", summary)
    lines = report_text(declaration_sha, metrics, operating, contrasts, decisions, side, figure_name)
    interpretation = REPORT / "INTERPRETATION.md"
    if interpretation.exists():
        lines += interpretation.read_text(encoding="utf-8").splitlines()
    (REPORT / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({name: d["overall"] for name, d in decisions.items()}))


if __name__ == "__main__":
    main()
