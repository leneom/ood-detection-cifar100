"""Build the frozen, post-result paired-decision Pet diagnostic."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/oxford-pets-paired-decisions"
REPORT = ROOT / "report/oxford-pets-paired-decisions"
FIELDS = ("seed", "variant", "first", "second", "target", "role", "n",
          "neither", "first_only", "second_only", "both", "first_accepted",
          "second_accepted", "delta_acceptance")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt(values: list[float]) -> str:
    return f"{100 * statistics.mean(values):.2f} ± {100 * statistics.stdev(values):.2f}"


def main() -> None:
    require(not REPORT.exists(), "write-once paired report already exists")
    d = json.loads((BASE / "declaration.json").read_text(encoding="utf-8"))
    require(d["state"] == "frozen_before_pairwise_calculation" and
            d["seeds"] == [0, 1, 2] and d["variants"] == ["T1", "calibrated"] and
            d["methods"] == ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"] and
            d["targets"] == [80, 90, 95, 99] and
            (d["id_count"], d["ood_count"]) == (10000, 3669), "frozen grid")
    require(len(d["input_sha256"]) == 15, "complete source input manifest")
    for rel, expected in d["input_sha256"].items():
        require(digest(ROOT / rel) == expected, f"source changed: {rel}")
    pairs = list(itertools.combinations(d["methods"], 2))
    rows: list[dict] = []
    arrays: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for seed in d["seeds"]:
        prior = ROOT / f"runs/validation-isolation/score-seed{seed}"
        pet = ROOT / f"runs/oxford-pets-extension/score-seed{seed}"
        manifest = json.loads((pet / "manifest.json").read_text(encoding="utf-8"))
        with (np.load(prior / "id_test_scores.npz", allow_pickle=False) as id_scores,
              np.load(pet / "scores.npz", allow_pickle=False) as ood_scores,
              np.load(pet / "acceptance_masks.npz", allow_pickle=False) as masks):
            for variant in d["variants"]:
                code_id = np.empty((len(pairs), len(d["targets"]), 10000), dtype=np.uint8)
                code_ood = np.empty((len(pairs), len(d["targets"]), 3669), dtype=np.uint8)
                for ti, target in enumerate(d["targets"]):
                    accepted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    for method in d["methods"]:
                        key = f"{variant}_{method}"
                        threshold = float(manifest["results"][variant][method]
                                          ["operating_grid"][f"target{target}"]["threshold"])
                        id_bool = np.asarray(id_scores[key] <= threshold, dtype=bool)
                        ood_bool = np.asarray(ood_scores[key] <= threshold, dtype=bool)
                        require(id_bool.shape == (10000,) and ood_bool.shape == (3669,),
                                f"source shape {seed}/{key}/{target}")
                        require(np.array_equal(id_bool, masks[f"{key}_target{target}_test_ID"]) and
                                np.array_equal(ood_bool, masks[f"{key}_target{target}_OOD"]),
                                f"original threshold masks {seed}/{key}/{target}")
                        point = manifest["results"][variant][method]["operating_grid"][f"target{target}"]
                        require(int(id_bool.sum()) == point["id_test_accepted"] and
                                int(ood_bool.sum()) == point["ood_accepted"],
                                f"original operating marginals {seed}/{key}/{target}")
                        accepted[method] = (id_bool, ood_bool)
                    for pi, (first, second) in enumerate(pairs):
                        for role, index, n, tensor in (("ID", 0, 10000, code_id),
                                                        ("OOD", 1, 3669, code_ood)):
                            a, b = accepted[first][index], accepted[second][index]
                            code = (a.astype(np.uint8) + 2 * b.astype(np.uint8))
                            tensor[pi, ti] = code
                            counts = np.bincount(code, minlength=4)
                            row = {"seed": seed, "variant": variant, "first": first,
                                   "second": second, "target": target, "role": role,
                                   "n": n, "neither": int(counts[0]),
                                   "first_only": int(counts[1]),
                                   "second_only": int(counts[2]), "both": int(counts[3]),
                                   "first_accepted": int(a.sum()),
                                   "second_accepted": int(b.sum()),
                                   "delta_acceptance": (int(counts[1]) - int(counts[2])) / n}
                            require(sum(row[k] for k in ("neither", "first_only", "second_only", "both")) == n,
                                    "complete table")
                            rows.append(row)
                arrays[seed, variant] = code_id, code_ood
    require(len(rows) == 3 * 2 * 15 * 4 * 2 == 720, "complete 720-table grid")
    aggregates = []
    for variant in d["variants"]:
        for first, second in pairs:
            for target in d["targets"]:
                entry = {"variant": variant, "first": first, "second": second, "target": target}
                for role in ("ID", "OOD"):
                    selected = [next(r for r in rows if r["seed"] == seed and r["variant"] == variant and
                                     r["first"] == first and r["second"] == second and
                                     r["target"] == target and r["role"] == role)
                                for seed in d["seeds"]]
                    entry[role] = {name: {"per_seed": [float(r[name]) for r in selected],
                                          "mean": statistics.mean(float(r[name]) for r in selected),
                                          "sample_sd": statistics.stdev(float(r[name]) for r in selected)}
                                   for name in ("first_accepted", "second_accepted", "first_only",
                                                "second_only", "both", "delta_acceptance")}
                aggregates.append(entry)
    REPORT.mkdir(parents=True)
    with (REPORT / "tables.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for (seed, variant), (id_codes, ood_codes) in arrays.items():
        np.savez_compressed(REPORT / f"codes-seed{seed}-{variant}.npz",
                            id_codes=id_codes, ood_codes=ood_codes)
    summary = {"state": "built_pending_independent_audit", "seeds": d["seeds"],
               "variants": d["variants"], "methods": d["methods"],
               "targets": d["targets"], "pairs": [list(p) for p in pairs],
               "code_meaning": {"0": "neither_accepts", "1": "first_only_accepts",
                                "2": "second_only_accepts", "3": "both_accept"},
               "declaration_sha256": digest(BASE / "declaration.json"),
               "tables": len(rows), "aggregates": aggregates}
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Oxford-IIIT Pet paired fixed-threshold decisions", "",
             "Post-result descriptive diagnostic of the complete 10,000 ID-test and 3,669 Pet-test images. "
             "Both methods in each pair use their own unchanged ID-validation threshold. "
             "A difference in acceptance is therefore not a causal attribution to ranking. "
             "No pair was selected to alter a detector.", "",
             "Each value below is a three-training-seed mean ± sample SD in percentage points "
             "at the frozen 95% ID-validation target. Positive first-minus-second means "
             "the first method accepts more samples. All 720 tables, both variants and all four "
             "targets are retained in [tables.csv](tables.csv); complete row-level codes are in six NPZ files.", "",
             "| T=1 first | Second | ID first−second | Pet first−second | Pet first only | Pet second only |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for first, second in pairs:
        e = next(x for x in aggregates if x["variant"] == "T1" and x["first"] == first and
                 x["second"] == second and x["target"] == 95)
        id_delta = e["ID"]["delta_acceptance"]["per_seed"]
        ood_delta = e["OOD"]["delta_acceptance"]["per_seed"]
        first_only = [x / 3669 for x in e["OOD"]["first_only"]["per_seed"]]
        second_only = [x / 3669 for x in e["OOD"]["second_only"]["per_seed"]]
        lines.append(f"| {first} | {second} | {fmt(id_delta)} | {fmt(ood_delta)} | "
                     f"{fmt(first_only)} | {fmt(second_only)} |")
    lines += ["", "The Entropy/KNN-50 row was motivated by the already viewed aggregate results: "
              "KNN-50 had higher Pet AUROC yet higher Pet acceptance at fixed95. "
              "Its discordant per-image cells show which accepted decisions differ; "
              "this is not a new winner-selection comparison. ID-test acceptance is shown "
              "alongside Pet acceptance because the actual coverages can differ. "
              "Three trained seeds and one sequentially chosen OOD population do not provide "
              "a confidence interval or universal generalization claim.", ""]
    (REPORT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"state": summary["state"], "tables": len(rows),
                      "pairs": len(pairs), "declaration_sha256": summary["declaration_sha256"]}))


if __name__ == "__main__":
    main()
