"""Independently reconstruct every frozen Oxford Pet paired acceptance decision."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "runs/oxford-pets-paired-decisions"
REPORT = ROOT / "report/oxford-pets-paired-decisions"


def check(ok, msg):
    if not ok:
        raise AssertionError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(1 << 20)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def same_float(actual, expected, label):
    check(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12), label)


def main():
    declaration = json.loads((SOURCE / "declaration.json").read_text(encoding="utf-8"))
    from tools.verify_pets_completion import validate as verify_pet
    old_completion = json.loads((ROOT / "report/oxford-pets-extension/completion.json").read_text(encoding="utf-8"))
    check(old_completion["state"] == "complete" and old_completion["inputs_sha256"] == verify_pet(),
          "original Pet completion chain")
    check(declaration["state"] == "frozen_before_pairwise_calculation", "declaration state")
    check(len(declaration["input_sha256"]) == 15, "declaration file census")
    for rel, expected in declaration["input_sha256"].items():
        check(sha(ROOT / rel) == expected, "source hash " + rel)
    expected_methods = ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"]
    check(declaration["methods"] == expected_methods and declaration["seeds"] == [0, 1, 2] and
          declaration["variants"] == ["T1", "calibrated"] and
          declaration["targets"] == [80, 90, 95, 99] and
          (declaration["id_count"], declaration["ood_count"]) == (10000, 3669), "grid")
    pairs = list(itertools.combinations(expected_methods, 2))
    summary = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    check(summary["declaration_sha256"] == sha(SOURCE / "declaration.json") and
          summary["pairs"] == [list(p) for p in pairs] and
          summary["tables"] == 720 and len(summary["aggregates"]) == 120,
          "summary metadata")
    check(summary["code_meaning"] == {"0": "neither_accepts", "1": "first_only_accepts",
                                      "2": "second_only_accepts", "3": "both_accept"}, "code meaning")
    with (REPORT / "tables.csv").open(newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    check(len(table) == 720, "CSV table census")
    by_key = {}
    for row in table:
        key = (int(row["seed"]), row["variant"], row["first"], row["second"],
               int(row["target"]), row["role"])
        check(key not in by_key, f"duplicate CSV table {key}")
        by_key[key] = row
    checked_codes = checked_masks = 0
    for seed in range(3):
        pet = ROOT / f"runs/oxford-pets-extension/score-seed{seed}"
        prior = ROOT / f"runs/validation-isolation/score-seed{seed}"
        manifest = json.loads((pet / "manifest.json").read_text(encoding="utf-8"))
        with (np.load(prior / "id_test_scores.npz", allow_pickle=False) as source_id,
              np.load(pet / "scores.npz", allow_pickle=False) as source_pet,
              np.load(pet / "acceptance_masks.npz", allow_pickle=False) as old_masks):
            for variant in ("T1", "calibrated"):
                with np.load(REPORT / f"codes-seed{seed}-{variant}.npz", allow_pickle=False) as saved:
                    check(set(saved.files) == {"id_codes", "ood_codes"} and
                          saved["id_codes"].shape == (15, 4, 10000) and
                          saved["ood_codes"].shape == (15, 4, 3669) and
                          saved["id_codes"].dtype == saved["ood_codes"].dtype == np.uint8,
                          "complete raw code arrays")
                    for ti, target in enumerate((80, 90, 95, 99)):
                        decisions = {}
                        for method in expected_methods:
                            key = f"{variant}_{method}"
                            point = manifest["results"][variant][method]["operating_grid"][f"target{target}"]
                            limit = point["threshold"]
                            id_accept = np.less_equal(source_id[key], limit)
                            pet_accept = np.less_equal(source_pet[key], limit)
                            check(id_accept.shape == (10000,) and pet_accept.shape == (3669,), "score shapes")
                            for role, actual, n, count in (("test_ID", id_accept, 10000, "id_test_accepted"),
                                                            ("OOD", pet_accept, 3669, "ood_accepted")):
                                check(np.array_equal(actual, old_masks[f"{key}_target{target}_{role}"]) and
                                      int(np.count_nonzero(actual)) == point[count],
                                      f"old acceptance {seed}/{key}/{target}/{role}")
                                checked_masks += 1
                            decisions[method] = (id_accept, pet_accept)
                        for pi, (first, second) in enumerate(pairs):
                            for role, col, saved_name, n in (("ID", 0, "id_codes", 10000),
                                                               ("OOD", 1, "ood_codes", 3669)):
                                a, b = decisions[first][col], decisions[second][col]
                                masks = ((~a & ~b), (a & ~b), (~a & b), (a & b))
                                counts = tuple(int(np.count_nonzero(x)) for x in masks)
                                reconstructed = np.zeros(n, dtype=np.uint8)
                                reconstructed[masks[1]] = 1
                                reconstructed[masks[2]] = 2
                                reconstructed[masks[3]] = 3
                                check(np.array_equal(reconstructed, saved[saved_name][pi, ti]),
                                      f"all row codes {seed}/{variant}/{first}/{second}/{target}/{role}")
                                key = (seed, variant, first, second, target, role)
                                row = by_key.pop(key)
                                check(tuple(int(row[k]) for k in
                                            ("neither", "first_only", "second_only", "both")) == counts and
                                      int(row["n"]) == n and sum(counts) == n and
                                      int(row["first_accepted"]) == counts[1] + counts[3] and
                                      int(row["second_accepted"]) == counts[2] + counts[3],
                                      f"CSV count {key}")
                                same_float(row["delta_acceptance"], (counts[1] - counts[2]) / n,
                                           f"CSV delta {key}")
                                checked_codes += n
    check(not by_key, "no unknown/missing CSV table")
    report_lines = (REPORT / "REPORT.md").read_text(encoding="utf-8").splitlines()
    displayed = 0
    expected_aggregate_keys = set()
    for entry in summary["aggregates"]:
        variant, first, second, target = (entry[x] for x in ("variant", "first", "second", "target"))
        key = (variant, first, second, target)
        check(key not in expected_aggregate_keys, f"duplicate aggregate {key}")
        expected_aggregate_keys.add(key)
        for role in ("ID", "OOD"):
            for metric in ("first_accepted", "second_accepted", "first_only", "second_only", "both",
                           "delta_acceptance"):
                value = entry[role][metric]
                original = []
                for seed in range(3):
                    # Re-read the independently reconstructed count from the saved complete row codes.
                    with np.load(REPORT / f"codes-seed{seed}-{variant}.npz", allow_pickle=False) as z:
                        pi = pairs.index((first, second))
                        ti = (80, 90, 95, 99).index(target)
                        c = np.bincount(z["id_codes" if role == "ID" else "ood_codes"][pi, ti], minlength=4)
                    n = 10000 if role == "ID" else 3669
                    measured = {"first_accepted": int(c[1] + c[3]),
                                "second_accepted": int(c[2] + c[3]),
                                "first_only": int(c[1]), "second_only": int(c[2]),
                                "both": int(c[3]),
                                "delta_acceptance": (int(c[1]) - int(c[2])) / n}[metric]
                    original.append(measured)
                check(len(value["per_seed"]) == 3, "three seed points")
                for a, b in zip(value["per_seed"], original):
                    same_float(a, b, f"aggregate seed {key}/{role}/{metric}")
                same_float(value["mean"], statistics.mean(original), f"aggregate mean {key}/{role}/{metric}")
                same_float(value["sample_sd"], statistics.stdev(original),
                           f"aggregate sample SD {key}/{role}/{metric}")
        if variant == "T1" and target == 95:
            pct = lambda vals: (f"{100 * statistics.mean(vals):.2f} ± "
                                f"{100 * statistics.stdev(vals):.2f}")
            id_delta = entry["ID"]["delta_acceptance"]["per_seed"]
            ood_delta = entry["OOD"]["delta_acceptance"]["per_seed"]
            first_only = [x / 3669 for x in entry["OOD"]["first_only"]["per_seed"]]
            second_only = [x / 3669 for x in entry["OOD"]["second_only"]["per_seed"]]
            expected = f"| {first} | {second} | {pct(id_delta)} | {pct(ood_delta)} | "
            expected += f"{pct(first_only)} | {pct(second_only)} |"
            check(expected in report_lines, f"display row {first}/{second}")
            displayed += 1
    check(expected_aggregate_keys == {(v, a, b, t) for v in ("T1", "calibrated")
                                      for a, b in pairs for t in (80, 90, 95, 99)} and
          displayed == 15, "complete summary and display grid")
    output = {"state": "independently_audited", "source_sha256": sha(SOURCE / "declaration.json"),
              "checked_tables": 720, "checked_row_codes": checked_codes,
              "checked_source_masks": checked_masks, "checked_aggregates": 120,
              "checked_display_rows": displayed}
    path = REPORT / "audit.json"
    if path.exists():
        check(json.loads(path.read_text(encoding="utf-8")) == output,
              "read-only audit matches its retained result")
    else:
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
