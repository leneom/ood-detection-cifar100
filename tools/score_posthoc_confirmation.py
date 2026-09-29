"""Score the frozen ASH-S / RMD contrast confirmation for one isolated seed.

Every input and source must match runs/posthoc-confirmation/declaration.json.
CPU float64 only; no OOD row fits anything. See docs/POSTHOC_CONFIRMATION.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "posthoc-confirmation"
ISOLATED = ROOT / "runs" / "validation-isolation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ash_core import ash_scores
from calibration_core import logit_scores
from rmd_core import fit_rmd, rmd_scores
from src.ood.metrics import detection_metrics, id_threshold

PRIMARY = {"tinyimagenet": "tinyimagenet-confirmation", "fashionmnist": "fashionmnist-extension",
           "places365": "places365-extension", "oxford_pets": "oxford-pets-extension"}
SECONDARY = ("cifar10", "mnist", "svhn", "textures")
POPULATIONS = (*PRIMARY, *SECONDARY)
NEW = ("ash_p_energy", "ash_b_energy", "ash_s_energy", "rmd", "matched_cpu_md")
REFERENCES = ("energy", "mahalanobis", "knn")
METHODS = (*NEW, *REFERENCES)
TARGETS = (80, 90, 95, 99)
LOGIT_TOLERANCE = 1e-4
MD_RELATIVE_TOLERANCE = 1e-9
ENERGY_TOLERANCE = 2e-5
SOURCES = ("docs/POSTHOC_CONFIRMATION.md", "tools/score_posthoc_confirmation.py",
           "tools/freeze_posthoc_confirmation.py", "tools/ash_core.py", "tools/rmd_core.py",
           "tools/calibration_core.py", "src/ood/metrics.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def save(path, **content):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **content)
    temp.replace(path)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def seed_inputs(seed):
    """Relative paths of every file this seed reads, keyed by role."""
    extract, score = f"runs/validation-isolation/extract-seed{seed}", f"runs/validation-isolation/score-seed{seed}"
    inputs = {"checkpoint": f"runs/validation-isolation/seed{seed}/final-best.pt",
              "selection": f"runs/validation-isolation/seed{seed}/final-selection.json",
              "id_train_raw": f"{extract}/id_train_raw.npz",
              "id_validation_raw": f"{extract}/id_validation_raw.npz",
              "id_test_raw": f"{extract}/id_test_raw.npz",
              "id_validation_scores": f"{score}/id_validation_scores.npz",
              "id_test_scores": f"{score}/id_test_scores.npz",
              "isolated_score_manifest": f"{score}/manifest.json"}
    for population, folder in PRIMARY.items():
        inputs[f"{population}_raw"] = f"runs/{folder}/extract-seed{seed}/raw.npz"
        inputs[f"{population}_scores"] = f"runs/{folder}/score-seed{seed}/scores.npz"
    for population in SECONDARY:
        inputs[f"{population}_raw"] = f"{extract}/{population}_raw.npz"
        inputs[f"{population}_scores"] = f"{score}/{population}_scores.npz"
    return inputs


def checked_declaration(seed):
    declaration = read(RUN / "declaration.json")
    if declaration["state"] != "frozen_before_scoring":
        raise ValueError("frozen declaration required")
    for name, expected in declaration["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("declared source changed: " + name)
    declared = declaration["inputs"][f"seed{seed}"]
    if set(declared) != set(seed_inputs(seed).values()):
        raise ValueError("declared input set differs")
    for name, expected in {**declared, **declaration["shared_inputs"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError("declared input changed: " + name)
    return declaration


def head(checkpoint, selection):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if sha(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen selection")
    return (state["fc.weight"].double().numpy(), state["fc.bias"].double().numpy())


def new_scores(raw, weight, bias, rmd_state):
    features = raw["features"]
    logits = features.astype(np.float64) @ weight.T + bias
    gap = float(np.abs(logits - raw["logits"]).max())
    if gap > LOGIT_TOLERANCE:
        raise ValueError(f"head does not reproduce saved logits ({gap})")
    ash, middle = ash_scores(features, weight, bias)
    rmd, _, diagnostics = rmd_scores(features, rmd_state)
    scores = {**ash, "rmd": rmd["rmd"], "matched_cpu_md": rmd["mahalanobis_cpu"]}
    extra = {"ash_s_predictions": middle["ash_s_energy_logits"].argmax(1),
             "ash_p_predictions": middle["ash_p_energy_logits"].argmax(1),
             "ash_b_predictions": middle["ash_b_energy_logits"].argmax(1)}
    agreement = int((logits.argmax(1) == raw["logits"].argmax(1)).sum())
    return scores, extra, {"max_logit_reconstruction_gap": gap, "reconstructed_argmax_agreement": agreement,
                           "rows": len(features), **diagnostics}


def check_references(raw, saved, scores, label):
    energy = logit_scores(raw["logits"], 1.)["energy"]
    energy_gap = float(np.abs(energy - saved["T1_energy"]).max())
    if energy_gap > ENERGY_TOLERANCE:
        raise ValueError(f"{label}: saved Energy differs ({energy_gap})")
    reference = saved["T1_mahalanobis"].astype(np.float64)
    relative = np.abs(scores["matched_cpu_md"] - reference) / np.maximum(np.abs(reference), 1e-12)
    if relative.max() > MD_RELATIVE_TOLERANCE:
        raise ValueError(f"{label}: matched MD differs from saved pooled Mahalanobis ({relative.max()})")
    return {"max_energy_gap": energy_gap, "max_md_relative_gap": float(relative.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    declaration = checked_declaration(seed)
    paths = {role: ROOT / path for role, path in seed_inputs(seed).items()}
    selection = read(paths["selection"])
    weight, bias = head(paths["checkpoint"], selection)
    isolated_manifest = read(paths["isolated_score_manifest"])
    if isolated_manifest["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise ValueError("isolated scores come from another checkpoint")
    threshold_rows = arrays(ISOLATED / "partition.npz")["threshold_rows"]
    if len(threshold_rows) != 1500:
        raise ValueError("frozen 1500-row threshold role differs")

    destination = RUN / f"score-seed{seed}"
    if destination.exists():
        raise FileExistsError("confirmation scores already exist for this seed")
    destination.mkdir(parents=True)
    snapshots = destination / "source_snapshot"
    for name in SOURCES:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(destination / "status.json", {"state": "scoring", "seed": seed,
                                        "started_at": datetime.now(timezone.utc).isoformat()})

    train = arrays(paths["id_train_raw"])
    fit_metadata, rmd_state = fit_rmd(train["features"], train["labels"])
    checks, score_arrays, extras = {}, {}, {}
    id_scores = {}
    for role in ("id_test", "id_validation"):
        raw = arrays(paths[f"{role}_raw"])
        scores, extra, diagnostics = new_scores(raw, weight, bias, rmd_state)
        saved = arrays(paths[f"{role}_scores"])
        checks[role] = {**diagnostics, **check_references(raw, saved, scores, role)}
        scores.update({name: saved[f"T1_{name}"].astype(np.float64) for name in REFERENCES})
        id_scores[role] = scores
        if role == "id_test":
            labels = raw["labels"]
            original = raw["logits"].argmax(1)
            extras["id_test"] = {
                "original_accuracy": float((original == labels).mean()),
                **{f"{variant}_accuracy": float((extra[f"{variant}_predictions"] == labels).mean())
                   for variant in ("ash_p", "ash_b", "ash_s")},
                **{f"{variant}_changed_predictions": int((extra[f"{variant}_predictions"] != original).sum())
                   for variant in ("ash_p", "ash_b", "ash_s")}}
            for name in NEW:
                score_arrays[f"id_test_{name}"] = scores[name]
        else:
            for name in NEW:
                score_arrays[f"id_threshold_{name}"] = scores[name][threshold_rows]

    thresholds = {}
    frozen = isolated_manifest["datasets"]["cifar10"]["methods"]["T1"]
    for method in METHODS:
        calibration = id_scores["id_validation"][method][threshold_rows]
        thresholds[method] = {t: id_threshold(calibration, t / 100) for t in TARGETS}
        if method in REFERENCES:
            for t in TARGETS:
                if thresholds[method][t] != frozen[method]["operating_grid"][f"target{t}"]["threshold"]:
                    raise ValueError(f"reference threshold differs from frozen isolated value: {method}")

    masks, results = {}, {}
    inside_all = id_scores["id_test"]
    for population in POPULATIONS:
        raw = arrays(paths[f"{population}_raw"])
        scores, _, diagnostics = new_scores(raw, weight, bias, rmd_state)
        saved = arrays(paths[f"{population}_scores"])
        checks[population] = {**diagnostics, **check_references(raw, saved, scores, population)}
        scores.update({name: saved[f"T1_{name}"].astype(np.float64) for name in REFERENCES})
        for name in NEW:
            score_arrays[f"{population}_{name}"] = scores[name]
        results[population] = {}
        for method in METHODS:
            inside, outside = inside_all[method], scores[method]
            metrics = detection_metrics(inside, outside)
            calibration = id_scores["id_validation"][method][threshold_rows]
            grid = {}
            for t in TARGETS:
                threshold = thresholds[method][t]
                id_mask = inside <= threshold
                ood_mask = outside <= threshold
                masks[f"{population}_{method}_target{t}_OOD"] = ood_mask.astype(np.uint8)
                if population == POPULATIONS[0]:
                    masks[f"id_test_{method}_target{t}"] = id_mask.astype(np.uint8)
                grid[f"target{t}"] = {"threshold": threshold,
                                      "calibration_accepted": int((calibration <= threshold).sum()),
                                      "calibration_count": len(calibration),
                                      "id_test_accepted": int(id_mask.sum()), "id_test_count": len(inside),
                                      "id_test_acceptance": float(id_mask.mean()),
                                      "ood_accepted": int(ood_mask.sum()), "ood_count": len(outside),
                                      "ood_acceptance": float(ood_mask.mean())}
            results[population][method] = {"metrics": metrics, "operating_grid": grid}

    save(destination / "scores.npz", **score_arrays)
    save(destination / "acceptance_masks.npz", **masks)
    outputs = {name: sha(destination / name) for name in ("scores.npz", "acceptance_masks.npz")}
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    if source_hashes != declaration["sources"]:
        raise RuntimeError("source changed during scoring")
    manifest = {"state": "scored_pending_independent_audit", "seed": seed,
                "checkpoint_sha256": selection["checkpoint_sha256"], "selected_epoch": selection["epoch"],
                "declaration_sha256": sha(RUN / "declaration.json"),
                "primary_populations": list(PRIMARY), "secondary_populations": list(SECONDARY),
                "methods": list(METHODS), "targets": list(TARGETS),
                "rmd_fit": fit_metadata, "id_side_effects": extras["id_test"],
                "reconstruction_checks": checks, "results": results,
                "source_hashes": source_hashes, "outputs": outputs,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "boundary": "Frozen isolated models and ID-only thresholds; no OOD-based fitting."}
    write(destination / "manifest.json", manifest)
    write(destination / "status.json", {"state": manifest["state"], "seed": seed,
                                        "manifest_sha256": sha(destination / "manifest.json")})
    print(json.dumps({"state": manifest["state"], "seed": seed, "populations": len(POPULATIONS)}), flush=True)


if __name__ == "__main__":
    main()
