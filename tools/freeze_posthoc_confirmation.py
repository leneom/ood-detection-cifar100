"""Write-once declaration binding the confirmation protocol, sources and inputs.

Run before any ASH/RMD score exists for the isolated classifiers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_posthoc_confirmation import RUN, SOURCES, seed_inputs, sha


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / "declaration.json"
    if target.exists():
        raise FileExistsError("declaration is write-once")
    if any(RUN.glob("score-seed*")):
        raise RuntimeError("scores already exist; the declaration would not precede scoring")
    declaration = {
        "state": "frozen_before_scoring",
        "protocol": "docs/POSTHOC_CONFIRMATION.md",
        "hypotheses": {
            "H1": "AUROC(ASH-S, p=65) - AUROC(T=1 Energy) > 0",
            "H2": "AUROC(RMD) - AUROC(matched CPU MD) > 0"},
        "decision_rule": ("supported on a population iff all three seed differences > 0; contradicted iff all < 0; "
                          "otherwise mixed. Confirmed iff supported on all four primary populations; partially "
                          "confirmed iff supported on 1-3 and contradicted on none; otherwise not confirmed."),
        "primary_populations": ["tinyimagenet", "fashionmnist", "places365", "oxford_pets"],
        "secondary_populations": ["cifar10", "mnist", "svhn", "textures"],
        "sources": {name: sha(ROOT / name) for name in SOURCES},
        "inputs": {f"seed{seed}": {path: sha(ROOT / path) for path in seed_inputs(seed).values()}
                   for seed in (0, 1, 2)},
        "shared_inputs": {"runs/validation-isolation/partition.npz":
                          sha(ROOT / "runs/validation-isolation/partition.npz")},
        "created_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": declaration["state"], "declaration_sha256": sha(target)}))


if __name__ == "__main__":
    main()
