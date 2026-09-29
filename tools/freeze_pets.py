"""Write-once Oxford-IIIT Pet test population declaration, before any archive access."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oxford-pets-extension"
SOURCES = (
    "docs/OXFORD_PETS_EXTENSION.md",
    "report/validation-isolation/completion.json",
    "report/places365-extension/completion.json",
    "src/ood/data.py", "src/ood/features.py", "src/ood/scores.py",
    "src/ood/metrics.py", "tools/calibration_core.py",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if RUN.exists() or any((ROOT / p).exists() for p in (
        "data/pets-images.tar.gz", "data/pets-annotations.tar.gz",
        "data/pets-test.npz")):
        raise FileExistsError("Oxford Pet archive or run state already exists; inspect before freeze")
    subprocess.run([sys.executable, str(ROOT / "tools/verify_isolated_completion.py")],
                   check=True, cwd=ROOT)
    prior = json.loads((ROOT / "report/validation-isolation/completion.json").read_text(encoding="utf-8"))
    if prior.get("state") != "complete" or prior.get("three_full_training_seeds") != [0, 1, 2]:
        raise ValueError("three full ID-isolated models not complete")
    names = list(SOURCES)
    for seed in range(3):
        prefix = f"runs/validation-isolation"
        names.extend((
            f"{prefix}/seed{seed}/final-best.pt",
            f"{prefix}/extract-seed{seed}/manifest.json",
            f"{prefix}/extract-seed{seed}/id_train_raw.npz",
            f"{prefix}/score-seed{seed}/manifest.json",
            f"{prefix}/score-seed{seed}/id_test_scores.npz",
            f"{prefix}/score-seed{seed}/id_validation_scores.npz",
            f"{prefix}/audit-seed{seed}/audit.json",
        ))
    hashes = {name: sha(ROOT / name) for name in names}
    record = {
        "state": "protocol_frozen_before_pet_archive_access",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashes["docs/OXFORD_PETS_EXTENSION.md"],
        "source_sha256": sha(Path(__file__)),
        "input_sha256": hashes,
        "source_urls": {
            "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        },
        "expected_md5": {
            "images": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
            "annotations": "95a8c909bbe2e81eed6a22bccdf3f68f",
        },
        "population": "Oxford-IIIT Pet official test all 3669 images",
        "seeds": [0, 1, 2],
        "selected_epochs": [92, 96, 98],
        "methods": ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"],
        "variants": ["T1", "calibrated"],
        "id_validation_targets": [80, 90, 95, 99],
        "boundary": "Sequential additional OOD population; no pet archive/image/score accessed yet.",
    }
    RUN.mkdir(parents=True)
    with (RUN / "declaration.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"state": record["state"], "inputs": len(hashes),
                      "declaration_sha256": sha(RUN / "declaration.json")}))


if __name__ == "__main__":
    main()
