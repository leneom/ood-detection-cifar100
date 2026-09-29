"""Write-once Places365 population declaration, before any archive access."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
SOURCES = (
    "docs/PLACES365_EXTENSION.md",
    "report/validation-isolation/completion.json",
    "report/tinyimagenet-confirmation/completion.json",
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
        "data/places365-val_256.tar", "data/places365-filelist_places365-standard.tar",
        "data/places365-categories_places365.txt", "data/places365-validation.npz")):
        raise FileExistsError("Places365 archive or run state already exists; inspect before freeze")
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
        "state": "protocol_frozen_before_places365_archive_access",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashes["docs/PLACES365_EXTENSION.md"],
        "source_sha256": sha(Path(__file__)),
        "input_sha256": hashes,
        "source_urls": {
            "images": "http://data.csail.mit.edu/places/places365/val_256.tar",
            "devkit": "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
            "categories": "http://data.csail.mit.edu/places/places365/categories_places365.txt",
        },
        "expected_md5": {
            "images": "e27b17d8d44f4af9a78502beb927f808",
            "devkit": "35a0585fee1fa656440f3ab298f8479c",
            "annotations": "e9f2fd57bfd9d07630173f4e8708e4b1",
            "categories": "06c963b85866bd0649f97cb43dd16673",
        },
        "population": "Places365-Standard official validation all 36500 images",
        "seeds": [0, 1, 2],
        "selected_epochs": [92, 96, 98],
        "methods": ["msp", "energy", "maxlogit", "entropy", "knn", "mahalanobis"],
        "variants": ["T1", "calibrated"],
        "id_validation_targets": [80, 90, 95, 99],
        "boundary": "Sequential additional OOD population; no Places365 archive/image/score accessed yet.",
    }
    RUN.mkdir(parents=True)
    with (RUN / "declaration.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"state": record["state"], "inputs": len(hashes),
                      "declaration_sha256": sha(RUN / "declaration.json")}))


if __name__ == "__main__":
    main()
