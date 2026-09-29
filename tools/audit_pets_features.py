"""Independent full-ID-fit feature-score probe audit for new Oxford Pet OOD."""
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
RUN = ROOT / "runs" / "oxford-pets-extension"
ISOLATED = ROOT / "runs" / "validation-isolation"
sys.path.insert(0, str(ROOT / "tools"))
from audit_isolated_features import fit_training_id, probe_rows, score_probes


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    seed = parser.parse_args().seed
    folder = RUN / f"extract-seed{seed}"
    manifest, status = read(folder / "manifest.json"), read(folder / "status.json")
    cache = read(RUN / "cache-manifest.json")
    if (status["state"] != "extracted_pending_independent_audit_and_scoring" or
            status["manifest_sha256"] != sha(folder / "manifest.json") or
            manifest["seed"] != seed or manifest["cache_sha256"] != cache["cache_sha256"] or
            manifest["cache_audit_sha256"] != sha(RUN / "cache-audit.json")):
        raise ValueError("complete frozen extraction and independently verified cache required")
    for name, expected in manifest["source_hashes"].items():
        if sha(folder / "source_snapshot" / name) != expected or sha(ROOT / name) != expected:
            raise ValueError("executed extraction source differs: " + name)
    for name, expected in manifest["outputs"].items():
        if sha(folder / name) != expected:
            raise ValueError("extraction output differs: " + name)
    raw, feature = arrays(folder / "raw.npz"), arrays(folder / "feature_scores.npz")
    cached = arrays(ROOT / "data" / "pets-test.npz")
    if (not np.array_equal(raw["source_rows"], np.arange(3669)) or
            not np.array_equal(raw["filenames"], cached["filenames"]) or
            not np.array_equal(raw["labels"], cached["labels"]) or
            raw["features"].shape != (3669, 512) or
            raw["logits"].shape != (3669, 100) or
            not np.array_equal(raw["predictions"], raw["logits"].argmax(axis=1))):
        raise ValueError("OOD raw alignment or shapes differ")
    old = ISOLATED / f"extract-seed{seed}"
    old_manifest = read(old / "manifest.json")
    if (manifest["training_id_bank_sha256"] != sha(old / "id_train_raw.npz") or
            old_manifest["outputs"]["id_train_raw.npz"] != manifest["training_id_bank_sha256"]):
        raise ValueError("45k training-ID feature bank differs")
    training = arrays(old / "id_train_raw.npz")
    original = arrays(ROOT / f"runs/cifar100-seed{seed}/split_indices.npz")
    if not np.array_equal(training["original_indices"], original["train"]):
        raise ValueError("training feature indices differ from original 45k ID split")
    bank, means, precision, scale = fit_training_id(training["features"], training["labels"])
    rows = probe_rows(3669)
    knn, maha = score_probes(raw["features"][rows], bank, means, precision)
    if (feature["knn"].shape != (3669,) or feature["mahalanobis"].shape != (3669,) or
            not np.allclose(knn, feature["knn"][rows], atol=1e-4, rtol=1e-5) or
            not np.allclose(maha, feature["mahalanobis"][rows], atol=1e-3, rtol=1e-5)):
        raise ValueError("independent KNN/Mahalanobis query probes differ")
    destination = RUN / f"feature-audit-seed{seed}"
    if destination.exists():
        raise FileExistsError("feature audit already exists")
    destination.mkdir()
    shutil.copyfile(__file__, destination / "executed_audit_source.py")
    result = {"state": "passed", "seed": seed, "fit_population": "45k ID training features only",
              "probe_rows": rows.tolist(), "probe_count": len(rows),
              "knn_max_abs_error": float(np.max(np.abs(knn-feature["knn"][rows]))),
              "mahalanobis_max_abs_error": float(np.max(np.abs(maha-feature["mahalanobis"][rows]))),
              "pooled_covariance_mean_diagonal": scale,
              "extraction_manifest_sha256": sha(folder / "manifest.json"),
              "cache_manifest_sha256": sha(RUN / "cache-manifest.json"),
              "fit_math_source_sha256": sha(ROOT / "tools/audit_isolated_features.py"),
              "source_sha256": sha(__file__),
              "audited_at": datetime.now(timezone.utc).isoformat(),
              "boundary": "Independent training-ID refit and OOD query probes; full raw score audit follows."}
    (destination / "audit.json").write_text(json.dumps(result, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"state": "passed", "seed": seed, "probes": len(rows)}), flush=True)


if __name__ == "__main__":
    main()

