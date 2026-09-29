"""CPU-only preflight for the frozen Fashion-MNIST GPU extraction inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "fashionmnist-extension"
ISOLATED = ROOT / "runs" / "validation-isolation"
CACHE = ROOT / "data" / "fashionmnist-test-32x32.npz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def main() -> None:
    output = RUN / "cpu-preflight.json"
    check(not output.exists(), "preflight evidence already exists")
    launch = read(RUN / "extraction-launch.json")
    declaration = read(RUN / "declaration.json")
    cache_manifest = read(RUN / "cache-manifest.json")
    cache_audit = read(RUN / "cache-audit.json")
    check(launch["state"] == "launched" and launch["seeds"] == [0, 1, 2] and
          launch["extract_source_sha256"] == sha(ROOT / "tools/extract_fashionmnist.py") and
          launch["cache_manifest_sha256"] == sha(RUN / "cache-manifest.json") and
          launch["cache_audit_sha256"] == sha(RUN / "cache-audit.json"),
          "frozen extraction launch differs")
    for name, digest in launch["source_dependencies_sha256"].items():
        check(sha(ROOT / name) == digest, "frozen source changed: " + name)
    check(declaration["state"] == "frozen_before_any_fashion_data_access" and
          declaration["protocol_sha256"] == sha(ROOT / "docs/FASHIONMNIST_EXTENSION.md") and
          cache_audit["state"] == "passed" and cache_audit["count"] == 10000 and
          cache_audit["cache_sha256"] == cache_manifest["cache_sha256"] == sha(CACHE) and
          cache_audit["manifest_sha256"] == sha(RUN / "cache-manifest.json") and
          declaration["previous_aggregate_completion_sha256"] ==
          sha(ROOT / "report/tinyimagenet-confirmation/completion.json") and
          declaration["previous_diagnostic_completion_sha256"] ==
          sha(ROOT / "report/tinyimagenet-class-heterogeneity/completion.json"),
          "pre-access declaration, pixel cache or prior completion differs")
    with np.load(CACHE, allow_pickle=False) as cache:
        check(cache["images"].shape == (10000, 32, 32, 3) and
              cache["images"].dtype == np.uint8 and
              np.array_equal(cache["source_rows"], np.arange(10000)) and
              cache["labels"].shape == (10000,), "Fashion cache dimensions/order")
    prior = read(ROOT / "report/validation-isolation/completion.json")
    check(prior["state"] == "complete" and prior["three_full_training_seeds"] == [0, 1, 2],
          "three full isolated classifier seeds")
    records = []
    for seed, expected_epoch in enumerate((92, 96, 98)):
        extracted = ISOLATED / f"extract-seed{seed}"
        scored = ISOLATED / f"score-seed{seed}"
        e, es = read(extracted / "manifest.json"), read(extracted / "status.json")
        s, ss = read(scored / "manifest.json"), read(scored / "status.json")
        audit = read(ISOLATED / f"audit-seed{seed}" / "audit.json")
        checkpoint = ISOLATED / f"seed{seed}" / "final-best.pt"
        bank = extracted / "id_train_raw.npz"
        check(e["seed"] == s["seed"] == seed and
              e["checkpoint_epoch"] == expected_epoch and
              e["checkpoint_sha256"] == s["checkpoint_sha256"] == sha(checkpoint) and
              es["manifest_sha256"] == sha(extracted / "manifest.json") and
              ss["manifest_sha256"] == sha(scored / "manifest.json") and
              e["outputs"]["id_train_raw.npz"] == sha(bank) and
              audit["state"] == "passed" and
              audit["scoring_manifest_sha256"] == sha(scored / "manifest.json") and
              s["temperature_fit_role"] == "ID_temperature_1500" and
              s["threshold_fit_role"] == "ID_threshold_1500",
              f"seed {seed} frozen checkpoint, bank or ID-only scoring differs")
        with np.load(bank, allow_pickle=False) as arrays:
            check(arrays["features"].shape == (45000, 512) and
                  arrays["labels"].shape == (45000,), f"seed {seed} ID bank shape")
        records.append({"seed": seed, "checkpoint_epoch": expected_epoch,
                        "checkpoint_sha256": sha(checkpoint),
                        "id_train_bank_sha256": sha(bank),
                        "id_score_manifest_sha256": sha(scored / "manifest.json"),
                        "id_score_audit_sha256": sha(ISOLATED / f"audit-seed{seed}" / "audit.json")})
    result = {"state": "passed_cpu_only", "checked_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(Path(__file__)),
              "original_extraction_launch_sha256": sha(RUN / "extraction-launch.json"),
              "fashion_cache_sha256": sha(CACHE), "seeds": records,
              "boundary": "Input integrity only; no GPU inference or Fashion score exists."}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "seeds": [r["seed"] for r in records]}))


if __name__ == "__main__":
    main()
