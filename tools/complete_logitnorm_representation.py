"""Write-once completion record for the LogitNorm representation study."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_logitnorm_representation import EVALUATION_SOURCES, PHASE, read, sha

REPORT = ROOT / "report" / "logitnorm-representation"
EXTRA = ("tools/audit_logitnorm_representation.py", "tools/complete_logitnorm_representation.py",
         "tests/test_logitnorm.py")


def main():
    target = REPORT / "completion.json"
    if target.exists():
        raise FileExistsError("completion record is write-once")
    audit = read(REPORT / "audit.json")
    if audit["state"] != "passed" or audit["report_sha256"] != sha(REPORT / "REPORT.md") or \
            audit["summary_sha256"] != sha(REPORT / "summary.json"):
        raise ValueError("a passed audit of the current report and summary is required")
    files = ["runs/logitnorm-representation/declaration.json",
             "runs/logitnorm-representation/evaluation-declaration.json",
             "runs/logitnorm-representation/pause-1.json",
             "runs/logitnorm-representation/train-seed0.resume-1.json"]
    for seed in (0, 1, 2):
        files += [f"runs/logitnorm-representation/seed{seed}/{name}" for name in
                  ("final-best.pt", "final-selection.json", "history.json", "config.json", "environment.json",
                   "source_hashes.json", "split_indices.npz", "status.json")]
        extract = PHASE / f"extract-seed{seed}"
        files += [str((extract / name).relative_to(ROOT)) for name in ["manifest.json", "status.json",
                                                                        *read(extract / "manifest.json")["outputs"]]]
        files += [f"runs/logitnorm-representation/score-seed{seed}/{name}" for name in ("manifest.json", "status.json")]
    files += [str(p.relative_to(ROOT)) for p in sorted(REPORT.iterdir()) if p.name != "completion.json"]
    training = read(PHASE / "declaration.json")["training_sources"]
    files += sorted(set(training) | set(EVALUATION_SOURCES) | set(EXTRA))
    record = {"state": "complete", "decisions": audit["overall"], "audit_counts": audit["counts"],
              "selected_epochs": {f"seed{s}": read(PHASE / f"seed{s}" / "final-selection.json")["epoch"]
                                  for s in (0, 1, 2)},
              "interruption": "seed 0 paused after epoch 80 and resumed unchanged from last.pt",
              "regression": {"base_env": "96 passed; 1 failed only because torch 2.0.1 lacks torch.amp.GradScaler",
                             "cifar_cl_env": "tests.test_protocol + tests.test_logitnorm 9/9 passed (torch 2.13)"},
              "files": {name: sha(ROOT / name) for name in files},
              "completed_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": record["state"], "files": len(record["files"]), "decisions": record["decisions"]}))


if __name__ == "__main__":
    main()
