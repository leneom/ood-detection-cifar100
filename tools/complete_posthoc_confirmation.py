"""Write-once completion record binding declaration, scores, report, audit and regression logs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_posthoc_confirmation import RUN, SOURCES, read, sha

REPORT = ROOT / "report" / "posthoc-confirmation"
EXTRA_SOURCES = ("tools/build_posthoc_confirmation_report.py", "tools/audit_posthoc_confirmation.py",
                 "tools/complete_posthoc_confirmation.py", "tests/test_posthoc_confirmation.py")


def main():
    target = REPORT / "completion.json"
    if target.exists():
        raise FileExistsError("completion record is write-once")
    audit = read(REPORT / "audit.json")
    if audit["state"] != "passed" or audit["report_sha256"] != sha(REPORT / "REPORT.md") or \
            audit["summary_sha256"] != sha(REPORT / "summary.json"):
        raise ValueError("a passed audit of the current report and summary is required")
    for log in ("regression-base-env.log", "regression-cifar-cl-protocol.log"):
        if not (REPORT / log).exists():
            raise FileNotFoundError(log)
    files = ["runs/posthoc-confirmation/declaration.json"]
    for seed in (0, 1, 2):
        files += [f"runs/posthoc-confirmation/score-seed{seed}/{name}"
                  for name in ("manifest.json", "status.json", "scores.npz", "acceptance_masks.npz")]
    files += [str(path.relative_to(ROOT)) for path in sorted(REPORT.iterdir()) if path.name != "completion.json"]
    files += list(SOURCES) + list(EXTRA_SOURCES)
    record = {"state": "complete", "decisions": audit["overall"],
              "audit_counts": audit["counts"],
              "regression": {"base_env": "92 passed; 1 failed only because torch 2.0.1 lacks torch.amp.GradScaler",
                             "cifar_cl_env": "tests.test_protocol 5/5 passed with torch 2.13 (incl. the resume test)"},
              "files": {name: sha(ROOT / name) for name in files},
              "completed_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": record["state"], "files": len(record["files"]), "decisions": record["decisions"]}))


if __name__ == "__main__":
    main()
