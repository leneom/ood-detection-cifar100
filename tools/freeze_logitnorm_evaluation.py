"""Write-once evaluation declaration: binds extraction/scoring/report code before any LogitNorm OOD output."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from score_logitnorm_representation import EVALUATION_SOURCES, PHASE, sha


def main():
    target = PHASE / "evaluation-declaration.json"
    if target.exists():
        raise FileExistsError("evaluation declaration is write-once")
    if any(PHASE.glob("extract-seed*")) or any(PHASE.glob("score-seed*")):
        raise RuntimeError("LogitNorm outputs already exist; the declaration would not precede them")
    declaration = {"state": "frozen_before_evaluation",
                   "training_declaration_sha256": sha(PHASE / "declaration.json"),
                   "sources": {name: sha(ROOT / name) for name in EVALUATION_SOURCES},
                   "created_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": declaration["state"], "evaluation_declaration_sha256": sha(target)}))


if __name__ == "__main__":
    main()
