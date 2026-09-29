"""Read-only verifier for the completed three-way ID-validation isolation study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "validation-isolation"
REPORT = ROOT / "report" / "validation-isolation"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    completed = read(REPORT / "completion.json")
    if completed["state"] != "complete" or completed["three_full_training_seeds"] != [0, 1, 2]:
        raise ValueError("three-seed completion status differs")
    inputs = {"training_status": PHASE / "status.json",
              "evaluation_status": PHASE / "evaluation-status.json",
              "evaluation_launch": PHASE / "evaluation-launch.json",
              "summary": REPORT / "summary.json",
              "summary_audit": REPORT / "summary-audit.json",
              "render_status": REPORT / "render-status.json",
              "visual_audit": REPORT / "visual-audit.json",
              "figure_inspection": REPORT / "figure-inspection.json"}
    for seed in range(3):
        for prefix in ("train-audit", "feature-audit", "audit"):
            inputs[f"{prefix}_seed{seed}"] = PHASE / f"{prefix}-seed{seed}" / "audit.json"
    if set(completed["inputs_sha256"]) != set(inputs):
        raise ValueError("completion input set differs")
    for name, path in inputs.items():
        if digest(path) != completed["inputs_sha256"][name]:
            raise ValueError("completed source changed: " + name)
    if (read(inputs["training_status"])["state"] != "training_complete_pending_evaluation" or
            read(inputs["evaluation_status"])["state"] != "all_scored_and_audited_pending_summary" or
            read(inputs["summary"])["state"] != "summary_pending_independent_report_audit" or
            read(inputs["summary_audit"])["state"] != "passed" or
            read(inputs["visual_audit"])["state"] != "passed_pending_human_figure_inspection" or
            read(inputs["figure_inspection"])["state"] != "passed"):
        raise ValueError("completion stage state differs")
    for seed in range(3):
        for prefix in ("train-audit", "feature-audit", "audit"):
            record = read(inputs[f"{prefix}_seed{seed}"])
            if record["state"] != "passed" or record["seed"] != seed:
                raise ValueError("per-seed independent audit differs")
    if (completed["report_cells"] != 72 or completed["independent_summary_fields"] != 1296 or
            completed["independent_plot_coordinates"] != 120 or
            completed["regression"]["total"] != 91 or
            completed["regression"]["exit_code"] != 0):
        raise ValueError("completion counts differ")
    print(json.dumps({"state": "verified_readonly", "bound_inputs": len(inputs),
                      "three_seeds": True, "report_cells": 72,
                      "boundary": completed["boundary"]}), flush=True)


if __name__ == "__main__":
    main()

