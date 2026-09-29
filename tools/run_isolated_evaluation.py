"""Sequentially extract, score and audit all frozen isolation classifiers.

Launch only after the training controller exits and all three 100-epoch runs
are complete. One subprocess at a time; preserved partial outputs require review.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "runs" / "validation-isolation"
CPU_PYTHON = Path(r"C:\Users\zxzx\anaconda3\python.exe")
STAGES = (("extract", "tools/extract_isolated_eval.py", "gpu"),
          ("feature-audit", "tools/audit_isolated_features.py", "cpu"),
          ("score", "tools/score_isolated_eval.py", "cpu"),
          ("score-audit", "tools/audit_isolated_scoring.py", "cpu"))
DEPENDENCIES = ("docs/VALIDATION_ISOLATION.md", "tools/calibration_core.py",
                "tools/audit_isolated_training.py",
                "src/ood/data.py", "src/ood/datasets.py", "src/ood/cache.py",
                "src/ood/textures_cache.py", "src/ood/evaluate.py", "src/ood/features.py",
                "src/ood/models.py", "src/ood/runtime.py", "src/ood/metrics.py",
                "src/ood/scores.py", "src/ood/isolated_train.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def output_for(stage, seed):
    prefix = {"extract": "extract", "feature-audit": "feature-audit",
              "score": "score", "score-audit": "audit"}[stage]
    return PHASE / f"{prefix}-seed{seed}"


def verified_stage(stage, seed):
    folder = output_for(stage, seed)
    if not folder.exists():
        return False
    if stage in ("extract", "score"):
        status = read(folder / "status.json")
        manifest = read(folder / "manifest.json")
        expected = "extracted_pending_CPU_calibration_and_audit" if stage == "extract" else "scored_pending_independent_audit"
        if status["state"] != expected or status["manifest_sha256"] != digest(folder / "manifest.json"):
            raise ValueError(f"partial or altered {stage} output for seed {seed}")
        for name, value in manifest["outputs"].items():
            if digest(folder / name) != value:
                raise ValueError(f"altered {stage} output: {name}")
    else:
        if read(folder / "audit.json")["state"] != "passed":
            raise ValueError(f"failed {stage} output for seed {seed}")
    return True


def run_stage(stage, script, mode, seed, source_hashes):
    for name, expected in source_hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError("evaluation source changed after launch: " + name)
    if verified_stage(stage, seed):
        return
    interpreter = sys.executable if mode == "gpu" else str(CPU_PYTHON)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    if mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logbase = PHASE / f"{stage}-seed{seed}"
    command = [interpreter, "-X", "utf8", script, "--seed", str(seed)]
    write(PHASE / "evaluation-status.json", {"state": "evaluating", "seed": seed,
          "stage": stage, "controller_pid": os.getpid(), "updated_at": now()})
    with logbase.with_suffix(".stdout.log").open("a", encoding="utf-8") as out, \
         logbase.with_suffix(".stderr.log").open("a", encoding="utf-8") as err:
        out.write(json.dumps({"stage_started_at": now(), "stage": stage, "seed": seed}) + "\n")
        out.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, check=True)
    if not verified_stage(stage, seed):
        raise ValueError(f"stage exited without complete output: {stage}/seed{seed}")


def run_all():
    state = read(PHASE / "status.json")
    if (state["state"] != "training_complete_pending_evaluation" or
            state["trainer_running"] or len(state["completed_seeds"]) != 3):
        raise RuntimeError("all three full training jobs must finish before evaluation")
    for seed in range(3):
        audit_path = PHASE / f"train-audit-seed{seed}" / "audit.json"
        audit = read(audit_path)
        if (audit["state"] != "passed" or audit["seed"] != seed or
                audit["training_epoch_count"] != 100 or
                audit["audit_source_sha256"] != digest(PHASE / f"train-audit-seed{seed}" / "executed_audit_source.py") or
                audit["audit_source_sha256"] != digest(ROOT / "tools/audit_isolated_training.py") or
                audit["selected_checkpoint_sha256"] != state["completed_seeds"][seed]["final_checkpoint_sha256"]):
            raise ValueError(f"independent frozen training audit missing or changed for seed {seed}")
    if not CPU_PYTHON.is_file():
        raise FileNotFoundError("verified SciPy CPU Python missing")
    launch_path = PHASE / "evaluation-launch.json"
    if launch_path.exists():
        raise FileExistsError("evaluation already launched; inspect status and outputs before recovery")
    sources = {name: digest(ROOT / name) for name in
               [script for _, script, _ in STAGES] + list(DEPENDENCIES)}
    sources["tools/run_isolated_evaluation.py"] = digest(__file__)
    launch = {"state": "launched", "launched_at": now(), "gpu_python": sys.executable,
              "cpu_python": str(CPU_PYTHON), "seeds": [0, 1, 2],
              "stage_order": [stage for stage, _, _ in STAGES], "source_hashes": sources,
              "training_launch_sha256": digest(PHASE / "training-launch.json"),
              "partition_sha256": digest(PHASE / "partition.npz"),
              "boundary": "One GPU extraction subprocess at a time after all three frozen training schedules; CPU fitting uses disjoint ID roles."}
    write(launch_path, launch)
    for seed in range(3):
        for stage, script, mode in STAGES:
            run_stage(stage, script, mode, seed, sources)
    write(PHASE / "evaluation-status.json", {"state": "all_scored_and_audited_pending_summary",
          "completed_seeds": [0, 1, 2], "controller_pid": os.getpid(),
          "evaluation_launch_sha256": digest(launch_path), "updated_at": now()})


def main():
    PHASE.mkdir(parents=True, exist_ok=True)
    with (PHASE / "evaluation-controller.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            run_all()
        except Exception as exc:
            write(PHASE / "evaluation-status.json", {"state": "evaluation_failed",
                  "error": repr(exc), "traceback": traceback.format_exc(),
                  "controller_pid": os.getpid(), "updated_at": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
