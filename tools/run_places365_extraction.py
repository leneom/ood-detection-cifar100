"""One-at-a-time GPU extraction of the frozen Places365 validation population."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
SOURCE = ROOT / "tools" / "extract_places365.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    lock_path = RUN / "extraction-controller.lock"
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if (RUN / "extraction-launch.json").exists():
                raise FileExistsError("Places365 extraction already launched")
            cache = read(RUN / "cache-manifest.json")
            audit = read(RUN / "cache-audit.json")
            if (cache["state"] != "verified_source_and_pixels_pending_independent_audit" or
                    audit["state"] != "passed" or audit["cache_sha256"] != cache["cache_sha256"] or
                    audit["manifest_sha256"] != sha(RUN / "cache-manifest.json")):
                raise ValueError("independent cache integrity audit missing")
            for seed in range(3):
                if (RUN / f"extract-seed{seed}").exists():
                    raise FileExistsError(f"existing seed {seed} extraction needs review")
            source_hash = sha(SOURCE)
            source_deps = {name: sha(ROOT / name) for name in
                           ("src/ood/features.py", "src/ood/models.py", "src/ood/runtime.py",
                            "src/ood/data.py", "src/ood/scores.py",
                            "tools/prepare_places365.py", "tools/audit_places365_cache.py",
                            "tools/freeze_places365.py",
                            "docs/PLACES365_EXTENSION.md",
                            "docs/PLACES365_AMENDMENT_1.md")}
            launch = {"state": "launched", "launched_at": now(), "pid": os.getpid(),
                      "python": sys.executable, "seeds": [0, 1, 2],
                      "extract_source_sha256": source_hash,
                      "controller_source_sha256": sha(Path(__file__)),
                      "source_dependencies_sha256": source_deps,
                      "cache_manifest_sha256": sha(RUN / "cache-manifest.json"),
                      "cache_audit_sha256": sha(RUN / "cache-audit.json"),
                      "boundary": "One frozen checkpoint GPU extraction at a time; no OOD fitting."}
            write(RUN / "extraction-launch.json", launch)
            for seed in range(3):
                if sha(SOURCE) != source_hash or any(sha(ROOT / name) != value
                                                     for name, value in source_deps.items()):
                    raise RuntimeError("source changed after launch")
                write(RUN / "extraction-status.json", {"state": "extracting", "seed": seed,
                                                       "pid": os.getpid(), "updated_at": now()})
                env = os.environ.copy()
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "4",
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
                stdout = RUN / f"extract-seed{seed}.stdout.log"
                stderr = RUN / f"extract-seed{seed}.stderr.log"
                with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                    subprocess.run([sys.executable, "-X", "utf8",
                                    "tools/extract_places365.py", "--seed", str(seed)],
                                   cwd=ROOT, env=env, stdout=out, stderr=err, check=True)
                folder = RUN / f"extract-seed{seed}"
                status = read(folder / "status.json")
                manifest = read(folder / "manifest.json")
                if (status["state"] != "extracted_pending_independent_audit_and_scoring" or
                        status["manifest_sha256"] != sha(folder / "manifest.json") or
                        manifest["seed"] != seed or
                        manifest["source_hashes"]["tools/extract_places365.py"] != source_hash or
                        any(sha(folder / name) != digest
                            for name, digest in manifest["outputs"].items())):
                    raise ValueError(f"seed {seed} extraction exited without complete evidence")
            write(RUN / "extraction-status.json", {"state": "three_seeds_extracted_pending_audit_and_scoring",
                                                   "completed_seeds": [0, 1, 2],
                                                   "launch_sha256": sha(RUN / "extraction-launch.json"),
                                                   "updated_at": now()})
        except Exception as exc:
            write(RUN / "extraction-status.json",
                  {"state": "extraction_failed_preserve_partial", "error": repr(exc),
                   "traceback": traceback.format_exc(), "pid": os.getpid(),
                   "updated_at": now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()

