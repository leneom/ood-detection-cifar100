"""Download the frozen Tiny ImageNet archive with byte and MD5 integrity guards."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "tinyimagenet-confirmation"
DATA = ROOT / "data"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_status(record):
    path = RUN / "download-status.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    declaration = json.loads((RUN / "declaration.json").read_text(encoding="utf-8"))
    if (declaration["state"] != "protocol_frozen_before_archive_access" or
            sha(ROOT / "docs/TINYIMAGENET_CONFIRMATION.md") != declaration["protocol_sha256"] or
            sha(ROOT / "report/validation-isolation/completion.json") !=
            declaration["prior_completion_sha256"]):
        raise ValueError("frozen declaration or predecessor changed")
    destination = ROOT / declaration["archive_relative_path"]
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists() or (RUN / "download-status.json").exists():
        raise FileExistsError("archive/download attempt already exists; inspect before retry")
    length = int(declaration["archive_expected_bytes"])
    expected_md5 = declaration["archive_expected_md5"]
    url = declaration["archive_url"]
    progress = {"state": "downloading", "started_at": now(), "pid": os.getpid(),
                "url": url, "bytes_downloaded": 0, "expected_bytes": length,
                "declaration_sha256": sha(RUN / "declaration.json")}
    write_status(progress)
    DATA.mkdir(exist_ok=True)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    count = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "OOD-local-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            announced = int(response.headers.get("Content-Length", "-1"))
            if status != 200 or announced != length:
                raise RuntimeError(f"HTTP response differs: status={status}, length={announced}")
            with partial.open("xb") as stream:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    stream.write(block)
                    md5.update(block)
                    sha256.update(block)
                    count += len(block)
                    if count % (16 << 20) < len(block):
                        progress["bytes_downloaded"] = count
                        progress["updated_at"] = now()
                        write_status(progress)
                stream.flush()
                os.fsync(stream.fileno())
        if count != length or md5.hexdigest() != expected_md5:
            raise ValueError(f"archive length/digest differs: bytes={count}, md5={md5.hexdigest()}")
        partial.replace(destination)
        write_status({"state": "verified_archive_downloaded", "completed_at": now(),
                      "archive": str(destination), "bytes": count, "md5": md5.hexdigest(),
                      "sha256": sha256.hexdigest(),
                      "declaration_sha256": sha(RUN / "declaration.json"),
                      "boundary": "Archive byte integrity only; images have not been decoded or scored."})
        print(json.dumps({"state": "verified_archive_downloaded", "bytes": count,
                          "sha256": sha256.hexdigest()}), flush=True)
    except Exception as exc:
        write_status({"state": "download_failed_partial_preserved", "failed_at": now(),
                      "bytes_downloaded": count, "expected_bytes": length,
                      "partial": str(partial), "error": repr(exc),
                      "declaration_sha256": sha(RUN / "declaration.json")})
        raise


if __name__ == "__main__":
    main()

