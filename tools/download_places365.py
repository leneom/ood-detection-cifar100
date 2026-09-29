"""Resumable official Places365 archive download; never reads images or scores."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
DATA = ROOT / "data"
FILES = (
    ("images", "val_256.tar", "places365-val_256.tar", 525158400),
    ("devkit", "filelist_places365-standard.tar", "places365-filelist_places365-standard.tar", 67498496),
)
METADATA = (
    ("places365_val.txt", "annotations"),
    ("categories_places365.txt", "categories"),
)
STATUS = RUN / "download-status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digests(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    count = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            md5.update(block)
            sha256.update(block)
            count += len(block)
    return count, md5.hexdigest(), sha256.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_status(record):
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATUS)


def verify_frozen():
    declaration = read(RUN / "declaration.json")
    amendment = read(RUN / "amendment-1.json")
    if (declaration["state"] != "protocol_frozen_before_places365_archive_access" or
            amendment["state"] != "frozen_before_archive_download" or
            amendment["original_declaration_sha256"] != sha(RUN / "declaration.json") or
            amendment["amendment_sha256"] != sha(ROOT / "docs/PLACES365_AMENDMENT_1.md") or
            declaration["protocol_sha256"] != sha(ROOT / "docs/PLACES365_EXTENSION.md") or
            declaration["source_sha256"] != sha(ROOT / "tools/freeze_places365.py")):
        raise ValueError("Places365 frozen declaration/amendment/source changed")
    for name, expected in declaration["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen predecessor input changed: " + name)
    return declaration, amendment


def download_one(declaration, key, remote_name, local_name, length, status):
    destination = DATA / local_name
    partial = DATA / (local_name + ".partial")
    expected = declaration["expected_md5"][key]
    if destination.exists():
        n, md5, sha256 = digests(destination)
        if n != length or md5 != expected:
            raise ValueError("existing final archive differs: " + local_name)
        status["files"][key] = {"state": "verified_archive", "bytes": n,
                                "md5": md5, "sha256": sha256,
                                "relative_path": "data/" + local_name}
        write_status(status)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > length:
        raise ValueError("partial archive exceeds frozen HEAD size")
    if offset and (status.get("declaration_sha256") != sha(RUN / "declaration.json") or
                   status.get("amendment_sha256") != sha(RUN / "amendment-1.json")):
        raise ValueError("partial archive not bound to current frozen inputs")
    base = declaration["source_urls"][key]
    url = base.replace("http://", "https://", 1)
    request = urllib.request.Request(url, headers={"User-Agent": "OOD-local-research/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    status["state"] = "downloading"
    status["files"][key] = {"state": "downloading", "bytes_downloaded": offset,
                            "expected_bytes": length, "url": url}
    write_status(status)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            code = response.status
            if offset:
                content_range = response.headers.get("Content-Range", "")
                if code != 206 or not re.match(rf"bytes {offset}-\d+/{length}$", content_range):
                    raise ValueError("server did not honor exact Range")
            elif code != 200:
                raise ValueError("unexpected HTTP status: " + str(code))
            mode = "ab" if offset else "xb"
            count = offset
            with partial.open(mode) as stream:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    stream.write(block)
                    count += len(block)
                    if count % (16 << 20) < len(block):
                        status["files"][key]["bytes_downloaded"] = count
                        status["files"][key]["updated_at"] = now()
                        write_status(status)
                stream.flush()
                os.fsync(stream.fileno())
        n, md5, sha256 = digests(partial)
        if n != length or md5 != expected:
            raise ValueError(f"{key} archive digest differs: bytes={n}, md5={md5}")
        partial.replace(destination)
        status["files"][key] = {"state": "verified_archive", "bytes": n,
                                "md5": md5, "sha256": sha256, "url": url,
                                "relative_path": "data/" + local_name}
        write_status(status)
        return destination
    except Exception as exc:
        status["state"] = "download_interrupted_partial_preserved"
        status["files"][key] = {"state": "interrupted", "bytes_downloaded":
                                partial.stat().st_size if partial.exists() else offset,
                                "expected_bytes": length, "url": url, "error": repr(exc)}
        write_status(status)
        raise


def extract_metadata(declaration, devkit, status):
    with tarfile.open(devkit, "r:") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate devkit member paths")
        for m in members:
            parts = Path(m.name.replace("\\", "/")).parts
            if (m.name.startswith(("/", "\\")) or ".." in parts or
                    not (m.isfile() or m.isdir())):
                raise ValueError("unsafe devkit member")
        for filename, key in METADATA:
            matches = [m for m in members if m.isfile() and Path(m.name).name == filename]
            if len(matches) != 1:
                raise ValueError("devkit metadata missing or ambiguous: " + filename)
            contents = archive.extractfile(matches[0]).read()
            md5 = hashlib.md5(contents).hexdigest()
            if md5 != declaration["expected_md5"][key]:
                raise ValueError("metadata MD5 differs: " + filename)
            target = DATA / ("places365-" + filename)
            if target.exists():
                if target.read_bytes() != contents:
                    raise ValueError("existing metadata differs: " + filename)
            else:
                with target.open("xb") as stream:
                    stream.write(contents)
            status["files"][key] = {"state": "verified_metadata", "bytes": len(contents),
                                    "md5": md5,
                                    "sha256": hashlib.sha256(contents).hexdigest(),
                                    "relative_path": "data/" + target.name}
            write_status(status)


def main():
    declaration, amendment = verify_frozen()
    if STATUS.exists():
        status = read(STATUS)
        if (status.get("declaration_sha256") != sha(RUN / "declaration.json") or
                status.get("amendment_sha256") != sha(RUN / "amendment-1.json")):
            raise ValueError("prior download attempt belongs to different declaration")
    else:
        status = {"state": "starting", "started_at": now(), "pid": os.getpid(),
                  "declaration_sha256": sha(RUN / "declaration.json"),
                  "amendment_sha256": sha(RUN / "amendment-1.json"),
                  "source_sha256": sha(Path(__file__)), "files": {}}
    DATA.mkdir(exist_ok=True)
    write_status(status)
    for key, remote_name, local_name, length in FILES:
        target = download_one(declaration, key, remote_name, local_name, length, status)
        print(json.dumps({"file": key, "state": "verified_archive", "bytes": target.stat().st_size}), flush=True)
    extract_metadata(declaration, DATA / FILES[1][2], status)
    status["state"] = "verified_archives_and_metadata_pending_pixel_cache"
    status["completed_at"] = now()
    status["source_sha256"] = sha(Path(__file__))
    write_status(status)
    print(json.dumps({"state": status["state"], "files": list(status["files"])}), flush=True)


if __name__ == "__main__":
    main()
