"""Independent all-row source/pixel audit for frozen Places365 validation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
DATA = ROOT / "data"
CACHE = DATA / "places365-validation.npz"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    declaration = read(RUN / "declaration.json")
    amendment = read(RUN / "amendment-1.json")
    downloaded = read(RUN / "download-status.json")
    status = read(RUN / "cache-status.json")
    manifest = read(RUN / "cache-manifest.json")
    check(declaration["protocol_sha256"] == sha(ROOT / "docs/PLACES365_EXTENSION.md") and
          amendment["original_declaration_sha256"] == sha(RUN / "declaration.json") and
          amendment["amendment_sha256"] == sha(ROOT / "docs/PLACES365_AMENDMENT_1.md"),
          "frozen protocol/amendment")
    check(downloaded["state"] == "verified_archives_and_metadata_pending_pixel_cache" and
          downloaded["declaration_sha256"] == sha(RUN / "declaration.json"),
          "verified download")
    for key in ("images", "devkit", "annotations", "categories"):
        item = downloaded["files"][key]
        check(item["sha256"] == sha(ROOT / item["relative_path"]) and
              item["md5"] == declaration["expected_md5"][key],
              "archive/metadata hash: " + key)
    check(status["state"] == manifest["state"] ==
          "verified_source_and_pixels_pending_independent_audit" and
          status["manifest_sha256"] == sha(RUN / "cache-manifest.json") and
          manifest["cache_sha256"] == sha(CACHE), "cache/manifest status")
    check(manifest["source_sha256"] == sha(ROOT / "tools/prepare_places365.py") ==
          sha(RUN / "executed_cache_source.py"), "executed cache source")
    with np.load(CACHE, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    check(set(arrays) == {"images", "filenames", "labels", "source_hw",
                          "jpeg_sha256", "resized_sha256"}, "cache keys")
    images = arrays["images"]
    check(images.shape == (36500, 32, 32, 3) and images.dtype == np.uint8,
          "cache pixel shape/dtype")
    names = arrays["filenames"].tolist()
    check(len(names) == 36500 and names == sorted(names) and len(set(names)) == 36500,
          "cache name census/order")
    official = {}
    for line in (DATA / "places365-places365_val.txt").read_text(encoding="utf-8").splitlines():
        name, label = line.split()
        check(name not in official, "duplicate official annotation")
        official[name] = int(label)
    check(set(names) == set(official) and
          len(Counter(official.values())) == 365 and
          set(Counter(official.values()).values()) == {100},
          "official 36500/365x100 annotation census")
    check(all(int(arrays["labels"][i]) == official[name]
              for i, name in enumerate(names)), "cache labels")
    category_ids = [int(line.split()[1]) for line in
                    (DATA / "places365-categories_places365.txt").read_text(
                        encoding="utf-8").splitlines()]
    check(sorted(category_ids) == list(range(365)), "category metadata census")
    with tarfile.open(DATA / "places365-val_256.tar", "r:") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        by_name = {Path(m.name).name: m for m in members}
        check(len(members) == len(by_name) == 36500 and set(by_name) == set(names),
              "tar/annotation file census")
        for i, name in enumerate(names):
            jpeg = archive.extractfile(by_name[name]).read()
            check(hashlib.sha256(jpeg).hexdigest() == arrays["jpeg_sha256"][i],
                  "JPEG byte hash: " + name)
            with Image.open(io.BytesIO(jpeg)) as picture:
                picture.load()
                check(tuple(arrays["source_hw"][i]) == (picture.height, picture.width),
                      "original dimensions: " + name)
                independent = np.asarray(picture.convert("RGB").resize(
                    (32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
            check(np.array_equal(independent, images[i]), "resized RGB pixels: " + name)
            check(hashlib.sha256(independent.tobytes()).hexdigest() ==
                  arrays["resized_sha256"][i], "resized pixel hash: " + name)
            if (i + 1) % 5000 == 0:
                print(json.dumps({"audited": i + 1, "total": 36500}), flush=True)
    duplicate_count = 36500 - len(set(arrays["resized_sha256"].tolist()))
    check(manifest["within_ood_resized_pixel_duplicates"] == duplicate_count,
          "within-OOD duplicate accounting")
    output = RUN / "cache-audit.json"
    check(not output.exists(), "cache audit already exists")
    snapshot = RUN / "executed_cache_audit_source.py"
    check(not snapshot.exists(), "cache auditor snapshot already exists")
    shutil.copyfile(__file__, snapshot)
    result = {
        "state": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
        "images_checked": 36500, "labels_checked": 36500,
        "original_jpeg_byte_hashes_checked": 36500,
        "resized_pixel_arrays_checked": 36500,
        "within_ood_resized_duplicates": duplicate_count,
        "cache_sha256": sha(CACHE),
        "archive_sha256": sha(DATA / "places365-val_256.tar"),
        "manifest_sha256": sha(RUN / "cache-manifest.json"),
        "source_sha256": sha(__file__),
        "boundary": "Image and metadata integrity only; no classifier or OOD score examined."}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "pixels_checked": 36500}), flush=True)


if __name__ == "__main__":
    main()
