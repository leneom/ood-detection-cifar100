"""Build a checked 32x32 RGB cache of every official Places365 validation image."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "places365-extension"
DATA = ROOT / "data"
CACHE = DATA / "places365-validation.npz"
sys.path.insert(0, str(ROOT))
from src.ood.data import load_cifar


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


def write(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def labels_from_annotation(path):
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        require(len(parts) == 2, "invalid Places365 validation annotation row")
        name = PurePosixPath(parts[0].lstrip("/")).name
        require(re.fullmatch(r"Places365_val_\d{8}\.jpg", name) is not None,
                "invalid official validation filename")
        require(name not in labels, "duplicate validation annotation")
        label = int(parts[1])
        require(0 <= label < 365, "invalid Places365 scene id")
        labels[name] = label
    counts = Counter(labels.values())
    require(len(labels) == 36500 and len(counts) == 365 and set(counts.values()) == {100},
            "official 36500/365x100 annotation census differs")
    return labels


def main():
    declaration, amendment, status = (read(RUN / name) for name in
                                      ("declaration.json", "amendment-1.json", "download-status.json"))
    require(declaration["state"] == "protocol_frozen_before_places365_archive_access" and
            amendment["original_declaration_sha256"] == sha(RUN / "declaration.json") and
            amendment["amendment_sha256"] == sha(ROOT / "docs/PLACES365_AMENDMENT_1.md") and
            declaration["protocol_sha256"] == sha(ROOT / "docs/PLACES365_EXTENSION.md"),
            "frozen Places365 protocol/amendment changed")
    require(status["state"] == "verified_archives_and_metadata_pending_pixel_cache" and
            status["declaration_sha256"] == sha(RUN / "declaration.json") and
            status["amendment_sha256"] == sha(RUN / "amendment-1.json"),
            "verified official archive status required")
    for name, expected in declaration["input_sha256"].items():
        require(sha(ROOT / name) == expected, "frozen ID input changed: " + name)
    for key in ("images", "devkit", "annotations", "categories"):
        item = status["files"][key]
        require(item["sha256"] == sha(ROOT / item["relative_path"]) and
                item["md5"] == declaration["expected_md5"][key],
                "archive or metadata integrity changed: " + key)
    require(not any(path.exists() for path in
                    (CACHE, RUN / "cache-status.json", RUN / "cache-manifest.json",
                     RUN / "executed_cache_source.py")),
            "Places365 cache attempt already exists")
    labels = labels_from_annotation(DATA / "places365-places365_val.txt")
    category_lines = (DATA / "places365-categories_places365.txt").read_text(
        encoding="utf-8").splitlines()
    categories = {}
    for line in category_lines:
        parts = line.split()
        require(len(parts) == 2, "invalid category row")
        idx = int(parts[1])
        require(idx not in categories, "duplicate category index")
        categories[idx] = parts[0]
    require(set(categories) == set(range(365)), "category index census differs")
    snapshot = RUN / "executed_cache_source.py"
    shutil.copyfile(__file__, snapshot)
    write(RUN / "cache-status.json", {
        "state": "preparing_verified_places365_pixels", "started_at": now(),
        "declaration_sha256": sha(RUN / "declaration.json"),
        "image_archive_sha256": status["files"]["images"]["sha256"],
        "source_sha256": sha(__file__)})
    try:
        with tarfile.open(DATA / "places365-val_256.tar", "r:") as archive:
            members = archive.getmembers()
            paths = [member.name for member in members]
            require(len(paths) == len(set(paths)), "duplicate archive member path")
            files = {}
            for member in members:
                name = member.name
                path = PurePosixPath(name)
                require(not path.is_absolute() and ".." not in path.parts and
                        "\\" not in name and ":" not in path.parts[0] and
                        (member.isfile() or member.isdir()),
                        "unsafe Places365 archive member")
                if member.isfile():
                    basename = path.name
                    require(re.fullmatch(r"Places365_val_\d{8}\.jpg", basename) is not None,
                            "unexpected nonvalidation archive file")
                    require(basename not in files, "duplicate archive validation basename")
                    files[basename] = member
            require(len(files) == 36500 and set(files) == set(labels),
                    "validation archive/annotation filenames differ")
            names = sorted(files)
            images = np.empty((36500, 32, 32, 3), dtype=np.uint8)
            dimensions = np.empty((36500, 2), dtype=np.int32)
            jpeg_hashes, pixel_hashes = [], []
            for i, name in enumerate(names):
                raw = archive.extractfile(files[name]).read()
                jpeg_hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as picture:
                    picture.load()
                    dimensions[i] = (picture.height, picture.width)
                    pixels = np.asarray(picture.convert("RGB").resize(
                        (32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
                require(pixels.shape == (32, 32, 3), "resized pixel shape differs")
                images[i] = pixels
                pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                if (i + 1) % 2500 == 0:
                    print(json.dumps({"decoded": i + 1, "total": 36500}), flush=True)
        train, _ = load_cifar(DATA, "cifar100", True)
        test, _ = load_cifar(DATA, "cifar100", False)
        id_hashes = {hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
                     for population in (train, test) for row in population}
        duplicate_id = set(pixel_hashes) & id_hashes
        require(not duplicate_id,
                f"{len(duplicate_id)} exact resized-pixel overlaps with CIFAR-100")
        within_ood_duplicates = len(pixel_hashes) - len(set(pixel_hashes))
        temporary = CACHE.with_suffix(".npz.tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, images=images,
                                filenames=np.asarray(names),
                                labels=np.asarray([labels[n] for n in names], dtype=np.int16),
                                source_hw=dimensions,
                                jpeg_sha256=np.asarray(jpeg_hashes),
                                resized_sha256=np.asarray(pixel_hashes))
        temporary.replace(CACHE)
        manifest = {
            "state": "verified_source_and_pixels_pending_independent_audit",
            "created_at": now(), "count": 36500,
            "scene_categories": 365, "per_category": 100,
            "image_shape": [36500, 32, 32, 3], "dtype": "uint8",
            "archive_sha256": status["files"]["images"]["sha256"],
            "devkit_sha256": status["files"]["devkit"]["sha256"],
            "annotation_sha256": status["files"]["annotations"]["sha256"],
            "categories_sha256": status["files"]["categories"]["sha256"],
            "cache_sha256": sha(CACHE), "pillow_version": pillow_version,
            "source_sha256": sha(__file__), "source_snapshot_sha256": sha(snapshot),
            "declaration_sha256": sha(RUN / "declaration.json"),
            "amendment_sha256": sha(RUN / "amendment-1.json"),
            "cifar100_exact_pixel_duplicates": 0,
            "within_ood_resized_pixel_duplicates": within_ood_duplicates,
            "boundary": "Source/pixel integrity only; no model inference or OOD score computed."}
        write(RUN / "cache-manifest.json", manifest)
        write(RUN / "cache-status.json", {
            "state": manifest["state"], "completed_at": now(),
            "manifest_sha256": sha(RUN / "cache-manifest.json")})
        print(json.dumps({"state": manifest["state"], "count": 36500,
                          "within_ood_resized_duplicates": within_ood_duplicates}), flush=True)
    except Exception as exc:
        write(RUN / "cache-status.json", {
            "state": "cache_failed_preserve_evidence", "failed_at": now(),
            "error": repr(exc), "source_sha256": sha(__file__)})
        raise


if __name__ == "__main__":
    main()
