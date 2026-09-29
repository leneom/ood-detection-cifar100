"""Validate and cache the predeclared Tiny ImageNet validation OOD population."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import shutil
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "tinyimagenet-confirmation"
sys.path.insert(0, str(ROOT))
from src.ood.data import load_cifar


def digest(path):
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
    declaration = read(RUN / "declaration.json")
    amendment = read(RUN / "amendment-1.json")
    downloaded = read(RUN / "download-status.json")
    if (declaration["state"] != "protocol_frozen_before_archive_access" or
            digest(ROOT / "docs/TINYIMAGENET_CONFIRMATION.md") != declaration["protocol_sha256"] or
            amendment["state"] != "frozen_before_first_image_decode" or
            amendment["original_declaration_sha256"] != digest(RUN / "declaration.json") or
            amendment["amendment_sha256"] != digest(ROOT / "docs/TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md") or
            amendment["preprocess_source_sha256"] != digest(ROOT / "src/ood/data.py") or
            amendment["corrected_layout"] != "uint8 NHWC [10000,32,32,3]"):
        raise ValueError("frozen protocol/amendment differs")
    archive = ROOT / declaration["archive_relative_path"]
    if (downloaded["state"] != "verified_archive_downloaded" or
            downloaded["declaration_sha256"] != digest(RUN / "declaration.json") or
            downloaded["bytes"] != declaration["archive_expected_bytes"] or
            downloaded["md5"] != declaration["archive_expected_md5"] or
            downloaded["sha256"] != digest(archive)):
        raise ValueError("archive byte integrity has not passed")
    cache = ROOT / "data" / "tinyimagenet-validation.npz"
    status = RUN / "cache-status.json"
    manifest_path = RUN / "cache-manifest.json"
    source_snapshot = RUN / "executed_cache_source.py"
    if cache.exists() or status.exists() or manifest_path.exists() or source_snapshot.exists():
        raise FileExistsError("validation cache attempt already exists")
    shutil.copyfile(__file__, source_snapshot)
    write(status, {"state": "validating_archive_before_scoring", "started_at": now(),
                   "archive_sha256": downloaded["sha256"],
                   "amendment_sha256": amendment["amendment_sha256"]})
    try:
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive member names")
            for name in names:
                path = PurePosixPath(name)
                if (path.is_absolute() or ".." in path.parts or not name.startswith("tiny-imagenet-200/")):
                    raise ValueError("unsafe or unexpected archive member path: " + name)
            prefix = "tiny-imagenet-200/val/"
            annotations = prefix + "val_annotations.txt"
            if annotations not in names:
                raise ValueError("official validation annotation file missing")
            entries = {}
            for line in z.read(annotations).decode("utf-8").splitlines():
                columns = line.split("\t")
                if len(columns) != 6:
                    raise ValueError("invalid validation annotation row")
                filename, synset = columns[:2]
                if filename in entries:
                    raise ValueError("duplicate annotation filename")
                entries[filename] = synset
            image_prefix = prefix + "images/"
            image_paths = sorted(name for name in names if
                                 name.startswith(image_prefix) and
                                 name.endswith(".JPEG") and
                                 len(PurePosixPath(name).parts) == 4)
            filenames = [PurePosixPath(name).name for name in image_paths]
            if (len(image_paths) != 10000 or len(entries) != 10000 or
                    len(set(filenames)) != 10000 or set(filenames) != set(entries)):
                raise ValueError("official 10,000 validation image/annotation census differs")
            labels = [entries[name] for name in filenames]
            counts = Counter(labels)
            if len(counts) != 200 or set(counts.values()) != {50}:
                raise ValueError("expected 200 synsets with 50 validation images each")
            images = np.empty((10000, 32, 32, 3), dtype=np.uint8)
            jpeg_hashes, pixel_hashes = [], []
            for i, member in enumerate(image_paths):
                raw = z.read(member)
                jpeg_hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as picture:
                    picture.load()
                    if picture.size != (64, 64):
                        raise ValueError(f"non-64x64 image: {filenames[i]}")
                    rgb = picture.convert("RGB")
                    resized = rgb.resize((32, 32), resample=Image.Resampling.BILINEAR)
                    pixels = np.asarray(resized, dtype=np.uint8)
                if pixels.shape != (32, 32, 3):
                    raise ValueError("unexpected resized pixel shape")
                images[i] = pixels
                pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                if (i+1) % 1000 == 0:
                    print(json.dumps({"decoded": i+1, "total": 10000}), flush=True)
        train, _ = load_cifar(ROOT / "data", "cifar100", True)
        test, _ = load_cifar(ROOT / "data", "cifar100", False)
        id_hashes = {hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
                     for source in (train, test) for row in source}
        duplicates = sorted(set(pixel_hashes) & id_hashes)
        if duplicates:
            raise ValueError(f"{len(duplicates)} exact resized-pixel duplicates with CIFAR-100 ID")
        if len(set(pixel_hashes)) != 10000:
            raise ValueError("duplicate resized Tiny ImageNet validation pixels")
        temp = cache.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, images=images,
                                filenames=np.asarray(filenames),
                                synsets=np.asarray(labels),
                                jpeg_sha256=np.asarray(jpeg_hashes),
                                resized_sha256=np.asarray(pixel_hashes))
        temp.replace(cache)
        manifest = {"state": "verified_pixels_pending_frozen_checkpoint_extraction",
                    "created_at": now(), "count": 10000, "synset_count": 200,
                    "per_synset_count": 50, "image_shape": [10000, 32, 32, 3],
                    "dtype": "uint8", "archive_sha256": downloaded["sha256"],
                    "cache_sha256": digest(cache), "pillow_version": pillow_version,
                    "source_sha256": digest(__file__),
                    "source_snapshot_sha256": digest(source_snapshot),
                    "protocol_sha256": declaration["protocol_sha256"],
                    "amendment_sha256": amendment["amendment_sha256"],
                    "cifar100_exact_pixel_duplicates": 0,
                    "boundary": "Source/pixel integrity only; no classifier or OOD detector score computed."}
        write(manifest_path, manifest)
        write(status, {"state": manifest["state"], "manifest_sha256": digest(manifest_path),
                       "completed_at": now()})
        print(json.dumps({"state": manifest["state"], "count": 10000,
                          "synsets": 200}), flush=True)
    except Exception as exc:
        write(status, {"state": "cache_failed_preserve_evidence", "failed_at": now(),
                       "error": repr(exc), "archive_sha256": downloaded["sha256"]})
        raise


if __name__ == "__main__":
    main()

