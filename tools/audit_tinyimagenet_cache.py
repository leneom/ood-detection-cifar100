"""Independent byte/order/pixel audit for the frozen Tiny ImageNet cache."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "tinyimagenet-confirmation"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def main():
    declaration, amendment = read(RUN / "declaration.json"), read(RUN / "amendment-1.json")
    download, cache_status = read(RUN / "download-status.json"), read(RUN / "cache-status.json")
    manifest = read(RUN / "cache-manifest.json")
    archive = ROOT / declaration["archive_relative_path"]
    cache = ROOT / "data" / "tinyimagenet-validation.npz"
    check(download["state"] == "verified_archive_downloaded" and
          download["sha256"] == sha(archive), "archive source")
    check(cache_status["state"] == manifest["state"] ==
          "verified_pixels_pending_frozen_checkpoint_extraction" and
          cache_status["manifest_sha256"] == sha(RUN / "cache-manifest.json"),
          "cache status")
    check(manifest["cache_sha256"] == sha(cache), "cache bytes")
    check(manifest["source_sha256"] == manifest["source_snapshot_sha256"] ==
          sha(RUN / "executed_cache_source.py") ==
          sha(ROOT / "tools/prepare_tinyimagenet.py"), "cache producer source")
    check(manifest["protocol_sha256"] == declaration["protocol_sha256"] ==
          sha(ROOT / "docs/TINYIMAGENET_CONFIRMATION.md") and
          manifest["amendment_sha256"] == amendment["amendment_sha256"] ==
          sha(ROOT / "docs/TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md"),
          "protocol and amendment")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    check(set(arrays) == {"images", "filenames", "synsets", "jpeg_sha256", "resized_sha256"},
          "cache array keys")
    images = arrays["images"]
    check(images.shape == (10000, 32, 32, 3) and images.dtype == np.uint8,
          "NHWC uint8 cache")
    filename, synsets = arrays["filenames"].tolist(), arrays["synsets"].tolist()
    check(filename == sorted(filename) and len(set(filename)) == 10000, "ordered unique names")
    check(len(Counter(synsets)) == 200 and set(Counter(synsets).values()) == {50},
          "200 balanced synsets")
    with zipfile.ZipFile(archive) as bundle:
        labels = {}
        for raw in bundle.read("tiny-imagenet-200/val/val_annotations.txt").decode().splitlines():
            name, synset, *_ = raw.split("\t")
            check(name not in labels, "unique annotation")
            labels[name] = synset
        check(set(labels) == set(filename), "archive/cache name set")
        for i, name in enumerate(filename):
            check(synsets[i] == labels[name], "archive/cache synset order")
            data = bundle.read("tiny-imagenet-200/val/images/" + name)
            check(hashlib.sha256(data).hexdigest() == arrays["jpeg_sha256"][i],
                  "original JPEG byte hash")
            with Image.open(io.BytesIO(data)) as picture:
                picture.load()
                check(picture.size == (64, 64), "source size")
                expected = np.asarray(picture.convert("RGB").resize(
                    (32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
            check(np.array_equal(expected, images[i]), "resized pixels")
            check(hashlib.sha256(expected.tobytes()).hexdigest() ==
                  arrays["resized_sha256"][i], "resized pixel hash")
    output = RUN / "cache-audit.json"
    check(not output.exists(), "cache audit already exists")
    shutil.copyfile(__file__, RUN / "executed_cache_audit_source.py")
    record = {"state": "passed", "count": 10000, "synset_count": 200,
              "original_byte_hashes_checked": 10000, "resized_pixel_arrays_checked": 10000,
              "cache_sha256": sha(cache), "archive_sha256": sha(archive),
              "manifest_sha256": sha(RUN / "cache-manifest.json"),
              "source_sha256": sha(__file__),
              "boundary": "Image integrity only; no classifier/OOD score was examined."}
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "pixels_checked": 10000}), flush=True)


if __name__ == "__main__":
    main()

