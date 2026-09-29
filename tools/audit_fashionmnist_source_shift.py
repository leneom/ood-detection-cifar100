"""Descriptive pixel-source audit; never used to fit or select an OOD method."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "fashionmnist-extension"
sys.path.insert(0, str(ROOT))
from src.ood.data import load_cifar


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(images: np.ndarray) -> dict:
    if images.shape != (10000, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("expected full 10k uint8 NHWC population")
    gray_pixels = ((images[..., 0] == images[..., 1]) &
                   (images[..., 1] == images[..., 2]))
    gray_images = np.all(gray_pixels, axis=(1, 2))
    spread = images.max(axis=3).astype(np.float32) - images.min(axis=3).astype(np.float32)
    brightness = images.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
    spatial_contrast = images.astype(np.float32).mean(axis=3).std(axis=(1, 2)) / 255.0
    return {
        "count": int(len(images)),
        "exact_gray_image_count": int(gray_images.sum()),
        "exact_gray_pixel_fraction": float(gray_pixels.mean()),
        "mean_rgb_channel_range_per_pixel_uint8": float(spread.mean()),
        "mean_per_image_brightness_0_1": float(brightness.mean()),
        "median_per_image_brightness_0_1": float(np.median(brightness)),
        "mean_per_image_spatial_contrast_0_1": float(spatial_contrast.mean()),
        "median_per_image_spatial_contrast_0_1": float(np.median(spatial_contrast)),
    }


def main() -> None:
    output = RUN / "source-shift-descriptive.json"
    if output.exists():
        raise FileExistsError("descriptive audit already exists")
    cache = ROOT / "data" / "fashionmnist-test-32x32.npz"
    audit = json.loads((RUN / "cache-audit.json").read_text(encoding="utf-8"))
    if audit["state"] != "passed" or audit["count"] != 10000 or audit["cache_sha256"] != sha(cache):
        raise ValueError("independently audited Fashion cache required")
    with np.load(cache, allow_pickle=False) as arrays:
        fashion = arrays["images"]
        if not np.array_equal(arrays["source_rows"], np.arange(10000)):
            raise ValueError("Fashion source order differs")
    id_images = load_cifar(ROOT / "data", "cifar100", False)[0].numpy()
    id_stats, fashion_stats = stats(id_images), stats(fashion)
    if fashion_stats["exact_gray_image_count"] != 10000 or fashion_stats["exact_gray_pixel_fraction"] != 1.0:
        raise AssertionError("Fashion RGB-repeat transform differs")
    record = {
        "state": "descriptive_pixel_audit_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(Path(__file__)),
        "fashion_cache_sha256": sha(cache),
        "fashion_cache_audit_sha256": sha(RUN / "cache-audit.json"),
        "id_source": "CIFAR-100 official test, all 10000 raw uint8 NHWC images",
        "ood_source": "Fashion-MNIST official test, all 10000 frozen resized RGB-repeat images",
        "id_test": id_stats,
        "fashion_test": fashion_stats,
        "boundary": "Post-access descriptive pixel comparison only; no model scores, selection, fitting, threshold changes or semantic causal claim.",
    }
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": record["state"], "id_exact_gray_images": id_stats["exact_gray_image_count"],
                      "fashion_exact_gray_images": fashion_stats["exact_gray_image_count"]}))


if __name__ == "__main__":
    main()
