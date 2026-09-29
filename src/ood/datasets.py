"""Natural OOD datasets and explicitly labeled synthetic diagnostics."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_cifar


OOD_NAMES = ("cifar10", "mnist", "svhn", "textures", "noise", "constant")


def load_ood(root, name, count=10000, download=False, seed=110):
    if name == "cifar10":
        return load_cifar(root, name, False, download)[0]
    if name == "mnist":
        from pathlib import Path
        from torchvision.datasets import MNIST
        from torchvision.datasets.mnist import read_image_file
        from torchvision.datasets.utils import check_integrity, download_and_extract_archive
        raw = Path(root) / "MNIST" / "raw"
        # Fetch and verify test-only resources; OOD training digits are unnecessary.
        for filename, md5 in MNIST.resources:
            if filename.startswith("t10k"):
                archive = raw / filename
                extracted = raw / filename.removesuffix(".gz")
                if not check_integrity(str(archive), md5) or not extracted.is_file():
                    if not download:
                        raise FileNotFoundError("verified MNIST test files missing; use --download")
                    download_and_extract_archive("https://storage.googleapis.com/cvdf-datasets/mnist/" + filename, str(raw), md5=md5)
        images = read_image_file(str(raw / "t10k-images-idx3-ubyte"))
        gray = F.interpolate(images[:, None].float(), size=(32, 32), mode="bilinear", align_corners=False)
        return gray.round().clamp(0, 255).to(torch.uint8).repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
    if name == "svhn":
        from pathlib import Path
        from .cache import load_svhn_cache
        if (Path(root) / "svhn-test.npz").exists() or (Path(root) / "svhn-test.manifest.json").exists():
            images, _, _ = load_svhn_cache(root)
            return torch.from_numpy(images)
        try:
            import scipy
        except ImportError as error:
            raise FileNotFoundError("SVHN cache missing; run tools/prepare_svhn.py using the CPU Python with SciPy") from error
        from torchvision.datasets import SVHN
        dataset = SVHN(root, split="test", download=download)
        # Use the full official test split; its different prevalence is reported.
        return torch.from_numpy(dataset.data.transpose(0, 2, 3, 1).copy())
    if name == "textures":
        from .textures_cache import load_textures_cache
        images, _, _ = load_textures_cache(root)
        return torch.from_numpy(images)
    shape = (count, 32, 32, 3)
    rng = np.random.default_rng(seed)
    if name == "noise":
        return torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8))
    if name == "constant":
        return torch.full(shape, 128, dtype=torch.uint8)
    raise ValueError(f"unknown OOD dataset: {name}")


def description(name):
    return {
        "cifar10": "near-OOD natural images; official test split",
        "mnist": "far-OOD handwritten digits; official test; bilinear 28->32; repeated RGB",
        "svhn": "far-OOD natural digits; full official test split",
        "textures": "far-OOD natural textures; DTD r1.0.1 official test, partition 1; short-edge-32 bilinear then center crop",
        "noise": "uniform pixel noise; synthetic diagnostic only",
        "constant": "constant gray images; synthetic diagnostic only",
    }[name]
