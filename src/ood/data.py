from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def stratified_split(targets, val_per_class=50, seed=2026):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        if not 0 < val_per_class < len(indices):
            raise ValueError("each class must have both training and validation samples")
        indices = rng.permutation(indices)
        val.extend(indices[:val_per_class])
        train.extend(indices[val_per_class:])
    return np.sort(np.asarray(train)), np.sort(np.asarray(val))


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_cifar(root, name, train, download=False, archive_source=None):
    from torchvision.datasets import CIFAR10, CIFAR100
    dataset_type = {"cifar10": CIFAR10, "cifar100": CIFAR100}[name]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Reuse the existing public archive rather than protected extracted files.
    if archive_source and not (root / dataset_type.filename).exists():
        source = Path(archive_source) / dataset_type.filename
        if source.is_file():
            shutil.copyfile(source, root / dataset_type.filename)
    dataset = dataset_type(str(root), train=train, download=download)
    return torch.from_numpy(dataset.data.copy()), torch.tensor(dataset.targets, dtype=torch.long)


def preprocess(images, augment=False):
    # Dataset storage: uint8 NHWC; all datasets use the same ID normalization.
    inputs = images.permute(0, 3, 1, 2).float().div_(255.)
    if augment:
        n = len(inputs)
        padded = F.pad(inputs, (4, 4, 4, 4)).permute(0, 2, 3, 1)
        y = torch.randint(0, 9, (n, 1, 1), device=inputs.device)
        x = torch.randint(0, 9, (n, 1, 1), device=inputs.device)
        grid = torch.arange(32, device=inputs.device)
        inputs = padded[
            torch.arange(n, device=inputs.device)[:, None, None],
            y + grid[None, :, None], x + grid[None, None, :],
        ].permute(0, 3, 1, 2)
        flip = torch.rand(n, 1, 1, 1, device=inputs.device) < 0.5
        inputs = torch.where(flip, inputs.flip(3), inputs)
    mean = inputs.new_tensor(MEAN)[None, :, None, None]
    std = inputs.new_tensor(STD)[None, :, None, None]
    return ((inputs - mean) / std).contiguous(memory_format=torch.channels_last)


def synthetic(seed=0, train_per_class=48, test_per_class=16, classes=4):
    rng = np.random.default_rng(seed)
    prototypes = rng.integers(20, 70, size=(classes, 32, 32, 3), dtype=np.uint8)
    for label in range(classes):
        prototypes[label, :, :, label % 3] += 130
        prototypes[label, label * 5:label * 5 + 6, :, :] = 240
    def make(count):
        labels = np.repeat(np.arange(classes), count)
        noise = rng.normal(0, 8, (len(labels), 32, 32, 3))
        images = np.clip(prototypes[labels].astype(float) + noise, 0, 255).astype(np.uint8)
        return torch.from_numpy(images), torch.tensor(labels)
    return make(train_per_class), make(test_per_class)
