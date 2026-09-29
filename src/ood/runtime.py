from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.set_num_threads(4)


def environment():
    import torchvision
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_checkpoint(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(content, temp)
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_source_hashes():
    here = Path(__file__).parent
    return {name: file_hash(here / name) for name in ("data.py", "models.py", "runtime.py", "train.py")}


@torch.inference_mode()
def predict(model, images, labels=None, batch_size=256, device="cpu"):
    from .data import preprocess
    model.eval()
    logits, correct = [], 0
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        outputs = model(preprocess(batch))
        if not torch.isfinite(outputs).all():
            raise RuntimeError("non-finite model outputs")
        logits.append(outputs.cpu())
        if labels is not None:
            correct += (outputs.argmax(1).cpu() == labels[start:start + batch_size].cpu()).sum().item()
    return torch.cat(logits), correct / len(images) if labels is not None else None
