import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ood.datasets import load_ood, OOD_NAMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=OOD_NAMES, default=["cifar10", "mnist"])
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    for name in args.datasets:
        images = load_ood(args.data_root, name, download=True)
        print(f"{name}: {len(images)} verified images, shape={tuple(images.shape)}", flush=True)
