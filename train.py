"""Run from this directory without modifying the shared Python environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ood.train import main

if __name__ == "__main__":
    main()
