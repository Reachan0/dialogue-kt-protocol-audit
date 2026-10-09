"""Make driver-style label directories from the released flat label files (offline)."""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for split in ("train", "test"):
    for protocol in ("prefix", "full"):
        directory=ROOT / "runs" / "mathdial" / f"{protocol}_{split}"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "labels" / "mathdial" / f"{protocol}_{split}.jsonl", directory / "labels.jsonl")
print("Use: python code/export_mathdial.py --run-dir runs/mathdial")
