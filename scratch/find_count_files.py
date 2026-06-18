import os
from pathlib import Path

PROJECT_DIR = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")

def find_files():
    print(f"Searching {PROJECT_DIR} for *counts.json...")
    for root, dirs, files in os.walk(PROJECT_DIR):
        # Skip venv and .git
        if "venv" in root or ".git" in root:
            continue
        for f in files:
            if "counts.json" in f or "data_prep" in f:
                path = Path(root) / f
                print(f"FOUND: {path.relative_to(PROJECT_DIR)} (Size: {path.stat().st_size} bytes)")

if __name__ == "__main__":
    find_files()
