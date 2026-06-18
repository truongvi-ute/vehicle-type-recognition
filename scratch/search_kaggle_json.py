import os
from pathlib import Path

KAGGLE_DIR = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\kaggle_output")

def find_json():
    print(f"Searching {KAGGLE_DIR} for JSON files...")
    for root, dirs, files in os.walk(KAGGLE_DIR):
        for f in files:
            if f.endswith(".json"):
                path = Path(root) / f
                print(f"FOUND: {path.relative_to(KAGGLE_DIR)} (Size: {path.stat().st_size} bytes)")

if __name__ == "__main__":
    find_json()
