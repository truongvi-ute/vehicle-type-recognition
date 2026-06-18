import os
from pathlib import Path

DATA_DIR = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\data")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def count_dir(p: Path):
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)

def scan():
    print(f"Scanning data folders under {DATA_DIR}...")
    for folder in sorted(DATA_DIR.iterdir()):
        if folder.is_dir():
            total_images = count_dir(folder)
            print(f"- {folder.name}: {total_images} images")
            # If it has splits, print split totals
            for split in ["train", "valid_unseen", "valid_traincopy", "test"]:
                split_path = folder / split
                if split_path.exists():
                    print(f"  * {split}: {count_dir(split_path)} images")

if __name__ == "__main__":
    scan()
