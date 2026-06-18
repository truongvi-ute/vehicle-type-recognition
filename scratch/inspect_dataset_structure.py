import os
from pathlib import Path

DATA_DIR = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\data")

def inspect_structure(folder_path: Path):
    print(f"\nStructure of {folder_path.name}:")
    subdirs = sorted(p for p in folder_path.iterdir() if p.is_dir())
    for subdir in subdirs[:15]:
        files = list(subdir.glob("*"))
        sub_subdirs = [p for p in files if p.is_dir()]
        sub_files = [p for p in files if p.is_file()]
        print(f"  - {subdir.name}: {len(sub_subdirs)} subdirs, {len(sub_files)} files")
        if sub_subdirs:
            for ssd in sub_subdirs[:5]:
                print(f"    * {ssd.name}: {len(list(ssd.glob('*')))} files/dirs")

if __name__ == "__main__":
    inspect_structure(DATA_DIR / "cleaning")
    inspect_structure(DATA_DIR / "raw-cleaning")
