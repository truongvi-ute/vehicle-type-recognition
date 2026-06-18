import os
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
SRC_DIR = PROJECT_ROOT / "src"

def search():
    print("Searching for 'raw_original' in code...")
    for root, dirs, files in os.walk(SRC_DIR):
        for f in files:
            if f.endswith(".py"):
                path = Path(root) / f
                content = path.read_text(encoding="utf-8", errors="ignore")
                if "raw_original" in content:
                    print(f"FOUND in: {path.relative_to(PROJECT_ROOT)}")
                    # Print lines containing raw_original
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if "raw_original" in line:
                            print(f"  Line {idx+1}: {line.strip()}")

if __name__ == "__main__":
    search()
