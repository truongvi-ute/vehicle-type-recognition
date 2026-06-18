import os
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")

def search():
    print("Searching for roadmap keywords in all markdown files...")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip venv and .git
        if "venv" in root or ".git" in root:
            continue
        for f in files:
            if f.endswith(".md"):
                path = Path(root) / f
                content = path.read_text(encoding="utf-8", errors="ignore")
                for keyword in ["hướng phát triển", "future", "roadmap", "phát triển"]:
                    if keyword in content.lower():
                        print(f"FOUND '{keyword}' in: {path.relative_to(PROJECT_ROOT)}")
                        # Print some context lines
                        lines = content.splitlines()
                        for idx, line in enumerate(lines):
                            if keyword in line.lower():
                                print(f"  Line {idx+1}: {line.strip()}")
                        break

if __name__ == "__main__":
    search()
