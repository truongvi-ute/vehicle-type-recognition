import os
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
MODELS_DIR = PROJECT_ROOT / "models"

def check():
    if not MODELS_DIR.is_dir():
        print(f"Models directory not found: {MODELS_DIR}")
        return
        
    subdirs = sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())
    print(f"Checking {len(subdirs)} model folders under {MODELS_DIR}...")
    
    for subdir in subdirs:
        print(f"\nFolder: {subdir.name}")
        files = list(subdir.glob("*"))
        if not files:
            print("  (empty)")
            continue
            
        for f in files:
            if f.suffix in (".pth", ".pt", ".yaml", ".onnx"):
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            else:
                print(f"  - {f.name}")

if __name__ == "__main__":
    check()
