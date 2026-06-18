import json
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REGISTRY_PATH = OUTPUTS_DIR / "experiment_registry.json"

def check():
    if not REGISTRY_PATH.is_file():
        print(f"Registry not found: {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    print(f"Checking {len(registry['experiments'])} experiments in registry...")
    missing_any = False
    
    for exp in registry["experiments"]:
        exp_id = exp["id"]
        print(f"\nExperiment: {exp_id}")
        
        evaluation_path = OUTPUTS_DIR / exp["evaluation"]
        history_path = OUTPUTS_DIR / exp["history"]
        metrics_path = OUTPUTS_DIR / exp["metrics"]
        
        for name, path in [("evaluation", evaluation_path), ("history", history_path), ("metrics", metrics_path)]:
            if path.is_file():
                print(f"  - {name}: FOUND ({path.stat().st_size} bytes)")
            else:
                print(f"  - {name}: MISSING! -> {path}")
                missing_any = True
                
    if not missing_any:
        print("\nAll experiment files in registry are present!")
    else:
        print("\nWarning: Some files are missing!")

if __name__ == "__main__":
    check()
