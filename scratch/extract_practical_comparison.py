import json
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REGISTRY_PATH = OUTPUTS_DIR / "experiment_registry.json"

def extract():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    print("=== PRACTICAL EXPERIMENT METRICS COMPARISON ===")
    
    # We will group by (dataset, version) to compare the 3 models on the exact same task
    groups = {}
    for exp in registry["experiments"]:
        dataset = exp["dataset"]
        version = exp["version"]
        key = (dataset, version)
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)
        
    for (dataset, version), exps in sorted(groups.items()):
        print(f"\nTask: Dataset = {dataset} | Version = {version}")
        print("-" * 85)
        print(f"{'Model Architecture':<15} | {'Test Acc':<10} | {'Test Macro F1':<13} | {'Best Epoch':<10} | {'Total Epochs':<12}")
        print("-" * 85)
        
        for exp in sorted(exps, key=lambda x: x["model"]):
            model_id = exp["model"]
            eval_path = OUTPUTS_DIR / exp["evaluation"]
            hist_path = OUTPUTS_DIR / exp["history"]
            
            # Read eval metrics
            acc, macro_f1 = "N/A", "N/A"
            if eval_path.is_file():
                with open(eval_path, "r", encoding="utf-8") as ef:
                    eval_data = ef.load() if hasattr(ef, "load") else json.load(ef)
                    test_data = eval_data.get("test", {})
                    acc = f"{test_data.get('accuracy', 0) * 100:.2f}%"
                    
                    # Try to get macro F1
                    report = test_data.get("classification_report", {})
                    macro_avg = report.get("macro avg", {})
                    macro_f1 = f"{macro_avg.get('f1-score', 0) * 100:.2f}%"
                    
            # Read history
            best_epoch, completed_epochs = "N/A", "N/A"
            if hist_path.is_file():
                with open(hist_path, "r", encoding="utf-8") as hf:
                    hist_data = json.load(hf)
                    completed_epochs = len(hist_data)
                    
                    # best epoch calculation
                    if exp["selection_metric"] == "maximum_valid_unseen_accuracy":
                        best_row = max(hist_data, key=lambda x: x.get("valid_unseen_acc", 0))
                    else:
                        best_row = min(hist_data, key=lambda x: x.get("valid_unseen_loss", 999))
                    best_epoch = best_row.get("epoch", "N/A")
                    
            print(f"{model_id:<18} | {acc:<10} | {macro_f1:<13} | {best_epoch:<10} | {completed_epochs:<12}")
        print("-" * 85)

if __name__ == "__main__":
    extract()
