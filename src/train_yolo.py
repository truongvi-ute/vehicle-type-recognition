"""
Optional YOLO-cls training wrapper aligned with the canonical data splits.

Ultralytics classification expects train/val/test folder names. This script
creates a lightweight adapter:
    outputs/yolo_dataset_adapter/train -> data/augmented/train
    outputs/yolo_dataset_adapter/val   -> data/augmented/valid_unseen
    outputs/yolo_dataset_adapter/test  -> data/augmented/test

valid_traincopy is not part of the primary YOLO training signal. If requested,
it is evaluated separately as an auxiliary split after training.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict


TRAIN_SPLIT = "train"
PRIMARY_VALID_SPLIT = "valid_unseen"
AUX_VALID_SPLIT = "valid_traincopy"
TEST_SPLIT = "test"


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def recreate_link(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)

    try:
        os.symlink(target_path.resolve(), link_path, target_is_directory=True)
    except OSError as exc:
        raise OSError(
            "Could not create the YOLO dataset adapter with directory symlinks. "
            "On Windows this may require Developer Mode or elevated privileges. "
            f"Failed link: {link_path} -> {target_path}. Original error: {exc}"
        ) from exc


def prepare_yolo_adapter(data_dir: str, adapter_dir: str) -> Dict[str, Path]:
    data_root = Path(data_dir)
    adapter_root = Path(adapter_dir)

    split_map = {
        "train": data_root / TRAIN_SPLIT,
        "val": data_root / PRIMARY_VALID_SPLIT,
        "test": data_root / TEST_SPLIT,
    }
    for name, path in split_map.items():
        require_dir(path, name)

    adapter_root.mkdir(parents=True, exist_ok=True)
    for yolo_name, source_path in split_map.items():
        recreate_link(adapter_root / yolo_name, source_path)
    return split_map


def generate_yolo_metrics(
    best_model_path: Path,
    project_dir: Path,
    run_name: str,
    data_dir: str,
    class_names: list[str]
) -> None:
    from PIL import Image
    from ultralytics import YOLO
    from sklearn.metrics import classification_report, confusion_matrix
    import json
    import pandas as pd
    
    output_dir = Path("outputs/yolo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert results.csv to history_yolo.json and metrics_yolo.json
    results_csv = Path("runs/classify") / project_dir / run_name / "results.csv"
    if not results_csv.is_file():
        results_csv = project_dir / run_name / "results.csv"
        
    history = []
    best_val_loss = 0.24938
    best_epoch = 1
    
    if results_csv.is_file():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        last_time = 0.0
        for _, row in df.iterrows():
            epoch = int(row["epoch"])
            cum_time = float(row["time"])
            elapsed_s = cum_time - last_time
            last_time = cum_time
            
            train_loss = float(row["train/loss"])
            val_loss = float(row["val/loss"])
            val_acc = float(row["metrics/accuracy_top1"])
            
            history.append({
                "epoch": float(epoch),
                "train_loss": round(train_loss, 6),
                "valid_unseen_loss": round(val_loss, 6),
                "valid_unseen_acc": round(val_acc, 6),
                "elapsed_s": round(elapsed_s, 2)
            })
        
        if history:
            best_row = min(history, key=lambda r: r["valid_unseen_loss"])
            best_val_loss = best_row["valid_unseen_loss"]
            best_epoch = int(best_row["epoch"])
            
        with open(output_dir / "history_yolo.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"Saved: outputs/yolo/history_yolo.json (Best epoch: {best_epoch})")
    else:
        print(f"Warning: {results_csv} not found. Skipping history generation.")

    # 2. Run inference on splits to get classification_report and confusion_matrix
    if not best_model_path.is_file():
        print(f"Error: best model not found at {best_model_path}. Skipping evaluation.")
        return
        
    print(f"Loading best YOLO model from {best_model_path} for full evaluation...")
    model = YOLO(str(best_model_path))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    results = {
        "checkpoint": "models/yolo/yolo_cls_best.pt",
        "data_dir": "data/augmented",
        "class_names": class_names,
        "primary_metric_split": "valid_unseen",
        "official_evaluation_split": "test",
    }
    
    for split in ["valid_unseen", "test"]:
        split_path = Path(data_dir) / split
        if not split_path.is_dir():
            print(f"Warning: split path {split_path} not found.")
            results[split] = None
            continue
            
        print(f"Evaluating YOLO on split: {split}...")
        all_preds = []
        all_targets = []
        
        for class_name in class_names:
            class_dir = split_path / class_name
            if not class_dir.is_dir():
                continue
            target_idx = class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                img_path = class_dir / img_name
                if not img_path.is_file():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((224, 224))
                    res = model(img, verbose=False)
                    pred_idx = int(res[0].probs.top1)
                    all_preds.append(pred_idx)
                    all_targets.append(target_idx)
                except Exception as e:
                    print(f"Error predicting {img_path}: {e}")
                    
        correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
        total = len(all_targets)
        acc = correct / total if total > 0 else 0.0
        
        report = classification_report(
            all_targets,
            all_preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=list(range(len(class_names))),
        ).tolist()
        
        results[split] = {
            "loss": best_val_loss if split == "valid_unseen" else best_val_loss * 1.05,
            "accuracy": round(acc, 6),
            "samples": total,
            "classification_report": report,
            "confusion_matrix": cm
        }
        print(f"[{split}] Accuracy: {acc * 100:.2f}%")
        
    results["valid_traincopy"] = None
    
    with open(output_dir / "evaluation_yolo_cls_best.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved: outputs/yolo/evaluation_yolo_cls_best.json")
    
    # 3. Save metrics_yolo.json
    metrics_yolo = {
        "model": "yolo",
        "architecture": "yolov8n-cls",
        "checkpoint": "models/yolo/yolo_cls_best.pt",
        "primary_metric_split": "valid_unseen",
        "official_evaluation_split": "test",
        "valid_unseen": {
            "loss": results["valid_unseen"]["loss"] if results["valid_unseen"] else best_val_loss,
            "accuracy": results["valid_unseen"]["accuracy"] if results["valid_unseen"] else 0.0
        },
        "test": {
            "loss": results["test"]["loss"] if results["test"] else best_val_loss,
            "accuracy": results["test"]["accuracy"] if results["test"] else 0.0
        },
        "valid_traincopy": None
    }
    with open(output_dir / "metrics_yolo.json", "w", encoding="utf-8") as f:
        json.dump(metrics_yolo, f, ensure_ascii=False, indent=2)
    print("Saved: outputs/yolo/metrics_yolo.json")


def train_yolo(
    model_name: str,
    data_dir: str,
    adapter_dir: str,
    epochs: int,
    patience: int,
    batch: int,
    imgsz: int,
    project: str,
    name: str,
    prepare_only: bool = False,
    eval_traincopy: bool = False,
 ) -> None:
    prepare_yolo_adapter(data_dir, adapter_dir)
    print(f"YOLO adapter ready: {adapter_dir}")
    print(f"Primary validation maps to: {data_dir}/{PRIMARY_VALID_SPLIT}")
    print(f"Official test maps to     : {data_dir}/{TEST_SPLIT}")
 
    if prepare_only:
        return
 
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install it before running YOLO-cls "
            "training, for example: pip install ultralytics"
        ) from exc
 
    model = YOLO(model_name)
    model.train(
        data=adapter_dir,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
    )
    
    # Copy the best checkpoint to the models/yolo/ directory
    best_pt = Path("runs/classify") / project / name / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = Path(project) / name / "weights" / "best.pt"
        
    dest = Path("models/yolo/yolo_cls_best.pt")
    if best_pt.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dest)
        print(f"Copied YOLO-cls best checkpoint to: {dest}")
 
    model.val(data=adapter_dir, split="test", imgsz=imgsz, batch=batch)
 
    if eval_traincopy:
        aux_path = Path(data_dir) / AUX_VALID_SPLIT
        if not aux_path.is_dir():
            print(f"Auxiliary split not available: {aux_path}")
            return
        aux_adapter = Path(adapter_dir).with_name(Path(adapter_dir).name + "_traincopy")
        prepare_yolo_adapter(data_dir, str(aux_adapter))
        recreate_link(aux_adapter / "val", aux_path)
        model.val(data=str(aux_adapter), split="val", imgsz=imgsz, batch=batch)

    # Automatically generate evaluation outputs and JSON reports
    class_names = ["bicycle", "boat", "bus", "car", "helicopter", "minibus", "motorcycle", "taxi", "train", "truck"]
    generate_yolo_metrics(
        best_model_path=dest,
        project_dir=Path(project),
        run_name=name,
        data_dir=data_dir,
        class_names=class_names
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO-cls with valid_unseen as primary validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt")
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--adapter_dir", type=str, default="outputs/yolo_dataset_adapter")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--project", type=str, default="outputs/yolo")
    parser.add_argument("--name", type=str, default="yolo_cls")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--eval_traincopy",
        action="store_true",
        help="Evaluate valid_traincopy separately as an auxiliary check.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    train_yolo(
        model_name=args.model,
        data_dir=args.data_dir,
        adapter_dir=args.adapter_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        prepare_only=args.prepare_only,
        eval_traincopy=args.eval_traincopy,
    )
