from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.class_names import CLASS_NAMES


def split_exists(data_dir: Path, split_name: str) -> bool:
    return (data_dir / split_name).is_dir()


def get_image_files(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def evaluate_split(
    model,
    split_dir: Path,
    class_names: List[str]
) -> Dict[str, object]:
    print(f"Đang đánh giá tập: {split_dir.name}...")
    
    # Initialize metric accumulators
    all_preds = []
    all_targets = []
    
    # We map class directory names to index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Load and predict images
    image_paths = get_image_files(split_dir)
    if not image_paths:
        print(f"Cảnh báo: Không tìm thấy ảnh trong {split_dir}")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "samples": 0,
            "classification_report": {},
            "confusion_matrix": []
        }
        
    for img_path in image_paths:
        true_class = img_path.parent.name
        if true_class not in class_to_idx:
            continue
        
        target_idx = class_to_idx[true_class]
        
        try:
            # YOLO prediction
            results = model(img_path, verbose=False)
            pred_idx = int(results[0].probs.top1)
            
            all_preds.append(pred_idx)
            all_targets.append(target_idx)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
            continue

    total_samples = len(all_targets)
    correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    accuracy = correct / max(total_samples, 1)

    metrics = {
        "loss": 0.0, # YOLO-cls validation doesn't easily expose raw cross-entropy without loader
        "accuracy": round(accuracy, 6),
        "samples": total_samples,
        "classification_report": {},
        "confusion_matrix": []
    }

    try:
        from sklearn.metrics import classification_report, confusion_matrix

        metrics["classification_report"] = classification_report(
            all_targets,
            all_preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(
            all_targets,
            all_preds,
            labels=list(range(len(class_names))),
        ).tolist()
    except Exception as exc:
        metrics["classification_report_error"] = str(exc)
        # Fallback simple classification report if sklearn is missing
        report = {}
        for idx, cls in enumerate(class_names):
            tp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t == idx)
            fp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t != idx)
            fn = sum(1 for p, t in zip(all_preds, all_targets) if p != idx and t == idx)
            support = sum(1 for t in all_targets if t == idx)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            report[cls] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1-score": round(f1, 4),
                "support": support
            }
        report["macro avg"] = {
            "precision": round(sum(r["precision"] for r in report.values()) / len(class_names), 4),
            "recall": round(sum(r["recall"] for r in report.values()) / len(class_names), 4),
            "f1-score": round(sum(r["f1-score"] for r in report.values()) / len(class_names), 4),
            "support": total_samples
        }
        metrics["classification_report"] = report

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLO-cls trên các tập splits.")
    parser.add_argument("--model_path", type=str, default="models/yolo_cls_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--output", type=str, default="outputs/evaluation_yolo_cls_best.json")
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.is_file():
        print(f"Lỗi: Không tìm thấy file mô hình tại {args.model_path}")
        sys.exit(1)

    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu {args.data_dir}")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Đang nạp mô hình YOLO từ: {args.model_path}...")
    model = YOLO(args.model_path)

    result = {
        "checkpoint": args.model_path,
        "data_dir": args.data_dir,
        "class_names": CLASS_NAMES,
        "primary_metric_split": "valid_unseen",
        "official_evaluation_split": "test",
    }

    # Evaluate splits
    for split in ["valid_unseen", "test", "valid_traincopy"]:
        split_dir = data_path / split
        if split_dir.is_dir():
            result[split] = evaluate_split(model, split_dir, CLASS_NAMES)
        else:
            result[split] = None

    # Write output JSON
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả đánh giá mô hình YOLO thành công vào: {args.output}")


if __name__ == "__main__":
    main()
