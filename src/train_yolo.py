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
    model.val(data=adapter_dir, split="test", imgsz=imgsz, batch=batch)

    # Tự động xuất báo cáo đánh giá định dạng JSON cho Dashboard
    try:
        print("\n--- Đang tự động tạo báo cáo đánh giá JSON cho Dashboard ---")
        best_weights_path = Path(project) / name / "weights" / "best.pt"
        if not best_weights_path.is_file():
            # Thử đường dẫn thay thế nếu Ultralytics tự động sinh thêm thư mục runs
            best_weights_path = Path("runs/classify") / project / name / "weights" / "best.pt"
            if not best_weights_path.is_file():
                # Quét mọi file best.pt trong thư mục runs
                candidate_weights = list(Path().glob("**/weights/best.pt"))
                if candidate_weights:
                    best_weights_path = candidate_weights[0]

        if best_weights_path.is_file():
            print(f"Tìm thấy trọng số tốt nhất tại: {best_weights_path}")
            from src.evaluate_yolo import evaluate_split
            from backend.utils.class_names import CLASS_NAMES
            import json

            best_model = YOLO(str(best_weights_path))
            data_path = Path(data_dir)

            eval_result = {
                "checkpoint": str(best_weights_path),
                "data_dir": data_dir,
                "class_names": CLASS_NAMES,
                "primary_metric_split": "valid_unseen",
                "official_evaluation_split": "test",
            }

            for split_name in ["valid_unseen", "test", "valid_traincopy"]:
                split_dir = data_path / split_name
                if split_dir.is_dir():
                    eval_result[split_name] = evaluate_split(best_model, split_dir, CLASS_NAMES)
                else:
                    eval_result[split_name] = None

            out_file = Path("outputs") / f"evaluation_{name}_best.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=2)
            print(f"Đã tự động lưu file báo cáo đánh giá thành công tại: {out_file}")
        else:
            print("Cảnh báo: Không tìm thấy file trọng số best.pt để chạy đánh giá tự động.")
    except Exception as exc:
        print(f"Không thể tự động xuất báo cáo đánh giá JSON: {exc}")

    if eval_traincopy:
        aux_path = Path(data_dir) / AUX_VALID_SPLIT
        if not aux_path.is_dir():
            print(f"Auxiliary split not available: {aux_path}")
            return
        aux_adapter = Path(adapter_dir).with_name(Path(adapter_dir).name + "_traincopy")
        prepare_yolo_adapter(data_dir, str(aux_adapter))
        recreate_link(aux_adapter / "val", aux_path)
        model.val(data=str(aux_adapter), split="val", imgsz=imgsz, batch=batch)


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
