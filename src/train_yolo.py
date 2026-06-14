"""Train YOLOv8n-cls on a locally prepared Vehicle-10 dataset.

The local pipeline already produces square 224x224 images and performs the
offline V1/V2 augmentation. A custom Ultralytics classification dataset keeps
the full image instead of applying the default RandomResizedCrop and disables
additional online augmentation so V1 and V2 remain directly comparable.

The test split is never evaluated by this training script. Use
``src/evaluate_yolo.py`` after the best validation checkpoint is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = [
    "bicycle",
    "boat",
    "bus",
    "car",
    "helicopter",
    "minibus",
    "motorcycle",
    "taxi",
    "train",
    "truck",
]
SPLIT_MAP = {
    "train": "train",
    "val": "valid_unseen",
    "test": "test",
}


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def class_directories(split_dir: Path) -> list[str]:
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def count_images(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(data_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "expected_classes": CLASS_NAMES,
        "splits": {},
    }

    for split_name in ["train", "valid_unseen", "test"]:
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing required split: {split_dir}")

        actual_classes = class_directories(split_dir)
        if actual_classes != CLASS_NAMES:
            missing = sorted(set(CLASS_NAMES) - set(actual_classes))
            unexpected = sorted(set(actual_classes) - set(CLASS_NAMES))
            raise ValueError(
                f"Invalid classes in {split_dir}. Missing={missing}, unexpected={unexpected}"
            )

        per_class = {name: count_images(split_dir / name) for name in CLASS_NAMES}
        empty_classes = [name for name, count in per_class.items() if count == 0]
        if empty_classes:
            raise ValueError(f"Empty classes in {split_name}: {empty_classes}")
        report["splits"][split_name] = {
            "total": sum(per_class.values()),
            "per_class": per_class,
        }

    auxiliary_dir = data_dir / "valid_traincopy"
    if auxiliary_dir.is_dir():
        auxiliary_classes = class_directories(auxiliary_dir)
        if auxiliary_classes != CLASS_NAMES:
            raise ValueError(
                f"Invalid classes in {auxiliary_dir}: {auxiliary_classes}"
            )
        per_class = {name: count_images(auxiliary_dir / name) for name in CLASS_NAMES}
        report["splits"]["valid_traincopy"] = {
            "total": sum(per_class.values()),
            "per_class": per_class,
            "role": "auxiliary_only_overlaps_train",
        }

    return report


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def recreate_link(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        remove_path(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path.resolve(), link_path, target_is_directory=True)


def prepare_yolo_adapter(data_dir: Path, adapter_dir: Path) -> None:
    if adapter_dir.exists() or adapter_dir.is_symlink():
        remove_path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for yolo_name, source_name in SPLIT_MAP.items():
        recreate_link(adapter_dir / yolo_name, data_dir / source_name)


def build_full_image_trainer():
    import torch
    import torchvision.transforms as transforms
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer

    class FullImageClassificationDataset(ClassificationDataset):
        """Use the complete local 224x224 image without crop augmentation."""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            super().__init__(root=root, args=args, augment=augment, prefix=prefix)
            self.torch_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        (args.imgsz, args.imgsz),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor([0.0, 0.0, 0.0]),
                        std=torch.tensor([1.0, 1.0, 1.0]),
                    ),
                ]
            )

    class FullImageClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            return FullImageClassificationDataset(
                root=img_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
            )

    return FullImageClassificationTrainer


def normalize_epochs(values) -> list[int]:
    epochs = [int(value) for value in values]
    if epochs and min(epochs) == 0:
        return [value + 1 for value in epochs]
    return epochs


def convert_results(results_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = frame.columns.str.strip()
    required = ["epoch", "train/loss", "val/loss", "metrics/accuracy_top1"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing results.csv columns: {missing}; found={list(frame.columns)}")

    epochs = normalize_epochs(frame["epoch"].tolist())
    cumulative_time = (
        pd.to_numeric(frame["time"], errors="coerce")
        if "time" in frame.columns
        else pd.Series([float("nan")] * len(frame))
    )
    elapsed = cumulative_time.diff().fillna(cumulative_time)

    history: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        top5 = row.get("metrics/accuracy_top5")
        history.append(
            {
                "epoch": epochs[position],
                "train_loss": round(float(row["train/loss"]), 6),
                "valid_unseen_loss": round(float(row["val/loss"]), 6),
                "valid_unseen_acc": round(float(row["metrics/accuracy_top1"]), 6),
                "valid_unseen_top5_acc": (
                    None if top5 is None or pd.isna(top5) else round(float(top5), 6)
                ),
                "elapsed_s": (
                    None if pd.isna(elapsed.iloc[position]) else round(float(elapsed.iloc[position]), 2)
                ),
            }
        )

    if not history:
        raise ValueError(f"No epoch records found in {results_csv}")

    best_accuracy = max(history, key=lambda row: row["valid_unseen_acc"])
    minimum_loss = min(history, key=lambda row: row["valid_unseen_loss"])
    summary = {
        "completed_epochs": len(history),
        "checkpoint_selection_metric": "valid_unseen_top1_accuracy",
        "best_checkpoint_epoch": best_accuracy["epoch"],
        "best_valid_unseen_accuracy": best_accuracy["valid_unseen_acc"],
        "valid_loss_at_best_checkpoint": best_accuracy["valid_unseen_loss"],
        "minimum_valid_unseen_loss_epoch": minimum_loss["epoch"],
        "minimum_valid_unseen_loss": minimum_loss["valid_unseen_loss"],
    }
    return history, summary


def copy_training_artifacts(run_dir: Path, output_dir: Path) -> None:
    for name in [
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "args.yaml",
    ]:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n-cls with valid_unseen as the only selection split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="yolo_aug_v1_cleaning")
    parser.add_argument("--dataset_variant", choices=["v1", "v2"], required=True)
    parser.add_argument(
        "--dataset_id",
        choices=["raw_original", "raw_cleaning"],
        default="raw_cleaning",
        help="Dataset source identity written to metrics and experiment manifest.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    adapter_dir = PROJECT_ROOT / "outputs" / "yolo_dataset_adapter"
    output_dir = PROJECT_ROOT / "outputs" / "yolo"
    run_root = PROJECT_ROOT / "outputs" / "yolo_runs"
    model_dir = PROJECT_ROOT / "models" / "yolo"

    if args.reset:
        for path in [adapter_dir, output_dir, run_root, model_dir]:
            if path.exists() or path.is_symlink():
                remove_path(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_report = validate_dataset(data_dir)
    save_json(output_dir / "dataset_audit.json", dataset_report)
    prepare_yolo_adapter(data_dir, adapter_dir)

    print("=" * 72)
    print("YOLOv8n-cls Vehicle-10 training")
    print("=" * 72)
    print(f"Dataset source        : {args.dataset_id}")
    print(f"Augmentation version  : {args.dataset_variant.upper()}")
    print(f"Dataset root          : {data_dir}")
    print(f"Training split        : {data_dir / 'train'}")
    print(f"Primary validation    : {data_dir / 'valid_unseen'}")
    print(f"Official test         : reserved; not used by train_yolo.py")
    print(f"Input policy          : full image resize to {args.imgsz}x{args.imgsz}, no crop")
    print("Online augmentation   : disabled (offline V1/V2 only)")
    print(f"Seed / deterministic  : {args.seed} / True")
    print("=" * 72)

    if args.prepare_only:
        print(f"YOLO adapter ready: {adapter_dir}")
        return

    import torch
    import ultralytics
    from ultralytics import YOLO

    trainer_class = build_full_image_trainer()
    model = YOLO(args.model)
    model.train(
        trainer=trainer_class,
        data=str(adapter_dir),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(run_root),
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=True,
        fliplr=0.0,
        flipud=0.0,
        erasing=0.0,
        dropout=0.0,
        plots=True,
        val=True,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir).resolve()
    results_csv = run_dir / "results.csv"
    best_source = run_dir / "weights" / "best.pt"
    if not results_csv.is_file() or not best_source.is_file():
        raise FileNotFoundError(f"Training artifacts are incomplete in {run_dir}")

    best_destination = model_dir / "yolo_cls_best.pt"
    shutil.copy2(best_source, best_destination)
    history, training_summary = convert_results(results_csv)
    save_json(output_dir / "history_yolo.json", history)
    copy_training_artifacts(run_dir, output_dir)

    metrics = {
        "model": "yolo",
        "architecture": "yolov8n-cls",
        "dataset_id": args.dataset_id,
        "dataset_variant": args.dataset_variant,
        "checkpoint": "models/yolo/yolo_cls_best.pt",
        "primary_metric_split": "valid_unseen",
        "official_evaluation_split": "test",
        "training_summary": training_summary,
        "valid_unseen": {
            "loss": training_summary["valid_loss_at_best_checkpoint"],
            "accuracy": training_summary["best_valid_unseen_accuracy"],
        },
        "test": None,
        "valid_traincopy": None,
    }
    save_json(output_dir / "metrics_yolo.json", metrics)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "dataset_variant": args.dataset_variant,
        "experiment_name": args.name,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "model": args.model,
        "training_strategy": "transfer_learning_from_imagenet_pretrained_yolov8n_cls",
        "split_policy": {
            "train": "weight updates",
            "valid_unseen": "early stopping and best checkpoint selection",
            "test": "reserved for post-training evaluation",
            "valid_traincopy": "auxiliary only; excluded from selection",
        },
        "input_pipeline": {
            "local_preprocessing": "aspect-ratio resize and zero-pad to 224x224",
            "training_transform": "full-image resize to 224x224 and ToTensor",
            "random_resized_crop": False,
            "online_auto_augment": False,
            "online_random_erasing": False,
            "online_flip": False,
        },
        "hyperparameters": {
            "max_epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "optimizer": "auto",
            "amp": True,
            "workers": args.workers,
            "seed": args.seed,
            "deterministic": True,
            "device": args.device,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_summary": training_summary,
    }
    save_json(output_dir / "experiment_manifest.json", manifest)

    print("\nTraining complete")
    print(f"Best checkpoint        : {best_destination}")
    print(f"Best checkpoint epoch  : {training_summary['best_checkpoint_epoch']}")
    print(f"Best validation Top-1  : {training_summary['best_valid_unseen_accuracy']:.4f}")
    print(f"Minimum validation loss: {training_summary['minimum_valid_unseen_loss']:.6f}")
    print("Next step: run src/evaluate_yolo.py on valid_unseen and test.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
