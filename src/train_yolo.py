"""Train YOLOv8m-cls on a locally prepared Vehicle-10 dataset.

The local pipeline already produces square 224x224 images and performs the
offline V1/V2 augmentation. A custom Ultralytics classification dataset keeps
the full image instead of applying the default RandomResizedCrop and disables
additional online augmentation so V1 and V2 remain directly comparable.

The improved YOLO configuration uses a larger YOLOv8m-cls checkpoint,
Class-Balanced Focal Loss based on unique source-image counts, and conservative
anti-overfitting hyperparameters for the offline-augmented Clean V2 training
set.

The test split is never evaluated by this training script. Use
``src/evaluate_yolo.py`` after the best validation checkpoint is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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
AUGMENTED_STEM_PATTERN = re.compile(
    r"(.+?)_"
    r"(normal|rain|sun|night|gaussian_blur|motion_blur|unsharp_mask)_"
    r"(orig|geo)_\d+$"
)


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


def canonical_source_stem(path: Path) -> str:
    """Return the original source stem for an offline-augmented training file."""
    match = AUGMENTED_STEM_PATTERN.match(path.stem)
    return match.group(1) if match else path.stem


def infer_class_balanced_counts(data_dir: Path) -> dict[str, int]:
    """Infer unique source-image counts for Class-Balanced Focal Loss.

    Offline augmentation deliberately balances the number of training files per
    class, so raw file counts would make the CB weights almost equal. The loss
    should instead reflect the number of unique source images represented by the
    augmented files.
    """
    train_dir = data_dir / "train"
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = train_dir / class_name
        stems = {
            canonical_source_stem(path)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if not stems:
            raise ValueError(f"Cannot infer CB-Focal count for empty class: {class_name}")
        counts[class_name] = len(stems)
    return counts


class ClassBalancedFocalLoss:
    """Pickle-safe CB-Focal Loss with optional pairwise confusion margin."""

    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pair_indices: list[tuple[int, int]] | None = None,
        pair_margin: float = 0.30,
        pair_lambda: float = 0.0,
    ):
        import torch

        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.pair_indices = pair_indices
        self.pair_margin = pair_margin
        self.pair_lambda = pair_lambda
        self.current_epoch = 0
        weights = [
            (1.0 - beta) / (1.0 - (beta ** max(int(count), 1)))
            for count in class_counts
        ]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        self.weights = weights_tensor / weights_tensor.sum() * len(class_counts)

    def __call__(self, preds, batch):
        import torch
        import torch.nn.functional as F

        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"]
        
        # Check if targets are 1D class indices or 2D soft labels (MixUp/CutMix)
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[-1] == 1):
            if targets.ndim == 2:
                targets = targets.view(-1)
            targets = targets.long()
            num_classes = logits.shape[-1]
            one_hot = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
            hard_targets = targets
        else:
            # Soft labels of shape (batch_size, num_classes)
            one_hot = targets.to(logits.dtype)
            hard_targets = targets.argmax(dim=-1)

        if self.label_smoothing > 0:
            num_classes = logits.shape[-1]
            one_hot = (
                one_hot * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        focal_factor = (1.0 - probs).pow(self.gamma)
        weights = self.weights.to(device=logits.device, dtype=logits.dtype)
        loss = -(one_hot * focal_factor * log_probs * weights).sum(dim=-1).mean()
        
        if self.pair_lambda > 0 and self.pair_indices is not None:
            pair_terms = []
            for first_idx, second_idx in self.pair_indices:
                first_mask = hard_targets == first_idx
                second_mask = hard_targets == second_idx

                if first_mask.any():
                    first_margin = logits[first_mask, first_idx] - logits[first_mask, second_idx]
                    pair_terms.append(F.relu(self.pair_margin - first_margin))
                if second_mask.any():
                    second_margin = logits[second_mask, second_idx] - logits[second_mask, first_idx]
                    pair_terms.append(F.relu(self.pair_margin - second_margin))

            if pair_terms:
                pair_loss = torch.cat(pair_terms).mean()
                loss = loss + (self.pair_lambda * pair_loss)

        return loss, loss.detach()


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


def build_full_image_trainer(
    class_balanced_counts: dict[str, int],
    cb_beta: float,
    cb_gamma: float,
    cb_label_smoothing: float,
    pair_classes: list[tuple[str, str]] | None,
    pair_margin: float,
    pair_lambda: float,
):
    import torch
    import torchvision.transforms as transforms
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer

    cb_counts_list = [class_balanced_counts[class_name] for class_name in CLASS_NAMES]
    pair_indices = (
        [(CLASS_NAMES.index(p[0]), CLASS_NAMES.index(p[1])) for p in pair_classes]
        if pair_classes is not None
        else None
    )

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
                        mean=torch.tensor([0.485, 0.456, 0.406]),
                        std=torch.tensor([0.229, 0.224, 0.225]),
                    ),
                ]
            )

    class FullImageClassificationTrainer(ClassificationTrainer):
        def build_cb_focal_loss(self):
            return ClassBalancedFocalLoss(
                class_counts=cb_counts_list,
                beta=cb_beta,
                gamma=cb_gamma,
                label_smoothing=cb_label_smoothing,
                pair_indices=pair_indices,
                pair_margin=pair_margin,
                pair_lambda=pair_lambda,
            )

        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model.criterion = self.build_cb_focal_loss()
            return model

        def setup_model(self):
            checkpoint = super().setup_model()
            self.model.criterion = self.build_cb_focal_loss()
            return checkpoint

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            self.model.criterion.current_epoch = int(self.epoch) + 1
            return batch

        def save_model(self):
            """Save checkpoints without transient training state."""
            stashed = []
            candidates = [getattr(self, "model", None)]
            ema = getattr(self, "ema", None)
            if ema is not None:
                candidates.append(getattr(ema, "ema", None))

            for module in candidates:
                if module is None:
                    continue
                state = {}
                if hasattr(module, "criterion"):
                    state["criterion"] = module.criterion
                    delattr(module, "criterion")
                if state:
                    stashed.append((module, state))

            try:
                return super().save_model()
            finally:
                for module, state in stashed:
                    if "criterion" in state:
                        setattr(module, "criterion", state["criterion"])

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

    maximum_accuracy = max(history, key=lambda row: row["valid_unseen_acc"])
    minimum_loss = min(history, key=lambda row: row["valid_unseen_loss"])
    summary = {
        "completed_epochs": len(history),
        "checkpoint_selection_metric": "minimum_valid_unseen_loss",
        "best_checkpoint_epoch": minimum_loss["epoch"],
        "best_valid_unseen_accuracy": minimum_loss["valid_unseen_acc"],
        "valid_loss_at_best_checkpoint": minimum_loss["valid_unseen_loss"],
        "minimum_valid_unseen_loss_epoch": minimum_loss["epoch"],
        "minimum_valid_unseen_loss": minimum_loss["valid_unseen_loss"],
        "maximum_valid_unseen_accuracy_epoch": maximum_accuracy["epoch"],
        "maximum_valid_unseen_accuracy": maximum_accuracy["valid_unseen_acc"],
        "valid_loss_at_maximum_accuracy": maximum_accuracy["valid_unseen_loss"],
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
        description="Train improved YOLOv8m-cls with valid_unseen as the only selection split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8m-cls.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch", type=int, default=32)
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
    parser.add_argument("--lr0", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--cb_gamma", type=float, default=2.0)
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler.")

    parser.add_argument(
        "--pair_margin_classes",
        type=str,
        default="car,truck",
        help="Comma-separated class pair for optional confusion-aware margin loss.",
    )
    parser.add_argument(
        "--pair_margin",
        type=float,
        default=0.30,
        help="Required logit margin between the true class and paired confusing class.",
    )
    parser.add_argument(
        "--pair_lambda",
        type=float,
        default=0.0,
        help="Weight for pairwise confusion-aware margin loss. Use 0 to disable.",
    )
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
    class_balanced_counts = infer_class_balanced_counts(data_dir)
    pair_classes_list = []
    if args.pair_lambda > 0:
        for pair_str in args.pair_margin_classes.split(";"):
            if not pair_str.strip():
                continue
            pair_classes = tuple(
                item.strip() for item in pair_str.split(",") if item.strip()
            )
            if len(pair_classes) != 2:
                raise ValueError(
                    "Each pair in --pair_margin_classes must contain exactly two comma-separated classes"
                )
            invalid_pair_classes = [name for name in pair_classes if name not in CLASS_NAMES]
            if invalid_pair_classes:
                raise ValueError(f"Invalid pair margin classes: {invalid_pair_classes}")
            pair_classes_list.append(pair_classes)

    dataset_report["class_balanced_focal_counts"] = class_balanced_counts

    dataset_report["pair_margin_loss"] = {
        "enabled": args.pair_lambda > 0,
        "classes": [list(p) for p in pair_classes_list] if pair_classes_list else None,
        "margin": args.pair_margin,
        "lambda": args.pair_lambda,
    }
    save_json(output_dir / "dataset_audit.json", dataset_report)
    prepare_yolo_adapter(data_dir, adapter_dir)

    print("=" * 72)
    print("Improved YOLOv8-cls Vehicle-10 training")
    print("=" * 72)
    print(f"Model checkpoint      : {args.model}")
    print(f"Dataset source        : {args.dataset_id}")
    print(f"Augmentation version  : {args.dataset_variant.upper()}")
    print(f"Dataset root          : {data_dir}")
    print(f"Training split        : {data_dir / 'train'}")
    print(f"Primary validation    : {data_dir / 'valid_unseen'}")
    print(f"Official test         : reserved; not used by train_yolo.py")
    print(f"Input policy          : full image resize to {args.imgsz}x{args.imgsz}, no crop")
    print("Online augmentation   : disabled (offline V1/V2 only)")
    print("Loss                  : Class-Balanced Focal Loss")

    if args.pair_lambda > 0 and pair_classes_list:
        pair_desc = ", ".join(f"{p[0]} <-> {p[1]}" for p in pair_classes_list)
        print(
            "Pair margin loss      : "
            f"{pair_desc}, "
            f"margin={args.pair_margin}, lambda={args.pair_lambda}"
        )
    else:
        print("Pair margin loss      : disabled")
    print(f"CB-Focal counts       : {class_balanced_counts}")
    print(f"Seed / deterministic  : {args.seed} / True")
    print("=" * 72)

    if args.prepare_only:
        print(f"YOLO adapter ready: {adapter_dir}")
        return

    import torch
    import ultralytics
    from ultralytics import YOLO

    trainer_class = build_full_image_trainer(
        class_balanced_counts=class_balanced_counts,
        cb_beta=args.cb_beta,
        cb_gamma=args.cb_gamma,
        cb_label_smoothing=args.label_smoothing,
        pair_classes=pair_classes_list,
        pair_margin=args.pair_margin,
        pair_lambda=args.pair_lambda,

    )
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
        optimizer="AdamW",
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        label_smoothing=0.0,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        cache=False,
        amp=True,
        augment=False,
        auto_augment=None,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        copy_paste=0.0,
        cutmix=0.0,
        erasing=0.0,
        dropout=args.dropout,
        mixup=args.mixup,
        cos_lr=args.cos_lr,
        plots=True,
        val=True,
        verbose=True,
        save_period=1,
    )

    run_dir = Path(model.trainer.save_dir).resolve()
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results.csv is missing in {run_dir}")

    history, training_summary = convert_results(results_csv)
    selected_epoch = int(training_summary["best_checkpoint_epoch"])
    weights_dir = run_dir / "weights"
    checkpoint_candidates = [
        weights_dir / f"epoch{selected_epoch}.pt",
        weights_dir / f"epoch{selected_epoch:03d}.pt",
    ]
    if selected_epoch == int(training_summary["completed_epochs"]):
        checkpoint_candidates.append(weights_dir / "last.pt")
    best_source = next((candidate for candidate in checkpoint_candidates if candidate.is_file()), None)
    if best_source is None:
        available = sorted(path.name for path in weights_dir.glob("*.pt"))
        raise FileNotFoundError(
            "Cannot find checkpoint for minimum valid_unseen_loss epoch "
            f"{selected_epoch}. Available weights: {available}. "
            "Keep save_period=1 enabled when selecting YOLO by loss."
        )

    best_destination = model_dir / "yolo_cls_best.pt"
    shutil.copy2(best_source, best_destination)
    save_json(output_dir / "history_yolo.json", history)
    copy_training_artifacts(run_dir, output_dir)

    metrics = {
        "model": "yolo",
        "architecture": Path(args.model).stem,
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
        "training_strategy": (
            f"transfer_learning_from_imagenet_pretrained_{Path(args.model).stem}"
            "_with_cb_focal_anti_overfit_config"
        ),
        "split_policy": {
            "train": "weight updates",
            "valid_unseen": "early stopping and best checkpoint selection",
            "test": "reserved for post-training evaluation",
            "valid_traincopy": "auxiliary only; excluded from selection",
        },
        "input_pipeline": {
            "local_preprocessing": "aspect-ratio resize and reflective-pad to 224x224",
            "training_transform": "full-image resize to 224x224 and ToTensor",
            "random_resized_crop": False,
            "online_auto_augment": False,
            "online_random_erasing": False,
            "online_flip": False,
            "online_mixup": args.mixup,
        },
        "hyperparameters": {
            "max_epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "optimizer": "AdamW",
            "lr0": args.lr0,
            "warmup_epochs": args.warmup_epochs,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "cos_lr": args.cos_lr,
            "ultralytics_label_smoothing": 0.0,
            "cb_focal_label_smoothing": args.label_smoothing,
            "mixup": args.mixup,
            "loss": "ClassBalancedFocalLoss",
            "cb_beta": args.cb_beta,
            "cb_gamma": args.cb_gamma,

            "pair_margin_loss": {
                "enabled": args.pair_lambda > 0,
                "classes": [list(pair) for pair in pair_classes_list] if pair_classes_list else None,
                "margin": args.pair_margin,
                "lambda": args.pair_lambda,
            },
            "class_balanced_counts": class_balanced_counts,
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
