"""Evaluate a selected YOLO-cls checkpoint without modifying model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CLASS_NAMES = [
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


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def model_class_names(model) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def set_full_image_transform(model, imgsz: int) -> None:
    import torch
    import torchvision.transforms as transforms

    model.model.transforms = transforms.Compose(
        [
            transforms.Resize(
                (imgsz, imgsz),
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


def collect_split(
    split_dir: Path,
    class_names: list[str],
) -> tuple[list[str], list[int]]:
    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if actual_classes != class_names:
        raise ValueError(
            f"Class mismatch in {split_dir}. Expected={class_names}, actual={actual_classes}"
        )

    paths: list[str] = []
    targets: list[int] = []
    for target, class_name in enumerate(class_names):
        class_paths = sorted(
            path
            for path in (split_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(f"No images found in {split_dir / class_name}")
        paths.extend(str(path) for path in class_paths)
        targets.extend([target] * len(class_paths))
    return paths, targets


def evaluate_split(
    model,
    split_dir: Path,
    class_names: list[str],
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix

    image_paths, targets = collect_split(split_dir, class_names)
    predictions: list[int] = []
    chunk_size = max(batch * 4, 128)

    print(f"Evaluating {split_dir.name}: {len(image_paths):,} images")
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        results = model.predict(
            source=chunk,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )
        predictions.extend(int(result.probs.top1) for result in results)

    if len(predictions) != len(targets):
        raise RuntimeError(
            f"Prediction count mismatch: {len(predictions)} predictions, {len(targets)} targets"
        )

    accuracy = sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )
    return {
        "loss": None,
        "loss_note": "Cross-entropy loss was not measured by this prediction pass.",
        "accuracy": round(float(accuracy), 6),
        "samples": len(targets),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def update_metrics(metrics_path: Path, evaluation: dict[str, Any]) -> None:
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "model": "yolo",
            "architecture": "yolov8n-cls",
            "checkpoint": "models/yolo/yolo_cls_best.pt",
        }

    metrics["dataset_id"] = evaluation.get("dataset_id")
    metrics["dataset_variant"] = evaluation.get("dataset_variant")

    for split in ["valid_unseen", "test", "valid_traincopy"]:
        split_result = evaluation.get(split)
        if split_result is None:
            if split not in metrics:
                metrics[split] = None
            continue
        report = split_result["classification_report"]
        previous_loss = None
        if isinstance(metrics.get(split), dict):
            previous_loss = metrics[split].get("loss")
        metrics[split] = {
            "loss": previous_loss,
            "accuracy": split_result["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "samples": split_result["samples"],
        }
    save_json(metrics_path, metrics)


def save_class_metrics_csv(evaluation: dict[str, Any], output_path: Path) -> None:
    import pandas as pd

    report = evaluation["test"]["classification_report"]
    rows = [
        {
            "class": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "support": int(report[class_name]["support"]),
        }
        for class_name in evaluation["class_names"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-cls on valid_unseen and official test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="models/yolo/yolo_cls_best.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/yolo/evaluation_yolo_cls_best.json",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="outputs/yolo/metrics_yolo.json",
    )
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--dataset_id",
        choices=["raw_original", "raw_cleaning"],
        default=None,
    )
    parser.add_argument("--dataset_variant", choices=["v1", "v2"], default=None)
    parser.add_argument("--include_valid_traincopy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model_path)
    data_dir = resolve_path(args.data_dir)
    output_path = resolve_path(args.output)
    metrics_path = resolve_path(args.metrics_output)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    set_full_image_transform(model, args.imgsz)
    class_names = model_class_names(model)
    if class_names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"Unexpected checkpoint class mapping. Expected={EXPECTED_CLASS_NAMES}, actual={class_names}"
        )

    result: dict[str, Any] = {
        "checkpoint": str(model_path),
        "data_dir": str(data_dir),
        "dataset_id": args.dataset_id,
        "dataset_variant": args.dataset_variant,
        "class_names": class_names,
        "primary_metric_split": "valid_unseen",
        "official_evaluation_split": "test",
        "input_transform": "full-image resize to 224x224; no crop; no TTA",
    }

    for split_name in ["valid_unseen", "test"]:
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")
        result[split_name] = evaluate_split(
            model,
            split_dir,
            class_names,
            args.imgsz,
            args.batch,
            args.device,
        )

    auxiliary_dir = data_dir / "valid_traincopy"
    if args.include_valid_traincopy and auxiliary_dir.is_dir():
        result["valid_traincopy"] = evaluate_split(
            model,
            auxiliary_dir,
            class_names,
            args.imgsz,
            args.batch,
            args.device,
        )
    else:
        result["valid_traincopy"] = None

    save_json(output_path, result)
    update_metrics(metrics_path, result)
    save_class_metrics_csv(result, output_path.parent / "yolo_class_metrics.csv")

    test_report = result["test"]["classification_report"]
    print(f"Saved evaluation: {output_path}")
    print(f"Validation accuracy: {result['valid_unseen']['accuracy']:.4f}")
    print(f"Test accuracy      : {result['test']['accuracy']:.4f}")
    print(f"Test macro F1      : {test_report['macro avg']['f1-score']:.4f}")
    print(f"Test weighted F1   : {test_report['weighted avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
