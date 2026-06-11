"""
Evaluate a trained checkpoint with separated split metrics.

Default behavior:
    - valid_unseen: secondary official report split
    - test: official final evaluation split
    - valid_traincopy: auxiliary check if the folder exists
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import AUX_VALID_SPLIT, PRIMARY_VALID_SPLIT, TEST_SPLIT, create_eval_loader  # noqa: E402
from src.model import load_for_inference  # noqa: E402


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    class_names: List[str],
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    pbar = tqdm(
        loader,
        desc=f"{split_name:>18} eval",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item())
        n_batches += 1
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.size(0))
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    metrics: Dict[str, object] = {
        "loss": round(total_loss / max(n_batches, 1), 6),
        "accuracy": round(total_correct / max(total_samples, 1), 6),
        "samples": total_samples,
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

    return metrics


def split_exists(data_dir: str, split_name: str) -> bool:
    return (Path(data_dir) / split_name).is_dir()


def evaluate_checkpoint(
    checkpoint: str,
    data_dir: str = "data/augmented",
    batch_size: int = 32,
    num_workers: int = 0,
    include_traincopy: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_unseen_loader, class_names = create_eval_loader(
        data_dir=data_dir,
        split=PRIMARY_VALID_SPLIT,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader, test_class_names = create_eval_loader(
        data_dir=data_dir,
        split=TEST_SPLIT,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if test_class_names != class_names:
        raise RuntimeError("Class names differ between valid_unseen and test.")

    model = load_for_inference(
        checkpoint_path=checkpoint,
        num_classes=len(class_names),
        device=device,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    result: Dict[str, object] = {
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "class_names": class_names,
        "primary_metric_split": PRIMARY_VALID_SPLIT,
        "official_evaluation_split": TEST_SPLIT,
        PRIMARY_VALID_SPLIT: evaluate_loader(
            model,
            valid_unseen_loader,
            criterion,
            device,
            PRIMARY_VALID_SPLIT,
            class_names,
        ),
        TEST_SPLIT: evaluate_loader(
            model,
            test_loader,
            criterion,
            device,
            TEST_SPLIT,
            class_names,
        ),
    }

    if include_traincopy and split_exists(data_dir, AUX_VALID_SPLIT):
        aux_loader, aux_class_names = create_eval_loader(
            data_dir=data_dir,
            split=AUX_VALID_SPLIT,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if aux_class_names != class_names:
            raise RuntimeError("Class names differ between valid_unseen and valid_traincopy.")
        result[AUX_VALID_SPLIT] = evaluate_loader(
            model,
            aux_loader,
            criterion,
            device,
            AUX_VALID_SPLIT,
            class_names,
        )
    else:
        result[AUX_VALID_SPLIT] = None

    if output_path is None:
        checkpoint_path = Path(checkpoint)
        model_key = checkpoint_path.parent.name
        checkpoint_stem = checkpoint_path.stem
        if model_key in ["resnet50", "vit", "yolo"]:
            output_path = str(Path("outputs") / model_key / f"evaluation_{checkpoint_stem}.json")
        else:
            output_path = str(Path("outputs") / f"evaluation_{checkpoint_stem}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["output_path"] = output_path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on valid_unseen, test, and optional valid_traincopy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--skip_valid_traincopy",
        action="store_true",
        help="Do not evaluate the auxiliary valid_traincopy split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_traincopy=not args.skip_valid_traincopy,
        output_path=args.output,
    )
    print(f"Saved separated evaluation metrics: {metrics['output_path']}")
    print(f"valid_unseen accuracy: {metrics[PRIMARY_VALID_SPLIT]['accuracy']:.4f}")  # type: ignore[index]
    print(f"test accuracy        : {metrics[TEST_SPLIT]['accuracy']:.4f}")  # type: ignore[index]
    if metrics.get(AUX_VALID_SPLIT) is not None:
        print(f"valid_traincopy acc : {metrics[AUX_VALID_SPLIT]['accuracy']:.4f}")  # type: ignore[index]
