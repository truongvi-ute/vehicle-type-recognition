"""
Main training entry point for VehicleTypeRecognition.

The canonical split policy is:
    - train: data/augmented/train
    - primary validation: data/augmented/valid_unseen
    - auxiliary validation: data/augmented/valid_traincopy, if present
    - official evaluation: data/augmented/test

Only valid_unseen is used for early stopping and best-checkpoint selection.
valid_traincopy is reported as an auxiliary sanity check and test is reported
only after training finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import (  # noqa: E402
    AUX_VALID_SPLIT,
    PRIMARY_VALID_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    create_dataloaders,
)
from src.model import (  # noqa: E402
    build_model,
    get_backbone_parameters,
    get_head_parameters,
    load_checkpoint,
    model_summary,
    save_checkpoint,
    switch_strategy,
)


PHASE1_EPOCHS = 5
CHECKPOINT_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_NAME_MAP: Dict[str, str] = {
    "resnet50": "resnet50",
    "vit": "vit_base_patch16_224",
}


class EarlyStopping:
    """Early stopping driven by valid_unseen loss only."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        checkpoint_dir: str = CHECKPOINT_DIR,
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.early_stop = False

    def step(
        self,
        valid_unseen_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        if valid_unseen_loss < self.best_loss - self.min_delta:
            improvement = self.best_loss - valid_unseen_loss
            self.best_loss = valid_unseen_loss
            self.best_epoch = epoch
            self.counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=self.checkpoint_dir,
                is_best=True,
            )
            if self.verbose:
                print(
                    "  [EarlyStopping] valid_unseen_loss improved "
                    f"by {improvement:.6f}; saved best checkpoint."
                )
            return

        self.counter += 1
        if self.verbose:
            print(
                "  [EarlyStopping] no valid_unseen improvement "
                f"({self.counter}/{self.patience}). "
                f"Best={self.best_loss:.6f} @ epoch {self.best_epoch}."
            )
        if self.counter >= self.patience:
            self.early_stop = True


def build_phase1_optimizer(model: nn.Module, lr_head: float) -> torch.optim.AdamW:
    head_params = [p for p in get_head_parameters(model) if p.requires_grad]
    return torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)


def build_phase2_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_backbone: float,
) -> torch.optim.AdamW:
    head_params = [p for p in get_head_parameters(model) if p.requires_grad]
    backbone_params = [p for p in get_backbone_parameters(model) if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=1e-4,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch [{epoch:>3}/{total_epochs}] {TRAIN_SPLIT}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"\n  [OOM] skipped train batch {batch_idx}.")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

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

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


def _metric_record(loss: float, accuracy: float) -> Dict[str, float]:
    return {
        "loss": round(loss, 6),
        "accuracy": round(accuracy, 6),
    }


def _load_best_if_available(
    model: nn.Module,
    full_model_name: str,
    device: torch.device,
    checkpoint_dir: str,
) -> Optional[str]:
    best_path = Path(checkpoint_dir) / f"{full_model_name}_best.pth"
    if not best_path.is_file():
        return None
    load_checkpoint(str(best_path), model=model, optimizer=None, device=device)
    return str(best_path)


def train(
    data_dir: str = "data/augmented",
    model_key: str = "resnet50",
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 7,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-5,
    num_workers: int = 0,
    eval_traincopy_each_epoch: bool = False,
) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model_name = MODEL_NAME_MAP[model_key]

    print("=" * 72)
    print("VehicleTypeRecognition training")
    print("=" * 72)
    print(f"Device                  : {device}")
    print(f"Model                   : {model_key} ({full_model_name})")
    print(f"Data root               : {data_dir}")
    print(f"Train split             : {data_dir}/{TRAIN_SPLIT}")
    print(f"Primary validation      : {data_dir}/{PRIMARY_VALID_SPLIT}")
    print(f"Auxiliary validation    : {data_dir}/{AUX_VALID_SPLIT} (not primary)")
    print(f"Official test           : {data_dir}/{TEST_SPLIT}")
    print("=" * 72)

    train_loader, valid_unseen_loader, valid_traincopy_loader, test_loader, class_names = (
        create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )
    print(f"Classes                 : {len(class_names)} {class_names}")
    print(f"Train batches           : {len(train_loader)}")
    print(f"valid_unseen batches    : {len(valid_unseen_loader)}")
    print(
        "valid_traincopy batches : "
        f"{len(valid_traincopy_loader) if valid_traincopy_loader is not None else 'not available'}"
    )
    print(f"Test batches            : {len(test_loader)}")

    model = build_model(
        model_name=full_model_name,
        num_classes=len(class_names),
        pretrained=True,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_key)
    output_dir = os.path.join(OUTPUT_DIR, model_key)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    stopper = EarlyStopping(patience=patience, checkpoint_dir=checkpoint_dir)
    history: List[Dict[str, float]] = []
    current_phase = ""
    optimizer: Optional[torch.optim.AdamW] = None

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        desired_phase = "head_only" if epoch <= PHASE1_EPOCHS else "partial"

        if desired_phase != current_phase:
            current_phase = desired_phase
            switch_strategy(model, current_phase)
            optimizer = (
                build_phase1_optimizer(model, lr_head)
                if current_phase == "head_only"
                else build_phase2_optimizer(model, lr_head, lr_backbone)
            )
            print(f"\nPhase changed to {current_phase}")
            model_summary(model)

        assert optimizer is not None

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        valid_unseen_loss, valid_unseen_acc = evaluate_loss_acc(
            model=model,
            loader=valid_unseen_loader,
            criterion=criterion,
            device=device,
            split_name=PRIMARY_VALID_SPLIT,
        )

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": round(train_loss, 6),
            "valid_unseen_loss": round(valid_unseen_loss, 6),
            "valid_unseen_acc": round(valid_unseen_acc, 6),
            "elapsed_s": round(time.perf_counter() - epoch_start, 2),
        }

        if eval_traincopy_each_epoch and valid_traincopy_loader is not None:
            aux_loss, aux_acc = evaluate_loss_acc(
                model=model,
                loader=valid_traincopy_loader,
                criterion=criterion,
                device=device,
                split_name=AUX_VALID_SPLIT,
            )
            row["valid_traincopy_loss"] = round(aux_loss, 6)
            row["valid_traincopy_acc"] = round(aux_acc, 6)

        history.append(row)
        print(
            f"Epoch [{epoch:>3}/{epochs}] [{current_phase}] "
            f"train_loss={train_loss:.4f} "
            f"valid_unseen_loss={valid_unseen_loss:.4f} "
            f"valid_unseen_acc={valid_unseen_acc * 100:.2f}%"
        )

        checkpoint_metrics = {
            "train_loss": train_loss,
            "valid_unseen_loss": valid_unseen_loss,
            "valid_unseen_acc": valid_unseen_acc,
        }
        stopper.step(
            valid_unseen_loss=valid_unseen_loss,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=checkpoint_metrics,
        )
        if stopper.early_stop:
            print(f"\nStopped early at epoch {epoch}.")
            break

    history_path = Path(output_dir) / f"history_{model_key}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    checkpoint_path = _load_best_if_available(model, full_model_name, device, checkpoint_dir)

    valid_unseen_loss, valid_unseen_acc = evaluate_loss_acc(
        model=model,
        loader=valid_unseen_loader,
        criterion=criterion,
        device=device,
        split_name=PRIMARY_VALID_SPLIT,
    )
    test_loss, test_acc = evaluate_loss_acc(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name=TEST_SPLIT,
    )

    final_metrics: Dict[str, object] = {
        "model": model_key,
        "architecture": full_model_name,
        "checkpoint": checkpoint_path,
        "primary_metric_split": PRIMARY_VALID_SPLIT,
        "official_evaluation_split": TEST_SPLIT,
        PRIMARY_VALID_SPLIT: _metric_record(valid_unseen_loss, valid_unseen_acc),
        TEST_SPLIT: _metric_record(test_loss, test_acc),
    }

    if valid_traincopy_loader is not None:
        aux_loss, aux_acc = evaluate_loss_acc(
            model=model,
            loader=valid_traincopy_loader,
            criterion=criterion,
            device=device,
            split_name=AUX_VALID_SPLIT,
        )
        final_metrics[AUX_VALID_SPLIT] = _metric_record(aux_loss, aux_acc)
    else:
        final_metrics[AUX_VALID_SPLIT] = None

    metrics_path = Path(output_dir) / f"metrics_{model_key}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("\nTraining complete")
    print(f"Best epoch               : {stopper.best_epoch}")
    print(f"Best valid_unseen_loss   : {stopper.best_loss:.6f}")
    print(f"History                  : {history_path}")
    print(f"Separated metrics        : {metrics_path}")
    print(f"Test accuracy            : {test_acc * 100:.2f}%")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet/ViT with the canonical split strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--model", type=str, default="resnet50", choices=list(MODEL_NAME_MAP))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--eval_traincopy_each_epoch",
        action="store_true",
        help="Also report valid_traincopy each epoch. It remains auxiliary only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] data_dir does not exist: {args.data_dir}")
        sys.exit(1)

    train(
        data_dir=args.data_dir,
        model_key=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        num_workers=args.num_workers,
        eval_traincopy_each_epoch=args.eval_traincopy_each_epoch,
    )
