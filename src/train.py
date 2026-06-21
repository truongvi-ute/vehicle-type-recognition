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
import re
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AUGMENTED_STEM_PATTERN = re.compile(
    r"(.+?)_"
    r"(normal|rain|sun|night|gaussian_blur|motion_blur|unsharp_mask)_"
    r"(orig|geo)_\d+$"
)
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


def canonical_source_stem(path: Path) -> str:
    match = AUGMENTED_STEM_PATTERN.match(path.stem)
    return match.group(1) if match else path.stem


def infer_class_balanced_counts(data_dir: str, class_names: List[str]) -> List[int]:
    train_dir = Path(data_dir) / TRAIN_SPLIT
    counts: List[int] = []
    for class_name in class_names:
        class_dir = train_dir / class_name
        stems = {
            canonical_source_stem(path)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if not stems:
            raise ValueError(f"Cannot infer CB-Focal count for empty class: {class_name}")
        counts.append(len(stems))
    return counts


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss hỗ trợ cả nhãn 1D (đánh giá) và nhãn 2D soft labels (MixUp/CutMix).
    """
    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pair_indices: Optional[List[Tuple[int, int]]] = None,
        pair_margin: float = 0.30,
        pair_lambda: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.pair_indices = pair_indices
        self.pair_margin = pair_margin
        self.pair_lambda = pair_lambda
        
        # Tính toán trọng số Class-Balanced
        weights = []
        for count in class_counts:
            w = (1.0 - beta) / (1.0 - (beta ** count))
            weights.append(w)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        # Chuẩn hóa để tổng các trọng số bằng số lượng lớp (K=10)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum() * len(class_counts))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        if targets.ndim == 1:
            num_classes = logits.size(-1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
        else:
            targets_one_hot = targets
            
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            targets_one_hot = targets_one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        focal_term = ((1.0 - p) ** self.gamma) * log_p
        weighted_loss = - targets_one_hot * focal_term * self.weights
        
        loss = weighted_loss.sum(dim=-1).mean()

        if self.pair_lambda > 0 and self.pair_indices is not None:
            # Lấy nhãn cứng (hard labels) từ nhãn mềm (nếu ndim == 2 do MixUp/CutMix)
            hard_targets = targets if targets.ndim == 1 else targets.argmax(dim=-1)
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
                loss = loss + self.pair_lambda * pair_loss

        return loss



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_phase1_optimizer(
    model: nn.Module, lr_head: float
) -> torch.optim.AdamW:
    head_params = [p for p in get_head_parameters(model) if p.requires_grad]
    return torch.optim.AdamW(
        head_params,
        lr=lr_head,
        weight_decay=1e-4,
    )


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


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "arch": getattr(model, "_arch", "model"),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


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
    cb_beta: float = 0.999,
    cb_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    pair_margin_classes: str = "car,truck",
    pair_margin: float = 0.30,
    pair_lambda: float = 0.0,
    seed: int = 42,
    resume: Optional[str] = None,
    run_epochs: Optional[int] = None,
    skip_test: bool = False,
) -> List[Dict[str, float]]:
    set_seed(seed)
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
    print(f"Seed / deterministic    : {seed} / True")
    print("=" * 72)

    train_loader, valid_unseen_loader, valid_traincopy_loader, test_loader, class_names = (
        create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
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
    class_counts = infer_class_balanced_counts(data_dir, class_names)
    resume_payload = None
    start_epoch = 1
    if resume:
        resume_payload = load_checkpoint(resume, model=model, optimizer=None, device=device)
        start_epoch = int(resume_payload["epoch"]) + 1
        if start_epoch > epochs:
            raise ValueError(f"Resume epoch {start_epoch - 1} already reaches max epochs={epochs}")
        print(f"Resume                 : {resume} (next epoch {start_epoch})")
    pair_classes_list = []
    pair_indices = None
    if pair_lambda > 0:
        for pair_str in pair_margin_classes.split(";"):
            if not pair_str.strip():
                continue
            parsed_pair = tuple(
                item.strip() for item in pair_str.split(",") if item.strip()
            )
            if len(parsed_pair) != 2:
                raise ValueError(
                    "Each pair in --pair_margin_classes must contain exactly two comma-separated classes"
                )
            invalid_pair_classes = [name for name in parsed_pair if name not in class_names]
            if invalid_pair_classes:
                raise ValueError(f"Invalid pair margin classes: {invalid_pair_classes}")
            pair_classes_list.append(parsed_pair)
        pair_indices = [(class_names.index(p[0]), class_names.index(p[1])) for p in pair_classes_list]

    print(f"CB-Focal counts        : {dict(zip(class_names, class_counts))}")
    print(f"CB-Focal beta/gamma    : {cb_beta} / {cb_gamma}")
    print(f"Label smoothing        : {label_smoothing}")
    print(f"Online MixUp/CutMix    : {mixup_alpha} / {cutmix_alpha}")
    if pair_lambda > 0 and pair_classes_list:
        pair_desc = ", ".join(f"{p[0]} <-> {p[1]}" for p in pair_classes_list)
        print(
            "Pair margin loss       : "
            f"{pair_desc}, "
            f"margin={pair_margin}, lambda={pair_lambda}"
        )
    else:
        print("Pair margin loss       : disabled")

    criterion = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=cb_beta,
        gamma=cb_gamma,
        label_smoothing=label_smoothing,
        pair_indices=pair_indices,
        pair_margin=pair_margin,
        pair_lambda=pair_lambda,
    ).to(device)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_key)
    output_dir = os.path.join(OUTPUT_DIR, model_key)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    stopper = EarlyStopping(patience=patience, checkpoint_dir=checkpoint_dir)
    history: List[Dict[str, float]] = []
    current_phase = ""
    optimizer: Optional[torch.optim.AdamW] = None
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None

    end_epoch = min(epochs, start_epoch + run_epochs - 1) if run_epochs else epochs
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.perf_counter()
        desired_phase = (
            "head_only" if epoch <= PHASE1_EPOCHS
            else "partial"
        )

        if desired_phase != current_phase:
            current_phase = desired_phase
            switch_strategy(model, current_phase)
            optimizer = (
                build_phase1_optimizer(model, lr_head)
                if current_phase == "head_only"
                else build_phase2_optimizer(
                    model, lr_head, lr_backbone
                )
            )
            # Reset scheduler khi doi phase de tranh state cu anh huong
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
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

        # Step scheduler dua tren valid_unseen_loss
        if scheduler is not None:
            scheduler.step(valid_unseen_loss)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr_head
            row["lr"] = round(float(optimizer.param_groups[0]["lr"]), 8)

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
        lr_display = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch [{epoch:>3}/{epochs}] [{current_phase}] "
            f"train_loss={train_loss:.4f} "
            f"valid_unseen_loss={valid_unseen_loss:.4f} "
            f"valid_unseen_acc={valid_unseen_acc * 100:.2f}% "
            f"lr={lr_display:.2e}"
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
        save_last_checkpoint(
            Path(checkpoint_dir) / f"{full_model_name}_last.pth",
            model, optimizer, epoch, checkpoint_metrics
        )
        if epoch > PHASE1_EPOCHS and stopper.early_stop:
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
    if skip_test:
        test_loss, test_acc = float("nan"), float("nan")
    else:
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
        "hyperparameters": {
            "max_epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "cb_beta": cb_beta,
            "cb_gamma": cb_gamma,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,

            "seed": seed,
            "pair_margin_loss": {
                "enabled": pair_lambda > 0,
                "classes": [list(pair) for pair in pair_classes_list] if pair_classes_list else None,
                "margin": pair_margin,
                "lambda": pair_lambda,
            },
            "class_balanced_counts": dict(zip(class_names, class_counts)),
        },
        PRIMARY_VALID_SPLIT: _metric_record(valid_unseen_loss, valid_unseen_acc),
        TEST_SPLIT: None if skip_test else _metric_record(test_loss, test_acc),
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
    if not skip_test:
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
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--cb_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_epochs", type=int, default=None)
    parser.add_argument("--skip_test", action="store_true")
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
        cb_beta=args.cb_beta,
        cb_gamma=args.cb_gamma,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        pair_margin_classes=args.pair_margin_classes,
        pair_margin=args.pair_margin,
        pair_lambda=args.pair_lambda,

        seed=args.seed,
        resume=args.resume,
        run_epochs=args.run_epochs,
        skip_test=args.skip_test,
    )
