"""
src/train.py
============
Vòng lặp huấn luyện (Training Loop) cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Gồm các tính năng:
  - Tích hợp DataLoader (dataset.py) và Model (model.py).
  - Hỗ trợ huấn luyện đa giai đoạn (Multi-phase Freeze Strategy).
  - Tối ưu với Early Stopping và Learning Rate Scheduler.
  - Lưu log tiến trình (JSON/CSV) và Checkpoint tốt nhất.
  - Hỗ trợ đánh giá trên tập Valid Copy và Valid Unseen độc lập.

Thư viện cần thiết:
  pip install torch torchvision tqdm
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import create_dataloaders
from model import (
    build_model, save_checkpoint, switch_strategy,
    SUPPORTED_MODELS
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING & UTILS
# ─────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Định dạng số giây thành hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TrainingLogger:
    """Lưu trữ và xuất log huấn luyện (Loss, Accuracy) ra file JSON."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"log_{experiment_name}_{timestamp}.json")
        self.history = []
        self.meta = {"experiment": experiment_name, "start_time": timestamp}

    def log_epoch(self, epoch: int, metrics: Dict[str, float], lr: float, phase: str):
        entry = {
            "epoch": epoch,
            "phase": phase,
            "lr": lr,
            **metrics
        }
        self.history.append(entry)
        self._save()

    def _save(self):
        data = {"meta": self.meta, "history": self.history}
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: torch.device = None,
):
    """
    Mixup: trộn 2 bức ảnh trong cùng 1 batch lại với nhau.

    Nguyên tắc:
      mixed_x = λ * x_i + (1 - λ) * x_j   (x_j là x bị xáo trộn ngẫu nhiên)
      λ ∈ [0, 1] lấy mẫu từ phân phối Beta(alpha, alpha).

    Tác dụng: Buộc mô hình học interpolate giữa các lủp thay vì 'học vẹt'
    điểm ảnh nẹ riêng biệt → giảm Overfitting đáng kể.

    Args:
        x     : Tensor ảnh (B, C, H, W)
        y     : Tensor nhãn (B,)
        alpha : Tham số Beta (0.1–0.4 — nhỏ hơn → trộn ít hơn)
        device: Device hiện tại

    Returns:
        (mixed_x, y_a, y_b, lam) — ảnh đã trộn và 2 nhãn gốc
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Hàm Loss cho Mixup:
      Loss = λ * CE(pred, y_a) + (1 - λ) * CE(pred, y_b)

    Đảm bảo gradient được cập nhật theo đú́ng tỷ lệ đóng góp của mỗi nhãn.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CORE
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.2,
) -> Tuple[float, float]:
    """
    Huấn luyện mô hình 1 epoch với Mixup Augmentation.
    Trả về (train_loss, train_acc).

    Args:
        mixup_alpha : Tham số Beta cho Mixup (0 = tắt Mixup, khuyến nghị 0.2).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False, ncols=80)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # ── MIXUP: trộn 2 bức ảnh trong cùng batch ──
        mixed_inputs, labels_a, labels_b, lam = mixup_data(
            inputs, labels, alpha=mixup_alpha, device=device
        )

        # Forward với ảnh đã trộn
        outputs = model(mixed_inputs)

        # Backward với Mixup Loss
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()

        # Thống kê (Accuracy ước tính theo nhãn chiếm tỷ trọng lớn hơn)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        # Lấy nhãn gốc (labels_a, y_b có tỷ lệ lam và 1-lam)
        correct += (
            lam * torch.sum(preds == labels_a).item()
            + (1 - lam) * torch.sum(preds == labels_b).item()
        )
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  Valid"
) -> Tuple[float, float]:
    """Đánh giá mô hình. Trả về (loss, acc)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False, ncols=80)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc  = correct / max(total, 1)
    return epoch_loss, epoch_acc


def _get_trainable_params(model: nn.Module):
    """Lấy danh sách các tham số có requires_grad=True."""
    return filter(lambda p: p.requires_grad, model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args):
    """Quy trình huấn luyện chính."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device đang sử dụng: {device}\n")

    # 1. Khởi tạo DataLoaders
    train_dl, valid_dl, test_dl, class_to_idx = create_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
        pin_memory=torch.cuda.is_available(),
    )
    num_classes = len(class_to_idx)

    # Nếu được yêu cầu, tách Valid thành Unseen và Copy để log chi tiết
    valid_unseen_dl, _ = create_dataloaders(
        processed_dir=args.processed_dir, batch_size=args.batch_size, num_workers=args.num_workers, valid_suffix="_unseen"
    )[1:3] # Chỉ lấy valid_dl (index 1), _ là placeholder
    valid_copy_dl, _ = create_dataloaders(
        processed_dir=args.processed_dir, batch_size=args.batch_size, num_workers=args.num_workers, valid_suffix="_copy"
    )[1:3]

    # Cần tạo lại valid_unseen_dl và valid_copy_dl đúng cách
    from dataset import create_single_loader
    valid_unseen_dl, _ = create_single_loader(
        os.path.join(args.processed_dir, "valid"), batch_size=args.batch_size, num_workers=args.num_workers, suffix_filter="_unseen", class_to_idx=class_to_idx
    )
    valid_copy_dl, _ = create_single_loader(
        os.path.join(args.processed_dir, "valid"), batch_size=args.batch_size, num_workers=args.num_workers, suffix_filter="_copy", class_to_idx=class_to_idx
    )

    # 2. Khởi tạo Mô hình
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        freeze_strategy="head_only",  # Bắt đầu với Phase 1
        device=device
    )

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer ban đầu (cho Phase 1)
    base_lr = args.lr
    optimizer = optim.Adam(_get_trainable_params(model), lr=base_lr, weight_decay=1e-4)
    
    # Scheduler: Giảm LR nếu val_loss không giảm
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4. Cấu hình Multi-phase Training
    # Phase 1: head_only
    phase1_epochs = args.phase1_epochs
    # Phase 2: partial (mở 1/3 cuối)
    phase2_epochs = args.phase2_epochs
    total_epochs  = phase1_epochs + phase2_epochs

    current_phase = "Phase 1 (Head Only)"

    # Cấu hình Early Stopping
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    epochs_no_improve = 0
    patience = args.patience

    # Khởi tạo Logger
    logger = TrainingLogger(log_dir=args.checkpoint_dir, experiment_name=args.model)

    start_time = time.time()
    print(f"{'═'*70}")
    print(f"BẮT ĐẦU HUẤN LUYỆN: {args.model.upper()} | Tổng Epoch: {total_epochs}")
    print(f"{'═'*70}")

    try:
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()

            # --- KIỂM TRA CHUYỂN PHASE ---
            if epoch == phase1_epochs + 1 and phase2_epochs > 0:
                print(f"\n[CHUYỂN GIAO] Bắt đầu Phase 2: Fine-tune (Partial Freeze)")
                current_phase = "Phase 2 (Partial)"
                switch_strategy(model, "partial")
                # Khởi tạo lại optimizer cho các tham số mới mở, giảm LR đi 10 lần
                optimizer = optim.Adam(_get_trainable_params(model), lr=base_lr * 0.1, weight_decay=1e-4)
                # Khởi tạo lại scheduler
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                # Reset early stopping
                epochs_no_improve = 0

            # Lấy LR hiện tại
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch}/{total_epochs} [{current_phase}] | LR: {current_lr:.2e}")
            print("-" * 50)

            # --- TRAIN ---
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)

            # --- VALID (Chung) ---
            val_loss, val_acc = evaluate(model, valid_dl, criterion, device)

            # --- VALID (Chi tiết) ---
            _, unseen_acc = evaluate(model, valid_unseen_dl, criterion, device, desc="  Val(Unseen)")
            _, copy_acc   = evaluate(model, valid_copy_dl, criterion, device, desc="  Val(Copy)")

            # --- SCHEDULER STEP ---
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start
            
            # --- IN KẾT QUẢ ---
            print(f"  Train : Loss {train_loss:.4f} | Acc {train_acc:.4f}")
            print(f"  Valid : Loss {val_loss:.4f} | Acc {val_acc:.4f}  (Unseen: {unseen_acc:.4f}, Copy: {copy_acc:.4f})")
            print(f"  Time  : {format_time(epoch_time)}")

            # --- LOGGING ---
            metrics = {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_unseen_acc": unseen_acc, "val_copy_acc": copy_acc
            }
            logger.log_epoch(epoch, metrics, current_lr, current_phase)

            # --- LƯU CHECKPOINT & EARLY STOPPING ---
            # Lưu nếu acc tăng hoặc loss giảm
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                is_best = True
            elif val_acc > best_val_acc:
                # Nếu loss không giảm nhưng acc tăng thì vẫn là tín hiệu tốt
                best_val_acc = val_acc
                epochs_no_improve = 0
                is_best = True
            else:
                epochs_no_improve += 1

            if is_best:
                best_val_acc = max(best_val_acc, val_acc)
                save_checkpoint(
                    model=model, optimizer=optimizer, epoch=epoch,
                    metrics=metrics, checkpoint_dir=args.checkpoint_dir, is_best=True
                )
                
                # --- LƯU CONFIG FILE CHO APP.PY ---
                config_path = os.path.join(args.checkpoint_dir, f"{args.model}_best_config.json")
                try:
                    manifest_path = os.path.join(args.processed_dir, ".pipeline_manifest.json")
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    pipeline = manifest.get("_meta", {}).get("pipeline", "baseline_v1")
                except Exception:
                    pipeline = "baseline_v1"
                
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump({"pipeline": pipeline}, f, indent=4)
            
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING] Kích hoạt sau {patience} epochs không cải thiện Validation Loss/Acc.")
                if epoch < phase1_epochs:
                    print("Gợi ý: Cân nhắc giảm phase1_epochs.")
                break

    except KeyboardInterrupt:
        print("\n[NGẮT BỞI NGƯỜI DÙNG] Dừng huấn luyện.")

    total_time = time.time() - start_time
    print(f"\n{'═'*70}")
    print(f"KẾT THÚC HUẤN LUYỆN | Tổng thời gian: {format_time(total_time)}")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.4f}")
    
    # Đánh giá Test set cuối cùng với Best Model
    print(f"\nĐánh giá trên tập TEST độc lập với Best Model...")
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pth")
    if os.path.exists(best_ckpt_path):
        from model import load_for_inference
        best_model = load_for_inference(best_ckpt_path, num_classes, device)
        test_loss, test_acc = evaluate(best_model, test_dl, criterion, device, desc="  Test")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    else:
        print("  Không tìm thấy checkpoint tốt nhất để chạy Test.")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    import argparse
    import sys
    
    # Fix for printing Vietnamese characters in Windows console
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Vehicle Type Recognition — Training Loop")
    
    # Data & Paths
    parser.add_argument("--processed_dir", type=str, default=r"g:\Data\Projects\VehicleTypeRecognition\data\processed", help="Thư mục dữ liệu đã xử lý")
    parser.add_argument("--checkpoint_dir", type=str, default=r"g:\Data\Projects\VehicleTypeRecognition\models", help="Thư mục lưu mô hình và log")
    
    # Model
    parser.add_argument("--model", type=str, default="resnet50", choices=SUPPORTED_MODELS, help="Kiến trúc mô hình")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate ban đầu (cho Phase 1)")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu (Windows dùng 0)")
    
    # Multi-phase Config
    parser.add_argument("--phase1_epochs", type=int, default=5, help="Số epoch cho Phase 1 (Chỉ train Head)")
    parser.add_argument("--phase2_epochs", type=int, default=10, help="Số epoch cho Phase 2 (Fine-tune 1/3 Backbone)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()
    
    train_model(args)
