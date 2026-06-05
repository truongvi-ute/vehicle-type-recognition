"""
src/train.py
============
Script huấn luyện chính cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Quy trình:
    1. Đọc cấu hình từ CLI (argparse).
    2. Tạo DataLoaders (train / valid / test).
    3. Khởi tạo mô hình (ResNet-50 hoặc ViT-B/16).
    4. Huấn luyện đa giai đoạn (Multi-phase Fine-tuning):
       - Epoch 1-5  : head_only  → chỉ train Classification Head.
       - Epoch 6+   : partial    → mở thêm 1/3 block cuối Backbone.
    5. EarlyStopping theo val_loss, lưu best checkpoint tự động.
    6. Xuất lịch sử loss/acc ra outputs/history_<model>.json.

Cách chạy:
    python src/train.py --data_dir data/augmented --model resnet50

Ví dụ đầy đủ:
    python src/train.py \\
        --data_dir    data/augmented \\
        --model       resnet50 \\
        --batch_size  32 \\
        --epochs      30 \\
        --patience    7 \\
        --lr_head     1e-3 \\
        --lr_backbone 1e-5

Phụ thuộc:
    pip install torch torchvision tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Import nội bộ ──────────────────────────────────────────────────────────────
# Thêm thư mục gốc vào sys.path để import src.*
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import create_dataloaders          # noqa: E402
from src.model import (                             # noqa: E402
    build_model,
    get_backbone_parameters,
    get_head_parameters,
    model_summary,
    save_checkpoint,
    switch_strategy,
)

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

#: Số epoch đầu chỉ train Head (Phase 1).
PHASE1_EPOCHS: int = 5

#: Tên thư mục lưu checkpoints.
CHECKPOINT_DIR: str = "models"

#: Tên thư mục xuất file lịch sử JSON.
OUTPUT_DIR: str = "outputs"

#: Ánh xạ tên model ngắn gọn (CLI) → tên đầy đủ cho build_model().
MODEL_NAME_MAP: Dict[str, str] = {
    "resnet50": "resnet50",
    "vit":      "vit_base_patch16_224",
}

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 — EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Theo dõi val_loss qua từng epoch và kích hoạt early_stop nếu
    val_loss không giảm sau ``patience`` epochs liên tiếp.

    Tự động lưu checkpoint tốt nhất (is_best=True) thông qua
    ``save_checkpoint()`` khi tìm thấy val_loss mới thấp hơn.

    Attributes:
        patience   (int)  : Số epoch chờ trước khi dừng.
        min_delta  (float): Ngưỡng cải thiện tối thiểu (val_loss phải giảm
                            ít nhất ``min_delta`` mới tính là "cải thiện").
        counter    (int)  : Số epoch liên tiếp không cải thiện.
        best_loss  (float): val_loss thấp nhất ghi nhận được.
        early_stop (bool) : Cờ báo hiệu dừng huấn luyện.

    Example:
        >>> stopper = EarlyStopping(patience=5, checkpoint_dir="models")
        >>> stopper.step(val_loss, model, optimizer, epoch, metrics)
        >>> if stopper.early_stop:
        ...     break
    """

    def __init__(
        self,
        patience:       int   = 7,
        min_delta:      float = 1e-4,
        checkpoint_dir: str   = CHECKPOINT_DIR,
        verbose:        bool  = True,
    ) -> None:
        """
        Args:
            patience       : Số epoch tối đa không cải thiện trước khi dừng.
            min_delta      : Val_loss phải giảm ít nhất ``min_delta`` để tính là tốt hơn.
            checkpoint_dir : Thư mục lưu file best checkpoint.
            verbose        : In thông báo khi cập nhật best loss.
        """
        self.patience:       int   = patience
        self.min_delta:      float = min_delta
        self.checkpoint_dir: str   = checkpoint_dir
        self.verbose:        bool  = verbose

        self.counter:    int   = 0
        self.best_loss:  float = float("inf")
        self.early_stop: bool  = False
        self.best_epoch: int   = 0

    # ------------------------------------------------------------------
    def step(
        self,
        val_loss:  float,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   Dict[str, float],
    ) -> None:
        """
        Cập nhật trạng thái theo val_loss của epoch hiện tại.

        Nếu val_loss cải thiện  → reset counter, lưu best checkpoint.
        Nếu không cải thiện     → tăng counter; nếu vượt patience → early_stop = True.

        Args:
            val_loss  : Loss trên tập validation của epoch này.
            model     : Mô hình PyTorch hiện tại.
            optimizer : Optimizer hiện tại.
            epoch     : Chỉ số epoch (1-indexed).
            metrics   : Dict thống kê cần lưu vào checkpoint (val_loss, val_acc, ...).
        """
        if val_loss < self.best_loss - self.min_delta:
            # ── Cải thiện ────────────────────────────────────────────────
            improvement = self.best_loss - val_loss
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0

            # Lưu best checkpoint
            save_checkpoint(
                model          = model,
                optimizer      = optimizer,
                epoch          = epoch,
                metrics        = metrics,
                checkpoint_dir = self.checkpoint_dir,
                is_best        = True,
            )

            if self.verbose:
                print(
                    f"  [EarlyStopping] ✅ val_loss cải thiện "
                    f"{improvement:.6f} → {val_loss:.6f}  "
                    f"(best_epoch={epoch}). Đã lưu best checkpoint."
                )
        else:
            # ── Không cải thiện ──────────────────────────────────────────
            self.counter += 1
            if self.verbose:
                print(
                    f"  [EarlyStopping] ⚠️  Không cải thiện "
                    f"({self.counter}/{self.patience}). "
                    f"Best val_loss={self.best_loss:.6f} @ epoch {self.best_epoch}."
                )
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"  [EarlyStopping] 🛑 Dừng sớm sau {self.patience} epoch "
                    f"không cải thiện. Best epoch: {self.best_epoch}."
                )


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 — TRAIN / EVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    epoch:       int,
    total_epochs: int,
) -> float:
    """
    Chạy một epoch huấn luyện, trả về train_loss trung bình.

    - Dữ liệu từ train_loader có soft-labels (MixUp / CutMix) → CrossEntropyLoss
      nhận được cả hard và soft targets đều ổn (criterion đã có label_smoothing).
    - Tích hợp tqdm để hiển thị tiến trình theo batch.
    - Xử lý OOM bằng cách bỏ qua batch lỗi và gọi empty_cache().

    Args:
        model        : Mô hình ở chế độ train().
        loader       : DataLoader tập train (có MixUp/CutMix collate_fn).
        criterion    : Hàm loss (CrossEntropyLoss với label_smoothing).
        optimizer    : Optimizer hiện tại.
        device       : Thiết bị tính toán (cuda / cpu).
        epoch        : Epoch hiện tại (để hiển thị).
        total_epochs : Tổng số epoch (để hiển thị).

    Returns:
        train_loss_avg: Loss trung bình trên toàn bộ training set.
    """
    model.train()

    total_loss: float = 0.0
    n_batches:  int   = 0

    pbar = tqdm(
        loader,
        desc    = f"Epoch [{epoch:>3}/{total_epochs}] Train",
        unit    = "batch",
        dynamic_ncols = True,
        leave   = False,
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_loss  = loss.item()
            total_loss += batch_loss
            n_batches  += 1

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                # ── Xử lý OOM: bỏ batch, giải phóng VRAM ────────────────
                print(
                    f"\n  [OOM] Batch {batch_idx} bị bỏ qua do Out-Of-Memory. "
                    f"Đang giải phóng VRAM..."
                )
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise exc  # Re-raise nếu không phải OOM

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    split:     str = "Valid",
) -> Tuple[float, float]:
    """
    Đánh giá mô hình trên tập validation / test.

    Sử dụng hard labels (nhãn nguyên) từ valid_loader / test_loader.
    (Tập valid không dùng MixUp, labels là integer.)

    Args:
        model     : Mô hình ở chế độ eval().
        loader    : DataLoader tập valid hoặc test.
        criterion : Hàm loss.
        device    : Thiết bị tính toán.
        split     : Tên hiển thị ("Valid" / "Test").

    Returns:
        Tuple (avg_loss, accuracy):
            avg_loss : Loss trung bình trên toàn bộ split.
            accuracy : Độ chính xác Top-1 (0.0 → 1.0).
    """
    model.eval()

    total_loss:    float = 0.0
    total_correct: int   = 0
    total_samples: int   = 0
    n_batches:     int   = 0

    pbar = tqdm(
        loader,
        desc          = f"         {'':>14}{split} Eval",
        unit          = "batch",
        dynamic_ncols = True,
        leave         = False,
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)   # torch.int64

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item()
        n_batches     += 1

        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 — BUILDER OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def build_phase1_optimizer(
    model:   nn.Module,
    lr_head: float,
) -> torch.optim.AdamW:
    """
    Tạo AdamW Optimizer cho Phase 1 (head_only).

    Chỉ truyền tham số của Classification Head — backbone đã freeze,
    tuy nhiên lọc lại ``requires_grad=True`` để an toàn.

    Args:
        model   : Mô hình sau khi gọi switch_strategy(model, "head_only").
        lr_head : Learning rate cho Classification Head.

    Returns:
        torch.optim.AdamW sẵn sàng dùng.
    """
    head_params = [p for p in get_head_parameters(model) if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=1e-4)
    return optimizer


def build_phase2_optimizer(
    model:        nn.Module,
    lr_head:      float,
    lr_backbone:  float,
) -> torch.optim.AdamW:
    """
    Tạo AdamW Optimizer cho Phase 2 (partial) với 2 nhóm parameter:
      - Head     : lr_head     (LR cao — Head cần học nhanh hơn)
      - Backbone : lr_backbone (LR thấp — không phá vỡ features đã học)

    QUAN TRỌNG: Cần gọi hàm này SAU khi đã switch_strategy(model, "partial")
    để đảm bảo backbone params đã có requires_grad=True.

    Args:
        model       : Mô hình sau khi gọi switch_strategy(model, "partial").
        lr_head     : Learning rate cho Head.
        lr_backbone : Learning rate cho phần Backbone được mở khóa.

    Returns:
        torch.optim.AdamW với 2 param groups.
    """
    head_params     = [p for p in get_head_parameters(model)     if p.requires_grad]
    backbone_params = [p for p in get_backbone_parameters(model) if p.requires_grad]

    param_groups = [
        {"params": head_params,     "lr": lr_head,     "weight_decay": 1e-4},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 — TRAINING PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir:    str,
    model_key:   str,
    batch_size:  int   = 32,
    epochs:      int   = 30,
    patience:    int   = 7,
    lr_head:     float = 1e-3,
    lr_backbone: float = 1e-5,
    num_workers: int   = 0,
) -> List[Dict[str, float]]:
    """
    Pipeline huấn luyện đầy đủ: DataLoader → Model → Multi-phase Fine-tuning.

    Args:
        data_dir    : Thư mục gốc chứa train/ valid/ test/.
        model_key   : Tên model ngắn gọn: "resnet50" hoặc "vit".
        batch_size  : Số sample mỗi batch.
        epochs      : Số epoch tối đa.
        patience    : Số epoch EarlyStopping chờ không cải thiện.
        lr_head     : Learning rate cho Classification Head.
        lr_backbone : Learning rate cho Backbone (Phase 2).
        num_workers : Số worker DataLoader (0 trên Windows).

    Returns:
        history: List[Dict] — mỗi phần tử là dict thống kê 1 epoch:
            {"epoch", "phase", "train_loss", "val_loss", "val_acc", "elapsed_s"}
    """
    # ── Thiết lập device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" VEHICLE TYPE RECOGNITION — Training Pipeline")
    print("=" * 65)
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model       : {model_key}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Max epochs  : {epochs}")
    print(f"  Patience    : {patience}")
    print(f"  LR head     : {lr_head}")
    print(f"  LR backbone : {lr_backbone}")
    print("=" * 65)

    # ── Tạo DataLoaders ────────────────────────────────────────────────────
    print("\n[1/4] Khởi tạo DataLoaders ...")
    train_loader, valid_loader, _test_loader, class_names = create_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    print(f"  Số lớp        : {len(class_names)}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Valid batches : {len(valid_loader)}")
    print(f"  Classes       : {class_names}")

    # ── Xây dựng mô hình ───────────────────────────────────────────────────
    print("\n[2/4] Khởi tạo mô hình ...")
    full_model_name = MODEL_NAME_MAP[model_key]
    model = build_model(
        model_name  = full_model_name,
        num_classes = len(class_names),
        pretrained  = True,
        device      = device,
    )
    print(f"  Architecture  : {full_model_name}")

    # ── Khởi tạo Loss ──────────────────────────────────────────────────────
    # label_smoothing=0.1 phù hợp với soft-labels từ MixUp/CutMix ở train
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # ── Tạo thư mục output ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Khởi tạo EarlyStopping ─────────────────────────────────────────────
    stopper = EarlyStopping(
        patience       = patience,
        checkpoint_dir = CHECKPOINT_DIR,
        verbose        = True,
    )

    # ── Biến trạng thái Phase ──────────────────────────────────────────────
    current_phase: str                        = ""
    optimizer:     Optional[torch.optim.AdamW] = None
    history:       List[Dict[str, float]]     = []

    print("\n[3/4] Bắt đầu huấn luyện ...\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # ── Xác định phase & (re-)khởi tạo Optimizer nếu cần ──────────────
        if epoch <= PHASE1_EPOCHS:
            desired_phase = "head_only"
        else:
            desired_phase = "partial"

        if desired_phase != current_phase:
            current_phase = desired_phase
            switch_strategy(model, current_phase)

            if current_phase == "head_only":
                optimizer = build_phase1_optimizer(model, lr_head)
                print(f"\n  ── Phase 1 (head_only): Optimizer được tạo với lr_head={lr_head} ──")
            else:
                # BẮT BUỘC tạo lại Optimizer khi chuyển sang partial
                optimizer = build_phase2_optimizer(model, lr_head, lr_backbone)
                print(
                    f"\n  ── Phase 2 (partial): Optimizer được tạo lại "
                    f"(lr_head={lr_head}, lr_backbone={lr_backbone}) ──"
                )
            model_summary(model)

        assert optimizer is not None, "Optimizer chưa được khởi tạo — lỗi logic!"

        # ── Train 1 epoch ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model        = model,
            loader       = train_loader,
            criterion    = criterion,
            optimizer    = optimizer,
            device       = device,
            epoch        = epoch,
            total_epochs = epochs,
        )

        # ── Đánh giá trên Valid ────────────────────────────────────────────
        val_loss, val_acc = evaluate(
            model     = model,
            loader    = valid_loader,
            criterion = criterion,
            device    = device,
            split     = "Valid",
        )

        elapsed = time.perf_counter() - epoch_start

        # ── In kết quả epoch ───────────────────────────────────────────────
        phase_tag = "P1-head" if current_phase == "head_only" else "P2-part"
        print(
            f"  Epoch [{epoch:>3}/{epochs}] [{phase_tag}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc*100:.2f}%  "
            f"({elapsed:.1f}s)"
        )

        # ── Lưu lịch sử ────────────────────────────────────────────────────
        history.append({
            "epoch":      epoch,
            "phase":      current_phase,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "elapsed_s":  round(elapsed,    2),
        })

        # ── EarlyStopping step ─────────────────────────────────────────────
        metrics = {"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss}
        stopper.step(val_loss, model, optimizer, epoch, metrics)

        if stopper.early_stop:
            print(f"\n  Dừng huấn luyện sớm tại epoch {epoch}.")
            break

    # ── Lưu lịch sử JSON ───────────────────────────────────────────────────
    print("\n[4/4] Lưu lịch sử huấn luyện ...")
    history_path = os.path.join(OUTPUT_DIR, f"history_{model_key}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu: {history_path}")

    # ── Tổng kết ───────────────────────────────────────────────────────────
    best_epoch = stopper.best_epoch
    best_loss  = stopper.best_loss
    best_record = next(
        (r for r in history if r["epoch"] == best_epoch), {}
    )
    best_acc = best_record.get("val_acc", 0.0)

    print("\n" + "=" * 65)
    print(" KẾT QUẢ HUẤN LUYỆN")
    print("=" * 65)
    print(f"  Tổng số epoch đã chạy : {len(history)}")
    print(f"  Best epoch            : {best_epoch}")
    print(f"  Best val_loss         : {best_loss:.6f}")
    print(f"  Best val_acc          : {best_acc*100:.2f}%")
    print(f"  Best checkpoint       : {CHECKPOINT_DIR}/{full_model_name}_best.pth")
    print(f"  Lịch sử               : {history_path}")
    print("=" * 65 + "\n")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5 — CLI / __main__
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Phân tích tham số dòng lệnh (CLI).

    Returns:
        argparse.Namespace với tất cả tham số huấn luyện.
    """
    parser = argparse.ArgumentParser(
        prog        = "train.py",
        description = "Huấn luyện mô hình Vehicle Type Recognition (ResNet-50 / ViT-B/16).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dữ liệu ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir",
        type    = str,
        default = "data/augmented",
        help    = "Thư mục gốc chứa train/ valid/ test/ (output của data_prep.py).",
    )

    # ── Mô hình ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type    = str,
        default = "resnet50",
        choices = list(MODEL_NAME_MAP.keys()),
        help    = "Kiến trúc mô hình: 'resnet50' hoặc 'vit' (ViT-B/16).",
    )

    # ── Huấn luyện ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--batch_size",
        type    = int,
        default = 32,
        help    = "Kích thước batch (giảm nếu gặp OOM).",
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 30,
        help    = "Số epoch tối đa.",
    )
    parser.add_argument(
        "--patience",
        type    = int,
        default = 7,
        help    = "Số epoch EarlyStopping chờ không cải thiện val_loss.",
    )

    # ── Learning Rate ───────────────────────────────────────────────────────
    parser.add_argument(
        "--lr_head",
        type    = float,
        default = 1e-3,
        help    = "Learning rate cho Classification Head (cả 2 phase).",
    )
    parser.add_argument(
        "--lr_backbone",
        type    = float,
        default = 1e-5,
        help    = "Learning rate cho Backbone (chỉ áp dụng ở Phase 2 - partial).",
    )

    # ── DataLoader ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--num_workers",
        type    = int,
        default = 0,
        help    = "Số worker DataLoader. Dùng 0 trên Windows để tránh lỗi spawn.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Sửa encoding stdout cho Windows (hỗ trợ in ký tự Unicode / tiếng Việt)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    # Validate model key
    if args.model not in MODEL_NAME_MAP:
        print(f"[LỖI] --model '{args.model}' không hợp lệ. Chọn: {list(MODEL_NAME_MAP.keys())}")
        sys.exit(1)

    # Validate data_dir
    if not os.path.isdir(args.data_dir):
        print(f"[LỖI] --data_dir '{args.data_dir}' không tồn tại.")
        print("       Hãy chạy: python src/data_prep.py --all")
        sys.exit(1)

    # ── Bắt đầu huấn luyện ─────────────────────────────────────────────────
    train(
        data_dir    = args.data_dir,
        model_key   = args.model,
        batch_size  = args.batch_size,
        epochs      = args.epochs,
        patience    = args.patience,
        lr_head     = args.lr_head,
        lr_backbone = args.lr_backbone,
        num_workers = args.num_workers,
    )
