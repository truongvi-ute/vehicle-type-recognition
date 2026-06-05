"""
src/model.py
============
Quản lý kiến trúc mô hình cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Hỗ trợ 2 kiến trúc pre-trained:
  - ResNet-50  : CNN cổ điển với Skip Connections, ~24.5M tham số.
  - ViT-B/16   : Vision Transformer, ~86M tham số, Self-Attention toàn cục.

Chiến lược Fine-tuning đa giai đoạn:
  Phase 1 — "head_only" : Chỉ train Classification Head (Head mới toanh).
  Phase 2 — "partial"   : Mở thêm 1/3 block cuối Backbone (đặc trưng xe cộ).
  Phase 3 — "full"      : Mở toàn bộ (khi cần tinh chỉnh sâu nhất).

Thư viện cần thiết:
    pip install torch torchvision

Cách dùng:
    from model import build_model, switch_strategy, model_summary

    model = build_model("resnet50", num_classes=10)
    switch_strategy(model, "head_only")
    model_summary(model)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
)

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

#: Tên các kiến trúc được hỗ trợ — dùng để kiểm tra đầu vào.
SUPPORTED_MODELS: List[str] = ["resnet50", "vit_base_patch16_224"]

#: Chiến lược freeze/unfreeze hợp lệ.
SUPPORTED_STRATEGIES: List[str] = ["head_only", "partial", "full"]

#: Số lượng encoder block của ViT-B/16 (cố định theo kiến trúc gốc).
VIT_NUM_ENCODER_LAYERS: int = 12

#: 1/3 số block ViT sẽ được mở khóa ở chiến lược "partial".
VIT_PARTIAL_OPEN_LAYERS: int = 4   # floor(12 / 3) = 4 block cuối

#: Prefix tên key của từng encoder block trong ViT (từ torchvision).
VIT_ENCODER_LAYER_PREFIX: str = "encoder_layer_"

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 — XÂY DỰNG MÔ HÌNH
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name:  str,
    num_classes: int  = 10,
    pretrained:  bool = True,
    device:      Optional[torch.device] = None,
) -> nn.Module:
    """
    Khởi tạo mô hình Deep Learning với Classification Head tùy chỉnh.

    Quy trình:
        1. Nạp kiến trúc + trọng số ImageNet pre-trained (nếu pretrained=True).
        2. Thay thế Classification Head bằng Linear layer khớp num_classes.
        3. Gắn metadata (._arch, ._head_name) vào đối tượng model để
           switch_strategy() nhận diện kiến trúc.
        4. Chuyển model lên device đã chỉ định.

    Head replacement:
        - ResNet-50 : ``model.fc``           Linear(2048  → num_classes)
        - ViT-B/16  : ``model.heads.head``   Linear(768   → num_classes)

    Args:
        model_name  : Tên kiến trúc. Phải thuộc SUPPORTED_MODELS.
        num_classes : Số lớp đầu ra (mặc định = 10 cho Vehicle-10).
        pretrained  : True → nạp ImageNet weights; False → random init.
        device      : Device để chạy mô hình. None → tự động chọn CUDA / CPU.

    Returns:
        nn.Module đã sẵn sàng dùng (Classification Head = Linear mới,
        toàn bộ backbone vẫn đang freeze theo mặc định của torchvision).

    Raises:
        ValueError: Nếu model_name không thuộc SUPPORTED_MODELS.

    Example:
        >>> model = build_model("resnet50", num_classes=10)
        >>> model = build_model("vit_base_patch16_224", pretrained=False)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_name='{model_name}' không được hỗ trợ.\n"
            f"Chọn một trong: {SUPPORTED_MODELS}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Nạp kiến trúc + pretrained weights ──────────────────────────────
    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model: nn.Module = models.resnet50(weights=weights)

        # Thay thế FC head: Linear(2048, 1000) → Linear(2048, num_classes)
        in_features = model.fc.in_features          # 2048
        model.fc    = nn.Linear(in_features, num_classes)

        # Metadata cho switch_strategy()
        model._arch      = "resnet50"               # type: ignore[attr-defined]
        model._head_name = "fc"                     # type: ignore[attr-defined]

    elif model_name == "vit_base_patch16_224":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)

        # Thay thế head: Linear(768, 1000) → Linear(768, num_classes)
        in_features       = model.heads.head.in_features   # 768
        model.heads.head  = nn.Linear(in_features, num_classes)

        # Metadata cho switch_strategy()
        model._arch      = "vit_base_patch16_224"   # type: ignore[attr-defined]
        model._head_name = "heads"                  # type: ignore[attr-defined]

    model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 — CHIẾN LƯỢC FREEZE / UNFREEZE
# ─────────────────────────────────────────────────────────────────────────────

def _freeze_all(model: nn.Module) -> None:
    """Đóng băng toàn bộ tham số (requires_grad = False)."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    """Mở khóa toàn bộ tham số của 1 submodule."""
    for param in module.parameters():
        param.requires_grad = True


def switch_strategy(
    model:    nn.Module,
    strategy: Literal["head_only", "partial", "full"],
) -> None:
    """
    Điều chỉnh trạng thái đóng băng / mở khóa tham số của mô hình để
    phục vụ chiến lược Fine-tuning đa giai đoạn.

    Chiến lược:
        "head_only"
            Đóng băng toàn bộ Backbone.
            CHỈ mở Classification Head.
            → Dùng ở Phase 1: Head học nhanh, backbone không bị phá vỡ.

        "partial"
            Đóng băng phần lớn Backbone.
            Mở Classification Head + 1/3 block cuối cùng của Backbone.
                ResNet-50  : Mở ``layer4`` + ``avgpool`` + ``fc``
                ViT-B/16   : Mở 4 block encoder cuối + LayerNorm + ``heads``
            → Dùng ở Phase 2: Tinh chỉnh đặc trưng cấp cao của xe cộ.

        "full"
            Mở khóa toàn bộ mạng (requires_grad = True).
            → Dùng khi muốn Fine-tune sâu nhất (cần LR rất nhỏ).

    Args:
        model    : Mô hình khởi tạo bởi build_model().
        strategy : Một trong "head_only", "partial", "full".

    Raises:
        ValueError : Nếu strategy không hợp lệ.
        AttributeError : Nếu model thiếu thuộc tính ``._arch``
                         (không được tạo bởi build_model()).

    Example:
        >>> model = build_model("resnet50")
        >>> switch_strategy(model, "head_only")   # Phase 1
        >>> switch_strategy(model, "partial")     # Phase 2
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"strategy='{strategy}' không hợp lệ.\n"
            f"Chọn một trong: {SUPPORTED_STRATEGIES}"
        )

    arch = getattr(model, "_arch", None)
    if arch is None:
        raise AttributeError(
            "model thiếu thuộc tính ._arch. "
            "Hãy khởi tạo mô hình bằng build_model()."
        )

    # ── "full": mở khóa tất cả ───────────────────────────────────────────
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    # ── Bước chung: đóng băng toàn bộ trước ─────────────────────────────
    _freeze_all(model)

    # ── ResNet-50 ─────────────────────────────────────────────────────────
    if arch == "resnet50":
        # Luôn mở Head
        _unfreeze_module(model.fc)

        if strategy == "partial":
            # Mở 1/3 cuối backbone:
            #   ResNet-50 có 4 nhóm layer: layer1, layer2, layer3, layer4
            #   1/3 ≈ layer4 (Block cuối cùng, học đặc trưng cao cấp nhất)
            _unfreeze_module(model.layer4)   # 3 Bottleneck blocks
            _unfreeze_module(model.avgpool)  # Global Avg Pool không có params
            # model.fc đã được mở ở trên

    # ── Vision Transformer (ViT-B/16) ────────────────────────────────────
    elif arch == "vit_base_patch16_224":
        # Luôn mở Classification Head
        _unfreeze_module(model.heads)

        if strategy == "partial":
            # ViT-B/16 có 12 encoder blocks.
            # torchvision đặt key dạng: "encoder_layer_0" ... "encoder_layer_11"
            # Sequential.__getitem__ chỉ nhận int index, phải dùng ._modules
            # để tra cứu theo tên string.
            n_layers  = len(model.encoder.layers)
            n_to_open = min(VIT_PARTIAL_OPEN_LAYERS, n_layers)
            start_idx = n_layers - n_to_open

            for idx in range(start_idx, n_layers):
                block_key    = f"{VIT_ENCODER_LAYER_PREFIX}{idx}"
                block_module = model.encoder.layers._modules.get(block_key)
                if block_module is not None:
                    _unfreeze_module(block_module)

            # Mở thêm Layer Norm cuối encoder (encoder.ln)
            _unfreeze_module(model.encoder.ln)



# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 — THỐNG KÊ THAM SỐ
# ─────────────────────────────────────────────────────────────────────────────

def model_summary(model: nn.Module, print_layers: bool = False) -> Dict[str, int]:
    """
    In ra và trả về thống kê số lượng tham số của mô hình.

    Metrics xuất ra:
        - Total params     : Tổng số tham số (trainable + frozen).
        - Trainable params : Số tham số đang được cập nhật (requires_grad=True).
        - Frozen params    : Số tham số bị đóng băng (requires_grad=False).
        - Trainable %      : Tỷ lệ tham số được train.

    Args:
        model       : Mô hình PyTorch bất kỳ.
        print_layers: True → in thêm từng layer với trạng thái requires_grad.

    Returns:
        Dict với các key: "total", "trainable", "frozen".

    Example:
        >>> stats = model_summary(model)
        >>> print(stats["trainable"])
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    pct              = (trainable_params / max(total_params, 1)) * 100

    arch  = getattr(model, "_arch", type(model).__name__)
    strat = getattr(model, "_current_strategy", "unknown")

    print("─" * 55)
    print(f" Model Summary  :  {arch}")
    print(f" Strategy       :  {strat}")
    print("─" * 55)
    print(f" Total params   :  {total_params:>14,}")
    print(f" Trainable      :  {trainable_params:>14,}  ({pct:.1f} %)")
    print(f" Frozen         :  {frozen_params:>14,}  ({100-pct:.1f} %)")
    print("─" * 55)

    if print_layers:
        print("\n Layer-level requires_grad:")
        for name, module in model.named_children():
            n_params     = sum(p.numel() for p in module.parameters())
            n_trainable  = sum(p.numel() for p in module.parameters()
                               if p.requires_grad)
            status = "OPEN  " if n_trainable > 0 else "FROZEN"
            print(f"   [{status}] {name:<25} "
                  f"{n_params:>10,} params  ({n_trainable:>10,} trainable)")
        print()

    return {
        "total":     total_params,
        "trainable": trainable_params,
        "frozen":    frozen_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 — LƯU & NẠP CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    epoch:          int,
    metrics:        Dict[str, float],
    checkpoint_dir: str,
    is_best:        bool = False,
) -> str:
    """
    Lưu trạng thái huấn luyện (state_dict, optimizer, metrics) xuống disk.

    Tên file checkpoint:
        <arch>_epoch<N>_<timestamp>.pth   — checkpoint thường xuyên
        <arch>_best.pth                   — checkpoint tốt nhất (is_best=True)

    Nội dung checkpoint (dict):
        {
            "epoch"      : <int>,
            "arch"       : <str>,
            "state_dict" : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "metrics"    : {"val_loss": ..., "val_acc": ...},
            "saved_at"   : <ISO timestamp>,
        }

    Args:
        model          : Mô hình cần lưu.
        optimizer      : Optimizer hiện tại.
        epoch          : Epoch hiện tại.
        metrics        : Dict các chỉ số (vd: {"val_loss": 0.12, "val_acc": 0.95}).
        checkpoint_dir : Thư mục lưu checkpoint.
        is_best        : True → lưu thêm file ``<arch>_best.pth``.

    Returns:
        Đường dẫn tuyệt đối của file checkpoint vừa lưu.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    arch      = getattr(model, "_arch", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "epoch":      epoch,
        "arch":       arch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
        "saved_at":   datetime.now().isoformat(),
    }

    # ── Lưu checkpoint thường ────────────────────────────────────────────
    ckpt_name = f"{arch}_epoch{epoch:03d}_{timestamp}.pth"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save(payload, ckpt_path)

    # ── Lưu file "_best" nếu là checkpoint tốt nhất ──────────────────────
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{arch}_best.pth")
        torch.save(payload, best_path)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model:           nn.Module,
    optimizer:       Optional[torch.optim.Optimizer] = None,
    device:          Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Nạp checkpoint để tiếp tục huấn luyện (resume training).

    Nạp state_dict vào model và optimizer (nếu truyền vào).
    Trả về toàn bộ payload để lấy epoch/metrics.

    Args:
        checkpoint_path : Đường dẫn file .pth.
        model           : Mô hình cần restore weights.
        optimizer       : Optimizer cần restore state (None → bỏ qua).
        device          : Device để map trọng số. None → tự động CPU/CUDA.

    Returns:
        Dict checkpoint gốc (có thể lấy epoch, metrics,...).

    Raises:
        FileNotFoundError : Nếu file checkpoint không tồn tại.

    Example:
        >>> payload = load_checkpoint("models/resnet50_best.pth", model, optimizer)
        >>> start_epoch = payload["epoch"] + 1
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    print(f"[load_checkpoint] Đã nạp: {checkpoint_path}")
    print(f"  Arch   : {payload.get('arch', '?')}")
    print(f"  Epoch  : {payload.get('epoch', '?')}")
    print(f"  Metrics: {payload.get('metrics', {})}")

    return payload


def load_for_inference(
    checkpoint_path: str,
    num_classes:     int,
    device:          Optional[torch.device] = None,
) -> nn.Module:
    """
    Nạp mô hình từ checkpoint để chỉ dùng cho Inference (không train).

    Khác với load_checkpoint(): hàm này tự xây dựng lại mô hình từ tên
    kiến trúc lưu trong checkpoint, sau đó nạp state_dict.
    Model trả về ở chế độ eval() và requires_grad = False.

    Args:
        checkpoint_path : Đường dẫn file .pth tạo bởi save_checkpoint().
        num_classes     : Số lớp (phải khớp với lúc train).
        device          : Device để chạy inference.

    Returns:
        nn.Module ở chế độ eval, sẵn sàng nhận input.

    Raises:
        FileNotFoundError : Nếu file không tồn tại.
        ValueError        : Nếu kiến trúc lưu trong checkpoint không hỗ trợ.

    Example:
        >>> model = load_for_inference("models/resnet50_best.pth", num_classes=10)
        >>> with torch.no_grad():
        ...     logits = model(input_tensor)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device)
    arch    = payload.get("arch", "")

    # Xây dựng lại mô hình theo kiến trúc đã lưu
    model = build_model(
        model_name  = arch,
        num_classes = num_classes,
        pretrained  = False,   # Không cần ImageNet weights — sẽ load từ ckpt
        device      = device,
    )
    model.load_state_dict(payload["state_dict"])

    # Khóa toàn bộ tham số (inference chỉ cần forward pass)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5 — TIỆN ÍCH BỔ SUNG
# ─────────────────────────────────────────────────────────────────────────────

def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """
    Đếm nhanh số tham số (total, trainable) — không in ra màn hình.

    Args:
        model: Mô hình PyTorch.

    Returns:
        Tuple (total_params, trainable_params).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_head_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Trả về danh sách tham số của Classification Head.
    Tiện dùng để truyền riêng cho optimizer với LR cao hơn backbone.

    Args:
        model: Mô hình tạo bởi build_model().

    Returns:
        List các nn.Parameter thuộc Head.
    """
    head_name = getattr(model, "_head_name", None)
    if head_name is None:
        raise AttributeError(
            "model thiếu ._head_name. Dùng build_model() để khởi tạo."
        )
    head_module = getattr(model, head_name)
    return list(head_module.parameters())


def get_backbone_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Trả về danh sách tham số của Backbone (không bao gồm Head).
    Tiện dùng để truyền cho optimizer với LR nhỏ hơn.

    Args:
        model: Mô hình tạo bởi build_model().

    Returns:
        List các nn.Parameter không thuộc Head.
    """
    head_name  = getattr(model, "_head_name", None)
    head_ids   = {
        id(p)
        for p in (getattr(model, head_name).parameters() if head_name else [])
    }
    return [p for p in model.parameters() if id(p) not in head_ids]


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST  (python src/model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"PyTorch  : {torch.__version__}")

    for arch in SUPPORTED_MODELS:
        print(f"\n{'═' * 55}")
        print(f" Kiến trúc: {arch}")
        print(f"{'═' * 55}")

        # ── Khởi tạo mô hình ─────────────────────────────────────────────
        print(f"\n[1] build_model(pretrained=True) ...")
        model = build_model(arch, num_classes=10, pretrained=True, device=device)

        # ── Kiểm tra 3 chiến lược freeze ─────────────────────────────────
        for strat in SUPPORTED_STRATEGIES:
            switch_strategy(model, strat)
            model._current_strategy = strat      # type: ignore[attr-defined]
            total, trainable = count_trainable_params(model)
            pct = trainable / max(total, 1) * 100
            print(f"\n[strategy = '{strat}']")
            model_summary(model, print_layers=True)

        # ── Test forward pass ─────────────────────────────────────────────
        print(f"\n[2] Forward pass với fake batch (B=2, 3×224×224) ...")
        switch_strategy(model, "full")
        model.eval()
        with torch.no_grad():
            x      = torch.randn(2, 3, 224, 224, device=device)
            logits = model(x)
        print(f"   Input  shape : {tuple(x.shape)}")
        print(f"   Output shape : {tuple(logits.shape)}")   # (2, 10)
        assert logits.shape == (2, 10), \
            f"Output shape sai! Mong đợi (2, 10), nhận {tuple(logits.shape)}"
        print("   ✅ Forward pass PASSED")

        # ── Test get_head_parameters ──────────────────────────────────────
        switch_strategy(model, "head_only")
        head_params    = get_head_parameters(model)
        backbone_params = get_backbone_parameters(model)
        print(f"\n[3] Parameter groups:")
        print(f"   Head params     : {sum(p.numel() for p in head_params):,}")
        print(f"   Backbone params : {sum(p.numel() for p in backbone_params):,}")

    print(f"\n{'═' * 55}")
    print(" ✅ Tất cả smoke test PASSED — model.py hoạt động đúng.")
    print(f"{'═' * 55}\n")
