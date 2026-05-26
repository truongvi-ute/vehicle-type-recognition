"""
src/model.py
============
Khởi tạo và quản lý 3 kiến trúc mô hình Transfer Learning cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Mô hình hỗ trợ:
  ┌──────────────────┬──────────────────────────────────┬────────────┐
  │ Tên              │ Kiến trúc nổi bật                │ Vai trò    │
  ├──────────────────┼──────────────────────────────────┼────────────┤
  │ resnet50         │ Skip Connections (Residual Block) │ Baseline   │
  │ mobilenet_v3     │ Depthwise Separable Conv, nhẹ    │ Thực tế    │
  │ efficientnet_b0  │ Compound Scaling đa chiều         │ Cân bằng   │
  └──────────────────┴──────────────────────────────────┴────────────┘

Gồm 4 phần chính:
  1. Model Builder   — thay đổi classification head, chiến lược freeze
  2. Freeze Strategy — kiểm soát các tầng được huấn luyện
  3. Model Info      — đếm tham số, tóm tắt kiến trúc
  4. Save / Load     — lưu và khôi phục checkpoint đầy đủ

Thư viện cần thiết:
  pip install torch torchvision
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights,
)

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["resnet50", "mobilenet_v3", "efficientnet_b0"]

# Tên file checkpoint mặc định cho từng mô hình
CHECKPOINT_NAMES = {
    "resnet50":         "resnet50_best.pth",
    "mobilenet_v3":     "mobilenet_v3_best.pth",
    "efficientnet_b0":  "efficientnet_b0_best.pth",
}

# Dropout rate cho classification head
DEFAULT_DROPOUT = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_classifier_head(
    in_features: int,
    num_classes: int,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Sequential:
    """
    Tạo classification head tuỳ chỉnh:
        Linear(in_features → 512) → BN → ReLU → Dropout → Linear(512 → num_classes)

    Lý do thêm tầng ẩn 512:
      Tăng khả năng học đặc trưng trung gian đặc thù cho dataset phương tiện,
      vượt trội hơn so với chỉ dùng 1 Linear layer đơn giản.
    Lý do dùng BatchNorm trước Dropout:
      Ổn định phân phối đặc trưng từ backbone, giúp Dropout hiệu quả hơn.
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),
    )


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Module:
    """
    ResNet-50 với classification head tuỳ chỉnh.

    Kiến trúc:
      [Conv1] → [Layer1–4 (Residual Blocks)] → [AdaptiveAvgPool] → [FC Head]
      Backbone output: 2048 features
      Head: 2048 → 512 → num_classes

    Skip Connections giải quyết vấn đề gradient vanishing,
    cho phép huấn luyện mạng rất sâu (50 tầng) hiệu quả.

    Args:
        num_classes : Số lớp phân loại (5-7 lớp phương tiện)
        pretrained  : True = dùng trọng số ImageNet, False = khởi tạo ngẫu nhiên
        dropout     : Tỷ lệ dropout trong classification head

    Returns:
        nn.Module đã thay thế fc layer
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    in_features  = model.fc.in_features        # 2048
    model.fc     = _build_classifier_head(in_features, num_classes, dropout)

    model._model_name   = "resnet50"
    model._num_classes  = num_classes
    model._in_features  = in_features
    return model


def build_mobilenet_v3(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Module:
    """
    MobileNet-V3-Small với classification head tuỳ chỉnh.

    Kiến trúc:
      [Inverted Residual Blocks + SE] → [AdaptiveAvgPool] → [FC Head]
      Backbone output: 576 features
      Head: 576 → 512 → num_classes

    Depthwise Separable Convolution giảm ~8-9x FLOPs so với Conv thông thường.
    Squeeze-and-Excitation (SE) tăng khả năng tập trung vào kênh quan trọng.

    Lý do dùng Small thay vì Large:
      Dataset 7.5-14K ảnh không đủ lớn để phân biệt rõ lợi ích của Large.
      Small nhanh hơn ~2x, phù hợp thực nghiệm nhanh và so sánh.

    Args:
        num_classes : Số lớp phân loại
        pretrained  : True = dùng trọng số ImageNet
        dropout     : Tỷ lệ dropout

    Returns:
        nn.Module đã thay thế classifier
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v3_small(weights=weights)

    # MobileNetV3: model.classifier = Sequential([Linear, Hardswish, Dropout, Linear])
    in_features       = model.classifier[0].in_features   # 576
    model.classifier  = _build_classifier_head(in_features, num_classes, dropout)

    model._model_name  = "mobilenet_v3"
    model._num_classes = num_classes
    model._in_features = in_features
    return model


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = DEFAULT_DROPOUT,
) -> nn.Module:
    """
    EfficientNet-B0 với classification head tuỳ chỉnh.

    Kiến trúc:
      [MBConv Blocks + SE] → [AdaptiveAvgPool] → [FC Head]
      Backbone output: 1280 features
      Head: 1280 → 512 → num_classes

    Compound Scaling: mở rộng đồng thời width, depth, resolution theo hệ số φ.
    B0 là mô hình cơ sở (baseline) của dòng EfficientNet.

    Args:
        num_classes : Số lớp phân loại
        pretrained  : True = dùng trọng số ImageNet
        dropout     : Tỷ lệ dropout

    Returns:
        nn.Module đã thay thế classifier
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # EfficientNet: model.classifier = Sequential([Dropout, Linear])
    in_features      = model.classifier[1].in_features    # 1280
    model.classifier = _build_classifier_head(in_features, num_classes, dropout)

    model._model_name  = "efficientnet_b0"
    model._num_classes = num_classes
    model._in_features = in_features
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION — điểm truy cập duy nhất
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = DEFAULT_DROPOUT,
    freeze_strategy: str = "head_only",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function: tạo mô hình theo tên, áp dụng freeze strategy, chuyển lên device.

    Args:
        model_name      : 'resnet50' | 'mobilenet_v3' | 'efficientnet_b0'
        num_classes     : Số lớp phân loại đầu ra
        pretrained      : Dùng trọng số ImageNet pre-trained
        dropout         : Tỷ lệ dropout trong head
        freeze_strategy : Chiến lược đóng băng tầng (xem PHẦN 2)
                          'head_only'   — chỉ train classification head (Phase 1)
                          'partial'     — train head + 1/3 cuối backbone (Phase 2)
                          'full'        — train toàn bộ mạng (Phase 3 / fine-tune)
                          'none'        — không train gì (chỉ inference)
        device          : torch.device (mặc định: tự phát hiện CPU/CUDA)

    Returns:
        nn.Module đã được cấu hình và chuyển lên device

    Ví dụ:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = build_model('resnet50', num_classes=5, device=device)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_name '{model_name}' không hợp lệ.\n"
            f"Hỗ trợ: {SUPPORTED_MODELS}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Xây dựng mô hình ──────────────────────────────────────────────────
    builders = {
        "resnet50":        build_resnet50,
        "mobilenet_v3":    build_mobilenet_v3,
        "efficientnet_b0": build_efficientnet_b0,
    }
    model = builders[model_name](num_classes, pretrained, dropout)

    # ── Áp dụng freeze strategy ───────────────────────────────────────────
    apply_freeze_strategy(model, freeze_strategy)

    # ── Chuyển lên device ─────────────────────────────────────────────────
    model = model.to(device)
    model._device = device

    # ── In tóm tắt ────────────────────────────────────────────────────────
    total, trainable = count_parameters(model)
    frozen = total - trainable
    print(f"\n{'─'*50}")
    print(f"  Mô hình      : {model_name}")
    print(f"  Số lớp       : {num_classes}")
    print(f"  Pre-trained  : {pretrained}")
    print(f"  Freeze       : {freeze_strategy}")
    print(f"  Tổng tham số : {total:,}")
    print(f"  Trainable    : {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"  Frozen       : {frozen:,}  ({frozen/total*100:.1f}%)")
    print(f"  Device       : {device}")
    print(f"{'─'*50}\n")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – FREEZE STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

def _get_backbone_layers(model: nn.Module) -> List[nn.Module]:
    """
    Trả về danh sách các module backbone (không kể classification head)
    theo thứ tự từ đầu vào đến đầu ra.
    """
    name = getattr(model, "_model_name", "unknown")

    if name == "resnet50":
        # ResNet: features = conv1, bn1, relu, maxpool, layer1-4, avgpool
        return [
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        ]
    elif name == "mobilenet_v3":
        # MobileNetV3: model.features chứa tất cả block backbone
        return list(model.features.children())
    elif name == "efficientnet_b0":
        # EfficientNet: model.features chứa tất cả block backbone
        return list(model.features.children())
    else:
        # Fallback: lấy tất cả trừ classifier / fc
        head_names = {"fc", "classifier"}
        return [m for n, m in model.named_children() if n not in head_names]


def _get_head(model: nn.Module) -> nn.Module:
    """Trả về classification head của mô hình."""
    name = getattr(model, "_model_name", "unknown")
    if name == "resnet50":
        return model.fc
    else:  # mobilenet_v3, efficientnet_b0
        return model.classifier


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Bật/tắt gradient cho toàn bộ tham số của module."""
    for param in module.parameters():
        param.requires_grad = requires_grad


def apply_freeze_strategy(model: nn.Module, strategy: str) -> None:
    """
    Áp dụng chiến lược đóng băng (freeze) tầng mạng.

    Quy trình huấn luyện 3 Phase được khuyến nghị:
    ┌──────────┬──────────────┬───────────────────────────────────────┐
    │ Phase    │ Strategy     │ Mục đích                              │
    ├──────────┼──────────────┼───────────────────────────────────────┤
    │ Phase 1  │ head_only    │ Warm-up: train nhanh head mới,        │
    │ (5-10ep) │              │ backbone giữ nguyên ImageNet weights. │
    ├──────────┼──────────────┼───────────────────────────────────────┤
    │ Phase 2  │ partial      │ Fine-tune: mở dần 1/3 cuối backbone   │
    │ (10-20ep)│              │ để học đặc trưng đặc thù phương tiện. │
    ├──────────┼──────────────┼───────────────────────────────────────┤
    │ Phase 3  │ full         │ Fine-tune toàn bộ với LR rất nhỏ      │
    │ (5-10ep) │              │ để tinh chỉnh sâu (optional).         │
    └──────────┴──────────────┴───────────────────────────────────────┘

    Args:
        model    : nn.Module (đã được build_model tạo)
        strategy : 'head_only' | 'partial' | 'full' | 'none'
    """
    valid_strategies = {"head_only", "partial", "full", "none"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"freeze_strategy '{strategy}' không hợp lệ.\n"
            f"Hỗ trợ: {sorted(valid_strategies)}"
        )

    backbone_layers = _get_backbone_layers(model)
    head            = _get_head(model)
    n_layers        = len(backbone_layers)

    if strategy == "none":
        # Đóng băng tất cả — chỉ dùng inference
        _set_requires_grad(model, False)

    elif strategy == "head_only":
        # Đóng băng backbone, train head
        _set_requires_grad(model, False)
        _set_requires_grad(head, True)

    elif strategy == "partial":
        # Đóng băng 2/3 đầu backbone, mở 1/3 cuối + head
        _set_requires_grad(model, False)
        cutoff = n_layers * 2 // 3          # Chỉ mở từ layer này về sau
        for layer in backbone_layers[cutoff:]:
            _set_requires_grad(layer, True)
        _set_requires_grad(head, True)

    elif strategy == "full":
        # Mở toàn bộ
        _set_requires_grad(model, True)


def switch_strategy(model: nn.Module, new_strategy: str) -> None:
    """
    Chuyển chiến lược freeze trong khi đang huấn luyện (không cần tạo lại model).

    Ví dụ workflow:
        model = build_model('resnet50', num_classes=5, freeze_strategy='head_only')
        # ... train 10 epochs ...
        switch_strategy(model, 'partial')
        # ... train thêm 10 epochs với LR nhỏ hơn ...
        switch_strategy(model, 'full')
    """
    apply_freeze_strategy(model, new_strategy)
    total, trainable = count_parameters(model)
    print(f"  [switch_strategy] → '{new_strategy}' | "
          f"Trainable: {trainable:,}/{total:,} "
          f"({trainable/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 – MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Đếm tổng số tham số và số tham số được train.

    Returns:
        (total_params, trainable_params)
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> None:
    """
    In tóm tắt mô hình: tên, tham số, kích thước output từng layer chính.

    Args:
        model      : nn.Module
        input_size : Kích thước tensor đầu vào (B, C, H, W)
    """
    name = getattr(model, "_model_name", "unknown")
    num_classes = getattr(model, "_num_classes", "?")
    total, trainable = count_parameters(model)
    device = next(model.parameters()).device

    print(f"\n{'═'*55}")
    print(f"  MODEL SUMMARY — {name.upper()}")
    print(f"{'═'*55}")
    print(f"  Số lớp phân loại : {num_classes}")
    print(f"  Tổng tham số     : {total:>12,}")
    print(f"  Trainable        : {trainable:>12,}  ({trainable/total*100:.1f}%)")
    print(f"  Frozen           : {total-trainable:>12,}  ({(total-trainable)/total*100:.1f}%)")
    print(f"  Device           : {device}")

    # Forward pass để lấy output shape từng layer chính
    model.eval()
    x = torch.zeros(input_size, device=device)
    print(f"\n  Kiến trúc đầu vào → đầu ra:")
    print(f"  {'Module':<25} {'Output Shape':<25} {'Params':>12}")
    print(f"  {'─'*62}")

    try:
        with torch.no_grad():
            hooks = []
            def make_hook(name):
                def hook(module, inp, out):
                    if isinstance(out, torch.Tensor):
                        shape_str = str(tuple(out.shape))
                        params    = sum(p.numel() for p in module.parameters())
                        print(f"  {name:<25} {shape_str:<25} {params:>12,}")
                hooks.append(hook)
                return hook

            if name == "resnet50":
                tracked = {
                    "conv1+bn1+relu": nn.Sequential(model.conv1, model.bn1, model.relu),
                    "layer1": model.layer1,
                    "layer2": model.layer2,
                    "layer3": model.layer3,
                    "layer4": model.layer4,
                    "avgpool": model.avgpool,
                    "fc (head)": model.fc,
                }
            elif name == "mobilenet_v3":
                tracked = {
                    "features[0]":  model.features[0],
                    "features[1-4]": model.features[1],
                    "features[5-8]": model.features[5] if len(model.features) > 5 else model.features[-2],
                    "features[-1]": model.features[-1],
                    "classifier (head)": model.classifier,
                }
            else:  # efficientnet_b0
                tracked = {
                    "features[0]":  model.features[0],
                    "features[1-3]": model.features[1],
                    "features[4-6]": model.features[4] if len(model.features) > 4 else model.features[-2],
                    "features[-1]": model.features[-1],
                    "classifier (head)": model.classifier,
                }

            handles = []
            for layer_name, layer in tracked.items():
                h = layer.register_forward_hook(make_hook(layer_name))
                handles.append(h)

            _ = model(x)
            for h in handles:
                h.remove()

    except Exception:
        # Fallback đơn giản
        print(f"  [Không thể trace forward pass — in thông tin cơ bản]")

    model.train()
    print(f"{'═'*55}\n")


def get_model_info(model: nn.Module) -> Dict:
    """
    Trả về dict thông tin mô hình — dùng để log hoặc lưu cùng checkpoint.
    """
    total, trainable = count_parameters(model)
    return {
        "model_name":   getattr(model, "_model_name", "unknown"),
        "num_classes":  getattr(model, "_num_classes", None),
        "in_features":  getattr(model, "_in_features", None),
        "total_params": total,
        "trainable":    trainable,
        "frozen":       total - trainable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 – SAVE / LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    filename: Optional[str] = None,
    is_best: bool = False,
) -> str:
    """
    Lưu checkpoint đầy đủ bao gồm: trọng số, optimizer state, epoch, metrics.

    Cấu trúc checkpoint dict:
    {
        "model_name"    : str,
        "num_classes"   : int,
        "epoch"         : int,
        "model_state"   : OrderedDict,   ← model.state_dict()
        "optimizer_state": dict,          ← optimizer.state_dict()
        "metrics"       : dict,           ← {'val_acc': ..., 'val_loss': ...}
        "model_info"    : dict,           ← get_model_info()
    }

    Args:
        model          : nn.Module
        optimizer      : Optimizer đang dùng
        epoch          : Epoch hiện tại
        metrics        : Dict chứa val_acc, val_loss, v.v.
        checkpoint_dir : Thư mục lưu (vd: 'models/')
        filename       : Tên file (None = dùng tên mặc định theo model)
        is_best        : True → cũng lưu thêm bản '<model>_best.pth'

    Returns:
        Đường dẫn file checkpoint đã lưu
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_name = getattr(model, "_model_name", "model")
    if filename is None:
        filename = f"{model_name}_epoch{epoch:03d}.pth"

    save_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "model_name":      model_name,
        "num_classes":     getattr(model, "_num_classes", None),
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "model_info":      get_model_info(model),
    }

    torch.save(checkpoint, save_path)
    print(f"  [SAVE] Checkpoint: {save_path}  "
          f"(epoch={epoch}, {', '.join(f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, float))})")

    # Nếu là best model → lưu thêm bản <model>_best.pth
    if is_best:
        best_path = os.path.join(
            checkpoint_dir,
            CHECKPOINT_NAMES.get(model_name, f"{model_name}_best.pth")
        )
        torch.save(checkpoint, best_path)
        print(f"  [BEST] Lưu best model → {best_path}")

    return save_path


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    pretrained: bool = False,
    device: Optional[torch.device] = None,
    freeze_strategy: str = "full",
) -> Tuple[nn.Module, Dict]:
    """
    Nạp checkpoint và khôi phục model đầy đủ.

    Args:
        checkpoint_path : Đường dẫn file .pth
        num_classes     : Số lớp (phải khớp với lúc lưu)
        pretrained      : False (đã có weights trong checkpoint)
        device          : Target device
        freeze_strategy : Chiến lược freeze sau khi load

    Returns:
        (model, checkpoint_dict)
        → model đã được load weights và chuyển lên device
        → checkpoint_dict chứa epoch, metrics, v.v.

    Ví dụ:
        model, ckpt = load_checkpoint(
            'models/resnet50_best.pth',
            num_classes=5,
        )
        print(f"Epoch: {ckpt['epoch']}, Val Acc: {ckpt['metrics']['val_acc']:.4f}")
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (map to device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get("model_name")
    if model_name is None:
        raise ValueError("Checkpoint không chứa 'model_name'.")

    # Tạo lại model và load state_dict
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_strategy=freeze_strategy,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    epoch   = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"  [LOAD] {checkpoint_path}")
    print(f"         epoch={epoch} | "
          f"{', '.join(f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, float))}")

    return model, checkpoint


def load_for_inference(
    checkpoint_path: str,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Phiên bản rút gọn của load_checkpoint — chỉ dùng cho inference/deploy.
    Model được đặt ở eval mode, toàn bộ tham số bị freeze.

    Returns:
        nn.Module đã load weights, ở chế độ eval + no_grad
    """
    model, _ = load_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        pretrained=False,
        device=device,
        freeze_strategy="none",
    )
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT / DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Vehicle Type Recognition — Model Builder Demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=SUPPORTED_MODELS,
        help="Tên kiến trúc: resnet50 | mobilenet_v3 | efficientnet_b0",
    )
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Số lớp phân loại (mặc định: 5)",
    )
    parser.add_argument(
        "--freeze", type=str, default="head_only",
        choices=["head_only", "partial", "full", "none"],
        help="Chiến lược freeze:\n"
             "  head_only: chỉ train head (Phase 1)\n"
             "  partial  : train head + 1/3 cuối backbone (Phase 2)\n"
             "  full     : train toàn bộ (Phase 3)\n"
             "  none     : inference only",
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Không dùng trọng số ImageNet (khởi tạo ngẫu nhiên)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="In chi tiết kiến trúc từng layer",
    )
    parser.add_argument(
        "--compare_all", action="store_true",
        help="So sánh tham số của cả 3 mô hình",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare_all:
        # ── So sánh 3 mô hình ────────────────────────────────────────────
        print(f"\n{'═'*65}")
        print(f"  ĐỐI SÁNH 3 KIẾN TRÚC — {args.num_classes} classes")
        print(f"{'═'*65}")
        print(f"  {'Model':<20} {'Tổng params':>14} {'Trainable':>12} "
              f"{'Frozen':>12} {'head_only%':>10}")
        print(f"  {'─'*60}")
        for mname in SUPPORTED_MODELS:
            m = build_model(mname, args.num_classes,
                            pretrained=not args.no_pretrained,
                            freeze_strategy="head_only",
                            device=device)
            total, trainable = count_parameters(m)
            frozen = total - trainable
            print(f"  {mname:<20} {total:>14,} {trainable:>12,} "
                  f"{frozen:>12,} {trainable/total*100:>9.1f}%")
        print(f"{'═'*65}\n")
        sys.exit(0)

    # ── Tạo 1 mô hình ────────────────────────────────────────────────────
    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        freeze_strategy=args.freeze,
        device=device,
    )

    if args.summary:
        model_summary(model)

    # ── Smoke test forward pass ───────────────────────────────────────────
    print("Forward pass smoke test...")
    model.eval()
    dummy = torch.zeros(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, args.num_classes), \
        f"Output shape sai: {out.shape} (mong đợi (2, {args.num_classes}))"
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out.shape)}  ← PASS\n")

    # ── Demo switch strategy ──────────────────────────────────────────────
    print("Demo switch_strategy():")
    for strategy in ["head_only", "partial", "full"]:
        switch_strategy(model, strategy)
