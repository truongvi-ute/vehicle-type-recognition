"""
src/dataset.py
==============
Bộ nạp dữ liệu PyTorch (Dataset & DataLoader) cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Gồm 3 phần:
  1. VehicleDataset   — torch.utils.data.Dataset tuỳ chỉnh
  2. Transform Factory — bộ biến đổi ảnh cho train / valid / test
  3. DataLoader Factory— tạo DataLoader cho cả 3 split

Thư viện cần thiết:
  pip install torch torchvision pillow
"""

import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE   = 224       # Kích thước ảnh đầu vào CNN (đã chuẩn hoá bởi data_prep.py)
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Giá trị mean & std của ImageNet — dùng để chuẩn hoá ảnh
# phù hợp với các mô hình pre-trained (ResNet, MobileNet, EfficientNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – VEHICLE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDataset(Dataset):
    """
    Dataset tuỳ chỉnh đọc ảnh đã xử lý từ thư mục có cấu trúc:

        processed_dir/
          ├── train/<ClassName>/*.jpg
          ├── valid/<ClassName>/*.jpg   ← gồm cả *_unseen.jpg và *_copy.jpg
          └── test/<ClassName>/*.jpg

    Mỗi thư mục con (ClassName) là 1 nhãn phân loại.
    Nhãn số (label index) được ánh xạ theo thứ tự alphabet của tên lớp.

    Args:
        root_dir  : Đường dẫn đến `processed_dir/<split>` (vd: data/processed/train)
        transform : Biến đổi torchvision áp dụng lên ảnh (None = không biến đổi)
        class_to_idx : Dict ánh xạ tên lớp → index (nếu None, tự tạo từ thư mục)
        suffix_filter: Chỉ nạp ảnh có suffix nhất định, vd: '_unseen', '_copy', hoặc None (tất cả)

    Thuộc tính công khai:
        classes     : list[str]       — Danh sách tên lớp (sorted)
        class_to_idx: dict[str, int]  — Ánh xạ tên lớp → index số
        idx_to_class: dict[int, str]  — Ánh xạ ngược index → tên lớp
        samples     : list[(path, label)] — Tất cả cặp (đường dẫn ảnh, nhãn)
        targets     : list[int]       — Danh sách nhãn (phục vụ WeightedSampler)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        suffix_filter: Optional[str] = None,
    ) -> None:
        self.root_dir      = Path(root_dir)
        self.transform     = transform
        self.suffix_filter = suffix_filter

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Thư mục không tồn tại: {root_dir}\n"
                "Hãy chạy data_prep.py trước để tạo dữ liệu đã xử lý."
            )

        # ── Xây dựng ánh xạ lớp ──────────────────────────────────────────
        detected_classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not detected_classes:
            raise ValueError(
                f"Không tìm thấy lớp nào trong: {root_dir}\n"
                "Thư mục cần có dạng: <split>/<ClassName>/*.jpg"
            )

        if class_to_idx is not None:
            # Dùng ánh xạ được truyền vào (đảm bảo nhất quán giữa train/valid/test)
            self.class_to_idx = class_to_idx
            # Bổ sung lớp mới nếu có (tăng trưởng dataset)
            for cls in detected_classes:
                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = len(self.class_to_idx)
        else:
            self.class_to_idx = {cls: i for i, cls in enumerate(detected_classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes      = list(self.class_to_idx.keys())

        # ── Quét và tạo danh sách samples ────────────────────────────────
        self.samples: List[Tuple[Path, int]] = []
        self._build_samples()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_valid_image(self, path: Path) -> bool:
        """Kiểm tra extension hợp lệ và filter suffix nếu có."""
        if path.suffix.lower() not in IMG_EXTS:
            return False
        if self.suffix_filter is not None:
            return path.stem.endswith(self.suffix_filter)
        return True

    def _build_samples(self) -> None:
        """Quét thư mục và điền vào self.samples."""
        missing_classes = []
        for cls_name, label in self.class_to_idx.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                missing_classes.append(cls_name)
                continue
            imgs = sorted([p for p in cls_dir.iterdir() if self._is_valid_image(p)])
            for img_path in imgs:
                self.samples.append((img_path, label))

        if missing_classes:
            print(f"  [WARN] Lớp không có thư mục trong split này: {missing_classes}")

        if not self.samples:
            print(f"⚠️ Bỏ qua lỗi thiếu ảnh: Không tìm thấy ảnh nào trong: {self.root_dir}\nsuffix_filter={self.suffix_filter!r}")

    # ── Thuộc tính tiện lợi ───────────────────────────────────────────────────

    @property
    def targets(self) -> List[int]:
        """Danh sách nhãn của tất cả samples — dùng cho WeightedRandomSampler."""
        return [label for _, label in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Trả về (tensor ảnh, nhãn số nguyên).

        Tensor ảnh có shape (C, H, W) = (3, 224, 224), dtype float32,
        đã chuẩn hoá theo ImageNet mean/std nếu transform được áp dụng.
        """
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Không thể đọc ảnh: {img_path}\n{e}")

        if self.transform:
            img = self.transform(img)

        return img, label

    # ── Thông tin debug ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VehicleDataset(\n"
            f"  root={self.root_dir},\n"
            f"  samples={len(self.samples)},\n"
            f"  classes={self.classes},\n"
            f"  suffix_filter={self.suffix_filter!r}\n"
            f")"
        )

    def class_distribution(self) -> Dict[str, int]:
        """Trả về số lượng ảnh mỗi lớp — hỗ trợ phân tích imbalance."""
        dist: Dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dist


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – TRANSFORM FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = IMG_SIZE):
    """
    Tạo pipeline biến đổi ảnh phù hợp với từng split (dùng torchvision.transforms.v2).

    Train : Online Augmentation mạnh — tăng tính tổng quát hoá, chống Overfitting.
    Valid / Test : Chỉ Resize + Normalize — không augment → đánh giá khách quan.

    Kỹ thuật Augmentation trên tập Train:
      - RandomHorizontalFlip : Xe có thể chụp từ trái hoặc phải
      - RandomRotation(±15°)   : Xe bị nghiêng nhẹ khi chụp
      - ColorJitter            : Mô phỏng ánh sáng khác nhau (ngày/đêm/mưa)
      - RandomAffine           : Dịch chuyển & scale để tăng robustness
      - RandomErasing          : Mô phỏng bị che khuất (biển số, cột đèn)
      - Normalize(ImageNet)    : Căn chỉnh phân phối pixel với pre-trained weights

    Args:
        split    : 'train' | 'valid' | 'test'
        img_size : Kích thước ảnh đầu vào (mặc định 224)

    Returns:
        torchvision.transforms.v2.Compose
    """
    split = split.lower()

    _normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return v2.Compose([
            v2.ToImage(),                                  # PIL → TVTensor

            # ── Augmentation hình học ──
            v2.RandomHorizontalFlip(p=0.5),                # Chống thiên vị góc chụp
            v2.RandomRotation(degrees=15),                 # Xoay nghiêng ±15°
            v2.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),                    # Dịch chuyển ±5%
                scale=(0.9, 1.1),                          # Scale ±10%
            ),

            # ── Augmentation màu sắc ──
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),

            # ── Resize + Đưa về float [0, 1] ──
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            _normalize,

            # ── Random Erasing (sau ToDtype, hoạt động trên Tensor) ──
            v2.RandomErasing(
                p=0.2,
                scale=(0.02, 0.10),                        # Xóa 2–10% diện tích ảnh
                ratio=(0.3, 3.3),
                value=0,                                   # Màu đen (phù hợp padding viền đen)
            ),
        ])

    elif split in ("valid", "test"):
        # Tuyệt đối giữ nguyên → đảm bảo đánh giá công bằng
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            _normalize,
        ])

    else:
        raise ValueError(
            f"split phải là 'train', 'valid' hoặc 'test'. Nhận: '{split}'"
        )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Đảo ngược chuẩn hoá ImageNet để hiển thị ảnh.
    Input : Tensor (C, H, W) đã normalize
    Output: Tensor (C, H, W) giá trị [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 – DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: VehicleDataset) -> WeightedRandomSampler:
    """
    Tạo WeightedRandomSampler để xử lý dataset mất cân bằng lớp (class imbalance).

    Lớp ít ảnh hơn sẽ được lấy mẫu với xác suất cao hơn → mô hình học đều các lớp.
    Chỉ dùng cho tập TRAIN, không dùng cho valid/test.

    Returns:
        WeightedRandomSampler — truyền vào DataLoader(sampler=...)
    """
    targets = dataset.targets
    class_counts = torch.zeros(dataset.num_classes, dtype=torch.float)
    for label in targets:
        class_counts[label] += 1

    # Tránh chia cho 0 nếu lớp không có ảnh
    class_counts = torch.clamp(class_counts, min=1)
    class_weights = 1.0 / class_counts

    # Trọng số từng sample = trọng số lớp của nó
    sample_weights = torch.tensor([class_weights[t] for t in targets])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
    use_weighted_sampler: bool = True,
    pin_memory: bool = False,
    valid_suffix: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Tạo DataLoader cho train, valid, test từ thư mục `processed_dir`.

    Args:
        processed_dir        : Thư mục gốc chứa train/, valid/, test/
        batch_size           : Số ảnh mỗi batch (mặc định 32)
        num_workers          : Số luồng nạp dữ liệu song song (0 = main thread)
        img_size             : Kích thước ảnh đầu vào
        use_weighted_sampler : Bật cân bằng lớp cho tập train (mặc định True)
        pin_memory           : True nếu dùng GPU (tăng tốc copy RAM→VRAM)
        valid_suffix         : Lọc valid theo suffix: None (tất cả), '_unseen', '_copy'

    Returns:
        (train_loader, valid_loader, test_loader, class_to_idx)

    Ví dụ sử dụng:
        train_dl, valid_dl, test_dl, cls_map = create_dataloaders(
            processed_dir="data/processed",
            batch_size=32,
            num_workers=4,
        )
        for images, labels in train_dl:
            # images: Tensor(B, 3, 224, 224), labels: Tensor(B,)
            ...
    """
    processed_dir = Path(processed_dir)

    # ── Tạo Train Dataset & lấy class_to_idx chuẩn ──────────────────────
    train_dataset = VehicleDataset(
        root_dir=processed_dir / "train",
        transform=get_transforms("train", img_size),
    )
    class_to_idx = train_dataset.class_to_idx   # Dùng lại cho valid & test

    # ── Valid Dataset ─────────────────────────────────────────────────────
    valid_dataset = VehicleDataset(
        root_dir=processed_dir / "valid",
        transform=get_transforms("valid", img_size),
        class_to_idx=class_to_idx,
        suffix_filter=valid_suffix,
    )

    # ── Test Dataset ──────────────────────────────────────────────────────
    test_dataset = VehicleDataset(
        root_dir=processed_dir / "test",
        transform=get_transforms("test", img_size),
        class_to_idx=class_to_idx,
    )

    # ── In thống kê ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  VehicleDataset — Thống kê")
    print(f"{'='*55}")
    print(f"  processed_dir : {processed_dir}")
    print(f"  batch_size    : {batch_size}")
    print(f"  Số lớp        : {train_dataset.num_classes}")
    print(f"  Ánh xạ lớp    : {class_to_idx}")
    print(f"  {'Split':<8} {'Ảnh':>6} {'Batch':>6}")
    print(f"  {'-'*25}")
    for name, ds in [("train", train_dataset), ("valid", valid_dataset), ("test", test_dataset)]:
        n_batches = (len(ds) + batch_size - 1) // batch_size
        print(f"  {name:<8} {len(ds):>6} {n_batches:>6}")

    # Phân phối lớp tập train
    dist = train_dataset.class_distribution()
    print(f"\n  Phân phối tập Train:")
    for cls, cnt in dist.items():
        bar = "█" * (cnt * 20 // max(dist.values()))
        print(f"    {cls:<15} {cnt:>5}  {bar}")
    print(f"{'='*55}\n")

    # ── Tạo DataLoader ────────────────────────────────────────────────────
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Train: dùng WeightedRandomSampler hoặc shuffle
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            **common_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,   # Không shuffle valid/test → đảm bảo tính nhất quán đánh giá
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, valid_loader, test_loader, class_to_idx


def create_single_loader(
    split_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
    class_to_idx: Optional[Dict[str, int]] = None,
    split: str = "test",
    suffix_filter: Optional[str] = None,
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Tạo DataLoader cho một split đơn lẻ.
    Hữu ích khi chỉ cần chạy inference trên test hoặc phân tích riêng valid_unseen/valid_copy.

    Returns:
        (dataloader, class_to_idx)
    """
    dataset = VehicleDataset(
        root_dir=split_dir,
        transform=get_transforms(split, img_size),
        class_to_idx=class_to_idx,
        suffix_filter=suffix_filter,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, dataset.class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / SMOKE TEST (chạy trực tiếp)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="VehicleDataset — Kiểm tra DataLoader",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=r"g:\Data\Projects\VehicleTypeRecognition\data\processed",
        help="Thư mục processed/ (chứa train/, valid/, test/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Kích thước batch (mặc định: 8)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Số luồng nạp dữ liệu (mặc định: 0)",
    )
    parser.add_argument(
        "--valid_suffix",
        type=str,
        default=None,
        choices=[None, "_unseen", "_copy"],
        help="Lọc valid: None=tất cả, _unseen=chỉ ảnh mới, _copy=chỉ ảnh copy từ train",
    )
    args = parser.parse_args()

    print("\n=== VehicleDataset Smoke Test ===")

    try:
        train_dl, valid_dl, test_dl, cls_map = create_dataloaders(
            processed_dir=args.processed_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_weighted_sampler=True,
            valid_suffix=args.valid_suffix,
        )

        print("Lấy 1 batch từ mỗi split:")
        for name, loader in [("train", train_dl), ("valid", valid_dl), ("test", test_dl)]:
            imgs, labels = next(iter(loader))
            print(f"  [{name}] images: {tuple(imgs.shape)}  labels: {labels.tolist()}")
            # Kiểm tra giá trị tensor hợp lệ
            assert not torch.isnan(imgs).any(), f"[{name}] Có NaN trong tensor ảnh!"
            assert imgs.shape[1:] == (3, IMG_SIZE, IMG_SIZE), \
                f"[{name}] Shape sai: {imgs.shape}"

        # Kiểm tra denormalize
        imgs, _ = next(iter(test_dl))
        denorm = denormalize(imgs[0])
        assert denorm.min() >= 0.0 and denorm.max() <= 1.0, \
            "denormalize() cho kết quả ngoài [0,1]"
        print("\n  denormalize(): OK — giá trị nằm trong [0, 1]")

        print(f"\n=== TẤT CẢ KIỂM TRA THÀNH CÔNG — DataLoader sẵn sàng ===")

    except FileNotFoundError as e:
        print(f"\n[WARN] Chưa có dữ liệu processed:\n  {e}")
        print("  → Hãy chạy trước: python src/data_prep.py")
        sys.exit(0)
