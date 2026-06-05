"""
src/dataset.py
==============
Module PyTorch DataLoader cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Dữ liệu đầu vào đã được tiền xử lý hoàn chỉnh bởi data_prep.py:
  - Ảnh kích thước 224×224 (BGR → được xử lý bởi OpenCV pipelines).
  - Tập Train đã cân bằng class và augment vật lý x4 pipelines.
  - Tập Valid và Test KHÔNG augment thêm (giữ nguyên để đánh giá khách quan).

Sử dụng:
    from dataset import create_dataloaders

    train_loader, valid_loader, test_loader, class_names = create_dataloaders(
        data_dir   = "data/augmented",
        batch_size = 32,
        num_workers= 0,   # Windows: đặt 0 nếu gặp lỗi spawn
    )

Thư viện cần thiết:
    pip install torch torchvision
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ CHUẨN HOÁ IMAGENET
# ─────────────────────────────────────────────────────────────────────────────

# Giá trị mean/std chuẩn ImageNet — phù hợp với backbone pre-trained
# (ResNet-50, ViT) được huấn luyện trên ImageNet.
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

NUM_CLASSES: int = 10   # bicycle, boat, bus, car, helicopter,
                        # minibus, motorcycle, taxi, train, truck


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 — TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> v2.Compose:
    """
    Tạo pipeline biến đổi ảnh tối giản dùng torchvision.transforms.v2.

    Lý do KHÔNG thêm flip/rotate ở đây:
        Tập Train đã trải qua 4 pipelines tăng cường vật lý (Base/Night/Rain/Sun)
        trong data_prep.py.  Thêm biến đổi hình học online có thể làm "lệch"
        phân phối của tập Train so với Valid/Test và gây Train-Serving Skew.

    Quy trình chung cho MỌI split (train / valid / test):
        1. v2.ToImage()          — Chuyển PIL Image / np.ndarray → Tensor uint8
        2. v2.ToDtype(float32)   — Chuẩn hóa về [0.0, 1.0]
        3. v2.Normalize(...)     — Chuẩn hoá theo mean/std ImageNet

    Args:
        split: Một trong "train", "valid", "test" (không phân biệt hoa/thường).
               Giữ tham số này để dễ mở rộng sau (VD: thêm RandAugment cho train).

    Returns:
        v2.Compose — pipeline transform hoàn chỉnh.
    """
    split = split.lower().strip()

    # Pipeline chung — giống nhau cho cả 3 split trong thiết kế hiện tại
    pipeline = v2.Compose([
        v2.ToImage(),                                      # → Tensor uint8 (C, H, W)
        v2.ToDtype(torch.float32, scale=True),             # → [0.0, 1.0]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),# → chuẩn hoá ImageNet
    ])

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 — MIXUP / CUTMIX COLLATE
# ─────────────────────────────────────────────────────────────────────────────

def get_mixup_cutmix(
    num_classes: int = NUM_CLASSES,
    cutmix_alpha: float = 1.0,
    mixup_alpha: float  = 0.2,
) -> v2.RandomChoice:
    """
    Tạo transform trộn ảnh trong RAM dùng RandomChoice giữa CutMix và MixUp.

    Nguyên lý:
        - CutMix (alpha=1.0): Cắt một vùng chữ nhật ngẫu nhiên từ ảnh A,
          dán chồng lên ảnh B. Nhãn được trộn theo tỷ lệ diện tích.
          → Buộc mô hình học toàn bộ cấu trúc phương tiện, không chỉ tập
            trung vào 1 chi tiết nổi bật.

        - MixUp (alpha=0.2): Kết hợp tuyến tính 2 ảnh:
            x_mix = λ·x_a + (1−λ)·x_b,  y_mix = λ·y_a + (1−λ)·y_b
          với λ ~ Beta(alpha, alpha).
          → Làm mịn nhãn (Label Smoothing ngầm định), giảm Overfitting.

        - RandomChoice: Với mỗi batch, ngẫu nhiên chọn 1 trong 2 kỹ thuật.

    LƯU Ý QUAN TRỌNG:
        Transform này hoạt động trên CẢ BATCH (images + labels), không phải
        từng ảnh đơn lẻ. Phải được gắn vào `collate_fn` của DataLoader.
        Xem hàm `_mixup_cutmix_collate_fn()` bên dưới.

    Args:
        num_classes  : Số lượng lớp phân loại (mặc định = 10).
        cutmix_alpha : Tham số Beta cho CutMix (càng lớn → tỷ lệ cắt ngẫu nhiên hơn).
        mixup_alpha  : Tham số Beta cho MixUp (nhỏ → trộn ít, lớn → trộn nhiều).

    Returns:
        v2.RandomChoice — transform batch-level sẵn sàng dùng trong collate_fn.
    """
    return v2.RandomChoice([
        v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
        v2.MixUp(num_classes=num_classes,  alpha=mixup_alpha),
    ])


def _build_mixup_collate_fn(
    mixup_cutmix: v2.RandomChoice,
) -> Callable[[List], Tuple[Tensor, Tensor]]:
    """
    Xây dựng hàm collate_fn tùy chỉnh để áp dụng MixUp/CutMix trên batch.

    PyTorch DataLoader gọi collate_fn sau khi ghép các mẫu riêng lẻ thành
    batch.  Đây là điểm chèn đúng cho các transform cần xử lý cả batch.

    Args:
        mixup_cutmix: Transform RandomChoice(CutMix, MixUp) từ get_mixup_cutmix().

    Returns:
        collate_fn — hàm nhận list[(image, label)] và trả về (images, labels)
                     đã trộn theo CutMix hoặc MixUp.
    """
    # Lấy hàm collate mặc định của PyTorch để ghép tensor trước
    default_collate = torch.utils.data.default_collate

    def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        """
        1. Ghép batch bằng default_collate → images: (B, C, H, W), labels: (B,)
        2. Áp dụng mixup_cutmix(images, labels) → trả về tensor đã trộn.
        """
        images, labels = default_collate(batch)     # (B, C, H, W), (B,)
        images, labels = mixup_cutmix(images, labels)
        return images, labels

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 — CREATE DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    data_dir:    str,
    batch_size:  int  = 32,
    num_workers: int  = 0,
    pin_memory:  Optional[bool] = None,
    mixup_alpha:   float = 0.2,
    cutmix_alpha:  float = 1.0,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Khởi tạo và trả về 3 DataLoader cho Train / Valid / Test.

    Cấu trúc thư mục mong đợi tại `data_dir`:
        data_dir/
          train/<class>/   ← ảnh đã augment x4 pipelines
          valid/<class>/   ← ảnh nguyên bản (không augment thêm)
          test/<class>/    ← ảnh nguyên bản (không augment thêm)

    Thiết kế:
        - Dataset    : torchvision.datasets.ImageFolder (tự động đọc nhãn từ tên thư mục).
        - train_loader: shuffle=True + collate_fn MixUp/CutMix online.
        - valid/test : shuffle=False + collate mặc định (đánh giá khách quan).
        - pin_memory  : Tự động bật nếu CUDA khả dụng (tăng tốc copy CPU→GPU).

    Args:
        data_dir       : Đường dẫn thư mục gốc (ví dụ: "data/augmented").
        batch_size     : Kích thước mỗi batch.
        num_workers    : Số luồng nạp dữ liệu song song.
                         ⚠️ Windows: Đặt 0 nếu gặp lỗi BrokenPipeError.
        pin_memory     : Ghim bộ nhớ CPU → tăng tốc chuyển lên GPU.
                         None → tự động theo CUDA availability.
        mixup_alpha    : Tham số Beta cho MixUp (khuyến nghị: 0.2).
        cutmix_alpha   : Tham số Beta cho CutMix (khuyến nghị: 1.0).
        prefetch_factor: Số batch prefetch (None = tắt, chỉ bật khi num_workers > 0).

    Returns:
        Tuple gồm 4 phần tử:
            [0] train_loader : DataLoader với MixUp/CutMix collate_fn.
            [1] valid_loader : DataLoader chuẩn (không trộn ảnh).
            [2] test_loader  : DataLoader chuẩn (không trộn ảnh).
            [3] class_names  : List[str] tên các lớp theo thứ tự index
                               (ví dụ: ["bicycle", "boat", ..., "truck"]).

    Raises:
        FileNotFoundError: Nếu một trong các thư mục train/valid/test không tồn tại.
        RuntimeError     : Nếu class_to_idx không nhất quán giữa các split.
    """
    # ── Xác nhận thư mục tồn tại ──────────────────────────────────────────
    for split_name in ("train", "valid", "test"):
        split_path = os.path.join(data_dir, split_name)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục split '{split_name}': {split_path}\n"
                f"Hãy chạy data_prep.py --step 3 trước."
            )

    # ── Tự động xác định pin_memory ───────────────────────────────────────
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ── Tạo Transforms ────────────────────────────────────────────────────
    train_tf = get_transforms("train")
    valid_tf = get_transforms("valid")
    test_tf  = get_transforms("test")

    # ── Tạo Datasets bằng ImageFolder ────────────────────────────────────
    train_ds = datasets.ImageFolder(
        root      = os.path.join(data_dir, "train"),
        transform = train_tf,
    )
    valid_ds = datasets.ImageFolder(
        root      = os.path.join(data_dir, "valid"),
        transform = valid_tf,
    )
    test_ds = datasets.ImageFolder(
        root      = os.path.join(data_dir, "test"),
        transform = test_tf,
    )

    # ── Phát hiện và xác thực class_to_idx ───────────────────────────────
    class_to_idx: Dict[str, int] = train_ds.class_to_idx
    class_names:  List[str]      = train_ds.classes

    # Kiểm tra nhất quán giữa 3 split (tên lớp phải khớp nhau)
    for ds_name, ds in [("valid", valid_ds), ("test", test_ds)]:
        if ds.class_to_idx != class_to_idx:
            raise RuntimeError(
                f"class_to_idx của '{ds_name}' không khớp với 'train'.\n"
                f"Train : {class_to_idx}\n"
                f"{ds_name.capitalize()}: {ds.class_to_idx}\n"
                f"Kiểm tra lại cấu trúc thư mục trong {data_dir}."
            )

    # ── Tạo MixUp / CutMix collate_fn chỉ cho Train ──────────────────────
    mixup_cutmix     = get_mixup_cutmix(
        num_classes  = len(class_names),
        cutmix_alpha = cutmix_alpha,
        mixup_alpha  = mixup_alpha,
    )
    train_collate_fn = _build_mixup_collate_fn(mixup_cutmix)

    # ── Cấu hình chung cho DataLoader ─────────────────────────────────────
    # prefetch_factor chỉ hợp lệ khi num_workers > 0
    _prefetch = prefetch_factor if (num_workers > 0 and prefetch_factor) else None

    loader_kwargs = dict(
        pin_memory     = pin_memory,
        num_workers    = num_workers,
        prefetch_factor= _prefetch,
    )

    # ── Tạo 3 DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(
        dataset    = train_ds,
        batch_size = batch_size,
        shuffle    = True,             # Xáo trộn mỗi epoch
        collate_fn = train_collate_fn, # MixUp / CutMix online
        drop_last  = True,             # Bỏ batch cuối không đủ size (tránh lỗi BN)
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        dataset    = valid_ds,
        batch_size = batch_size,
        shuffle    = False,            # Không xáo trộn — đánh giá nhất quán
        collate_fn = None,             # Dùng default_collate (nhãn nguyên, không trộn)
        drop_last  = False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        dataset    = test_ds,
        batch_size = batch_size,
        shuffle    = False,
        collate_fn = None,
        drop_last  = False,
        **loader_kwargs,
    )

    return train_loader, valid_loader, test_loader, class_names


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 — TIỆN ÍCH THỐNG KÊ
# ─────────────────────────────────────────────────────────────────────────────

def describe_dataset(loader: DataLoader, split_name: str = "split") -> None:
    """
    In thống kê tóm tắt về DataLoader: số batch, số mẫu, shape batch.

    Args:
        loader     : DataLoader cần kiểm tra.
        split_name : Tên hiển thị (vd: "train", "valid", "test").
    """
    ds   = loader.dataset
    n    = len(ds)                  # type: ignore[arg-type]
    nb   = len(loader)
    bs   = loader.batch_size or "?"
    shuf = getattr(loader.sampler, "generator", None) is not None

    print(f"  [{split_name:>5}] {n:>7,} samples | {nb:>5,} batches "
          f"(batch_size={bs}) | shuffle={loader.shuffle if hasattr(loader, 'shuffle') else '?'}")  # type: ignore


def get_class_distribution(
    dataset: datasets.ImageFolder,
) -> Dict[str, int]:
    """
    Đếm số lượng ảnh của từng lớp trong ImageFolder dataset.

    Args:
        dataset: Đối tượng ImageFolder (train_ds, valid_ds, hoặc test_ds).

    Returns:
        Dict {class_name: count} — sắp xếp theo tên lớp tăng dần.
    """
    dist: Dict[str, int] = {cls: 0 for cls in dataset.classes}
    for _, label_idx in dataset.samples:
        dist[dataset.classes[label_idx]] += 1
    return dict(sorted(dist.items()))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST  (chạy: python src/dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Sửa encoding stdout cho Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Dataset Smoke Test — kiểm tra DataLoader hoạt động đúng.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/augmented",
        help="Thư mục gốc chứa train/ valid/ test/.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Kích thước batch dùng để test.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Số worker. Dùng 0 trên Windows để tránh lỗi spawn.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("VEHICLE TYPE RECOGNITION — Dataset Smoke Test")
    print("=" * 65)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  CUDA        : {torch.cuda.is_available()}")
    print()

    # ── Kiểm tra thư mục ─────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"[LỖI] Không tìm thấy thư mục: {args.data_dir}")
        print("       Hãy chạy `python src/data_prep.py --all` trước.")
        sys.exit(1)

    # ── Tạo DataLoaders ───────────────────────────────────────────────────
    print("Đang khởi tạo DataLoaders ...")
    train_loader, valid_loader, test_loader, class_names = create_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Thống kê dataset ──────────────────────────────────────────────────
    print(f"\nTổng số lớp  : {len(class_names)}")
    print(f"Tên các lớp  : {class_names}\n")

    print("Thống kê DataLoader:")
    for loader, name in [
        (train_loader, "train"),
        (valid_loader, "valid"),
        (test_loader,  "test"),
    ]:
        ds       = loader.dataset               # type: ignore[union-attr]
        n_sample = len(ds)
        n_batch  = len(loader)
        print(f"  [{name:>5}] {n_sample:>8,} samples | "
              f"{n_batch:>5,} batches (batch_size={loader.batch_size})")

    # ── Phân phối lớp trên từng split ────────────────────────────────────
    print("\nPhân phối lớp (Train):")
    dist = get_class_distribution(train_loader.dataset)  # type: ignore[arg-type]
    for cls, cnt in dist.items():
        bar = "█" * (cnt // max(1, max(dist.values()) // 30))
        print(f"  {cls:>12} : {cnt:>7,}  {bar}")

    # ── Lấy 1 batch Train và kiểm tra shape ──────────────────────────────
    print("\nLấy 1 batch từ train_loader ...")
    imgs, labels = next(iter(train_loader))

    print(f"\n  images.shape : {tuple(imgs.shape)}")    # (B, C, H, W)
    print(f"  images.dtype : {imgs.dtype}")             # torch.float32
    print(f"  labels.shape : {tuple(labels.shape)}")    # (B, NUM_CLASSES) — soft labels sau MixUp/CutMix
    print(f"  labels.dtype : {labels.dtype}")           # torch.float32
    print(f"  images min/max: [{imgs.min():.3f}, {imgs.max():.3f}]")

    # ── Kiểm tra Valid/Test batch (nhãn vẫn là long integers) ────────────
    print("\nLấy 1 batch từ valid_loader ...")
    v_imgs, v_labels = next(iter(valid_loader))
    print(f"  valid images.shape : {tuple(v_imgs.shape)}")
    print(f"  valid labels.shape : {tuple(v_labels.shape)}")
    print(f"  valid labels.dtype : {v_labels.dtype}")    # torch.int64

    print("\n✅ Smoke test PASSED — DataLoader hoạt động đúng.")
    print("=" * 65)
