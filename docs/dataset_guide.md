# `src/dataset.py` — Hướng dẫn sử dụng DataLoader

Bộ nạp dữ liệu PyTorch cho đồ án **Nhận dạng Phương tiện Giao thông**.

> **Yêu cầu tiên quyết:** Đã chạy `data_prep.py` để tạo thư mục `data/processed/`.

---

## Cài đặt

```powershell
pip install torch torchvision pillow
```

---

## Cấu trúc 3 thành phần

```
dataset.py
├── VehicleDataset        # torch.utils.data.Dataset tuỳ chỉnh
├── get_transforms()      # Transform Factory (train/valid/test)
└── create_dataloaders()  # DataLoader Factory (tất cả 3 split)
```

---

## 1. `VehicleDataset`

Đọc ảnh từ thư mục đã xử lý, ánh xạ tên thư mục → nhãn số.

```python
from src.dataset import VehicleDataset, get_transforms

dataset = VehicleDataset(
    root_dir="data/processed/train",
    transform=get_transforms("train"),
)

print(dataset.classes)        # ['Bus', 'Truck', ...]
print(dataset.class_to_idx)  # {'Bus': 0, 'Truck': 1, ...}
print(len(dataset))           # Số ảnh

img_tensor, label = dataset[0]
# img_tensor: torch.Tensor (3, 224, 224)
# label     : int
```

### Tham số khởi tạo

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `root_dir` | `str` | — | Đường dẫn đến `processed/<split>` |
| `transform` | `Callable` | `None` | Pipeline biến đổi torchvision |
| `class_to_idx` | `dict` | `None` | Nếu None, tự tạo từ thư mục (sorted) |
| `suffix_filter` | `str` | `None` | Lọc ảnh theo suffix filename |

### Lọc Valid theo suffix

Valid split chứa 2 loại ảnh (đặt tên khác nhau bởi `data_prep.py`):

```python
# Chỉ lấy ảnh Unseen (5% hoàn toàn mới)
ds_unseen = VehicleDataset("data/processed/valid", suffix_filter="_unseen")

# Chỉ lấy ảnh Copy từ Train (5% trùng lặp)
ds_copy = VehicleDataset("data/processed/valid", suffix_filter="_copy")

# Tất cả (mặc định)
ds_all = VehicleDataset("data/processed/valid")
```

### Phân phối lớp

```python
dist = dataset.class_distribution()
# {'Bus': 850, 'Truck': 820, ...}
```

---

## 2. `get_transforms()` — Transform Factory

Tránh Overfitting và tăng sức mạnh dự đoán (Robustness) bằng torchvision.transforms.v2 (Online Augmentation).

```python
from src.dataset import get_transforms

train_tf = get_transforms("train")  # Augmentation hình học & màu sắc đầy đủ
valid_tf = get_transforms("valid")  # Chỉ Resize + Normalize
test_tf  = get_transforms("test")   # Chỉ Resize + Normalize
```

### Bảng augmentation áp dụng

| Biến đổi | Train | Valid/Test | Lý do |
|---|:---:|:---:|---|
| `v2.Resize(224×224)` | ✅ | ✅ | Đảm bảo kích thước đúng |
| `v2.RandomHorizontalFlip(p=0.5)` | ✅ | ❌ | Xe chụp từ trái hoặc phải |
| `v2.RandomRotation(±15°)` | ✅ | ❌ | Camera hơi nghiêng khi chụp |
| `v2.RandomAffine(translate, scale)` | ✅ | ❌ | Góc nhìn, khoảng cách khác nhau |
| `v2.ColorJitter(...)` | ✅ | ❌ | Điều kiện ánh sáng thay đổi |
| `v2.ToImage()` & `v2.ToDtype()` | ✅ | ✅ | Chuyển PIL → TVTensor [0,1] |
| `v2.Normalize(ImageNet mean/std)` | ✅ | ✅ | Căn chỉnh với pre-trained weights |
| `v2.RandomErasing(p=0.2)` | ✅ | ❌ | Mô phỏng bị che khuất một phần |

> **Tại sao Normalize theo ImageNet?**
> ResNet-50, MobileNet-V3, EfficientNet-B0 đều được pre-train trên ImageNet.
> Phân phối pixel phải khớp để Transfer Learning hoạt động hiệu quả.

### Đảo ngược Normalize để hiển thị ảnh

```python
from src.dataset import denormalize

imgs, labels = next(iter(train_loader))  # (B, 3, 224, 224)
img_display = denormalize(imgs[0])       # Tensor (3, 224, 224) trong [0, 1]

# Hiển thị bằng matplotlib
import matplotlib.pyplot as plt
plt.imshow(img_display.permute(1, 2, 0).numpy())
plt.show()
```

---

## 3. `create_dataloaders()` — DataLoader Factory

Cách nhanh nhất để tạo cả 3 DataLoader cùng lúc.

```python
from src.dataset import create_dataloaders

train_loader, valid_loader, test_loader, class_to_idx = create_dataloaders(
    processed_dir="data/processed",
    batch_size=32,
    num_workers=4,          # Windows: dùng 0 nếu gặp lỗi multiprocessing
    use_weighted_sampler=True,
    pin_memory=True,        # True nếu dùng GPU
)
```

### Tham số

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `processed_dir` | `str` | — | Thư mục chứa train/, valid/, test/ |
| `batch_size` | `int` | `32` | Số ảnh mỗi batch |
| `num_workers` | `int` | `0` | Luồng nạp song song |
| `img_size` | `int` | `224` | Kích thước ảnh đầu vào |
| `use_weighted_sampler` | `bool` | `True` | Cân bằng lớp bất đối xứng |
| `pin_memory` | `bool` | `False` | Tối ưu copy RAM→VRAM cho GPU |
| `valid_suffix` | `str` | `None` | Lọc valid: `None`, `'_unseen'`, `'_copy'` |

### Returns

```python
(train_loader, valid_loader, test_loader, class_to_idx)
# class_to_idx: {'Bus': 0, 'Truck': 1, ...}  ← dùng lại cho model output
```

### Sử dụng trong vòng lặp huấn luyện

```python
for epoch in range(num_epochs):
    # ── Train ──
    model.train()
    for images, labels in train_loader:
        images = images.to(device)   # Tensor(B, 3, 224, 224)
        labels = labels.to(device)   # Tensor(B,)  — LongTensor
        outputs = model(images)      # Tensor(B, num_classes)
        loss = criterion(outputs, labels)
        ...

    # ── Validate ──
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            ...
```

---

## 4. `WeightedRandomSampler` — Xử lý mất cân bằng lớp

Khi một lớp ít ảnh hơn (ví dụ chỉ 500 ảnh Bike trong khi Truck có 1500 ảnh),
mô hình sẽ bị bias về lớp nhiều ảnh.

`WeightedRandomSampler` cân bằng xác suất lấy mẫu:

```
Lớp ít ảnh → xác suất lấy mẫu cao hơn
Lớp nhiều ảnh → xác suất lấy mẫu thấp hơn
```

```python
from src.dataset import VehicleDataset, make_weighted_sampler, get_transforms
from torch.utils.data import DataLoader

train_ds = VehicleDataset("data/processed/train", transform=get_transforms("train"))
sampler  = make_weighted_sampler(train_ds)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    sampler=sampler,     # Thay shuffle=True
    num_workers=4,
)
```

---

## 5. Tạo DataLoader đơn lẻ

Hữu ích khi chỉ cần inference hoặc so sánh riêng valid_unseen vs valid_copy:

```python
from src.dataset import create_single_loader

# Chỉ valid_unseen
unseen_loader, cls_map = create_single_loader(
    split_dir="data/processed/valid",
    batch_size=32,
    split="valid",
    suffix_filter="_unseen",
    class_to_idx=class_to_idx,  # Dùng lại ánh xạ từ train
)

# Chỉ valid_copy
copy_loader, _ = create_single_loader(
    split_dir="data/processed/valid",
    batch_size=32,
    split="valid",
    suffix_filter="_copy",
    class_to_idx=class_to_idx,
)
```

---

## 6. Kiểm tra nhanh từ command line

```powershell
# Kiểm tra DataLoader với dữ liệu thực
python src/dataset.py --processed_dir data/processed --batch_size 8

# Kiểm tra chỉ valid_unseen
python src/dataset.py --valid_suffix _unseen

# Kiểm tra với nhiều worker (Linux/Mac, không dùng trên Windows)
python src/dataset.py --num_workers 4
```

---

## 7. Lưu ý khi dùng trên Windows

```python
# Windows gặp lỗi multiprocessing khi num_workers > 0
# → Bọc code trong if __name__ == '__main__':

if __name__ == '__main__':
    train_loader, valid_loader, test_loader, cls_map = create_dataloaders(
        processed_dir="data/processed",
        batch_size=32,
        num_workers=4,   # Windows OK nếu có guard này
    )
```

Hoặc dùng `num_workers=0` (main thread) nếu không muốn phức tạp.

---

## 8. Tích hợp đầy đủ — Ví dụ thực tế

```python
import torch
from src.dataset import create_dataloaders, denormalize

# 1. Tạo DataLoader
train_dl, valid_dl, test_dl, class_to_idx = create_dataloaders(
    processed_dir="data/processed",
    batch_size=32,
    num_workers=0,
    use_weighted_sampler=True,
    pin_memory=torch.cuda.is_available(),
)

num_classes = len(class_to_idx)   # Truyền vào model
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 2. Dùng trong training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for images, labels in train_dl:
    images, labels = images.to(device), labels.to(device)
    # images.shape == (32, 3, 224, 224)
    # labels.shape == (32,)
    ...

# 3. Hiển thị batch đầu tiên
imgs, lbls = next(iter(test_dl))
for i in range(min(4, len(imgs))):
    img = denormalize(imgs[i]).permute(1, 2, 0).numpy()
    print(f"Label: {idx_to_class[lbls[i].item()]}")
```

---

*Cập nhật: 2026-05-25 | Phiên bản: 1.0*
