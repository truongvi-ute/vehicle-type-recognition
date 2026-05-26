# `src/model.py` — Hướng dẫn Khởi tạo & Quản lý Mô hình

Module quản lý 3 kiến trúc mạng nơ-ron tích chập (CNN) phục vụ bài toán **Nhận dạng Phương tiện Giao thông** bằng phương pháp Transfer Learning.

---

## 1. Các kiến trúc hỗ trợ

Cả 3 mô hình đều được khởi tạo từ pre-trained weights của ImageNet, với classification head được thay thế tuỳ chỉnh để phù hợp với số lớp xe.

| Tên lệnh (CLI) | Kiến trúc gốc | Số tham số | Đặc điểm & Vai trò |
|---|---|---|---|
| `resnet50` | ResNet-50 | ~24.5 M | Có Skip Connections, độ chính xác cao. Dùng làm **Baseline**. |
| `efficientnet_b0`| EfficientNet-B0 | ~4.6 M | Compound Scaling. Tối ưu cân bằng giữa độ chính xác và tốc độ. |
| `mobilenet_v3` | MobileNetV3-Small | ~1.2 M | Depthwise Separable Conv, siêu nhẹ. Dùng cho **Inference thực tế**. |

---

## 2. Cách khởi tạo mô hình (Factory Function)

Sử dụng hàm `build_model` là điểm truy cập duy nhất để tạo mô hình.

```python
import torch
from src.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo mô hình ResNet-50, phân loại 5 lớp xe, đóng băng backbone (chỉ train head)
model = build_model(
    model_name="resnet50",
    num_classes=5,
    pretrained=True,            # Tải trọng số ImageNet
    dropout=0.4,                # Dropout rate ở head
    freeze_strategy="head_only", # Chiến lược đóng băng
    device=device               # Tự động map lên GPU/CPU
)
```

---

## 3. Chiến lược đóng băng (Freeze Strategy)

Trong Transfer Learning, không nên train toàn bộ mô hình ngay từ đầu vì gradient lớn có thể phá hỏng trọng số đã học tốt từ ImageNet. Module hỗ trợ 3 giai đoạn (Phase) thông qua tham số `freeze_strategy`:

| Tên chiến lược | Mô tả (Những gì được train?) | Vai trò (Phase) |
|---|---|---|
| `head_only` | Chỉ train Classification Head (backbone bị freeze) | **Phase 1 (Warm-up):** Chạy 5-10 epoch đầu để head thích nghi với đặc trưng mới. |
| `partial` | Train Head + 1/3 cuối của Backbone | **Phase 2 (Fine-tune):** Mở khoá các khối Conv cuối để học đường nét đặc thù của xe. |
| `full` | Mở khoá train toàn bộ mô hình | **Phase 3:** Train với Learning Rate cực nhỏ (tuỳ chọn). |
| `none` | Đóng băng toàn bộ | Chỉ dùng khi Inference / Testing. |

### Chuyển đổi chiến lược khi đang huấn luyện

Sử dụng hàm `switch_strategy` mà không cần tạo lại mô hình:

```python
from src.model import switch_strategy

# Giả sử model đang ở 'head_only'
for epoch in range(1, 6):
    train_one_epoch(...)  # Train 5 epoch đầu

# Chuyển sang 'partial' (mở khoá 1/3 mạng)
switch_strategy(model, "partial")
# Nhớ cập nhật lại optimizer vì thông số trainable đã thay đổi!
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

for epoch in range(6, 16):
    train_one_epoch(...)
```

---

## 4. Lưu và Nạp Checkpoint

Hệ thống lưu checkpoint không chỉ có trọng số (weights) mà còn lưu cả thông tin mô hình, epoch, và metrics để dễ dàng resume (huấn luyện tiếp).

### Lưu mô hình (Save)

```python
from src.model import save_checkpoint

metrics = {"val_acc": 0.92, "val_loss": 0.25}

saved_path = save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=15,
    metrics=metrics,
    checkpoint_dir="models/",
    is_best=True  # Nếu True, sẽ copy thêm 1 bản lưu đè lên '<model_name>_best.pth'
)
# Output file: models/resnet50_epoch015.pth (và resnet50_best.pth)
```

### Khôi phục mô hình (Load để Train tiếp)

```python
from src.model import load_checkpoint

model, checkpoint = load_checkpoint(
    checkpoint_path="models/resnet50_best.pth",
    num_classes=5,
    device=device,
    freeze_strategy="partial" # Tự động áp dụng strategy sau khi load
)

# Lấy lại trạng thái để train tiếp
start_epoch = checkpoint["epoch"] + 1
optimizer.load_state_dict(checkpoint["optimizer_state"])
print("Best Validation Acc:", checkpoint["metrics"]["val_acc"])
```

### Nạp mô hình để Dự đoán (Inference only)

Dùng hàm rút gọn `load_for_inference` (Tự động set `eval()` và freeze toàn bộ mạng).

```python
from src.model import load_for_inference

model_infer = load_for_inference(
    checkpoint_path="models/mobilenet_v3_best.pth",
    num_classes=5,
    device=device
)

# Chạy trực tiếp, không cần torch.no_grad() thêm
preds = model_infer(image_tensor)
```

---

## 5. Tiện ích và Debug

In thông tin chi tiết về kiến trúc, số lượng tham số, tỉ lệ bị đóng băng.

```python
from src.model import model_summary

# In tóm tắt cấu trúc layer và param count
model_summary(model, input_size=(1, 3, 224, 224))
```

Chạy file dưới dạng script độc lập để test và so sánh 3 kiến trúc:

```powershell
# Chạy smoke test, in thông tin ResNet-50
python src/model.py --model resnet50 --summary

# So sánh số lượng tham số của cả 3 mô hình
python src/model.py --compare_all
```
