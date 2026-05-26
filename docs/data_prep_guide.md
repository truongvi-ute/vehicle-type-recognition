# Nhận Dạng Phương Tiện Giao Thông
## Vehicle Type Recognition — Tài liệu kỹ thuật

---

## 1. Tổng quan dự án

Hệ thống **Phân loại hình ảnh (Image Classification)** nhận diện **5–7 loại phương tiện giao thông** (bike, motorbike, car, bus, truck…) từ ảnh màu RGB.

| Thông số | Giá trị |
|---|---|
| Số lớp phân loại | 5–7 lớp |
| Số ảnh mỗi lớp | 1.500–2.000 ảnh |
| Tổng dataset | ~7.500–14.000 ảnh RGB |
| Kích thước đầu vào CNN | 224 × 224 px |
| Phương pháp | Transfer Learning |

---

## 2. Cấu trúc thư mục

```
VehicleTypeRecognition/
├── data/
│   ├── raw/                        # Ảnh thô (1500-2000 ảnh/lớp)
│   └── processed/                  # Ảnh đã xử lý (Resize, Filter, Split)
│       ├── train/<class>/
│       ├── valid/<class>/          # Unseen (5%) + Copy từ Train (5%)
│       ├── test/<class>/
│       └── .pipeline_manifest.json # Manifest theo dõi trạng thái pipeline
├── src/
│   ├── data_prep.py                # Pipeline tiền xử lý & phân chia dữ liệu
│   ├── dataset.py                  # PyTorch DataLoader
│   ├── model.py                    # Khởi tạo 3 kiến trúc CNN
│   └── train.py                    # Vòng lặp huấn luyện
├── models/                         # Trọng số tốt nhất (.pth / .h5)
└── app.py                          # Giao diện Streamlit thời gian thực
```

---

## 3. Pipeline tiền xử lý (`src/data_prep.py`)

### 3.1 Chuẩn hóa hình học

**Resize + Padding viền đen** — bảo toàn Aspect Ratio trước khi đưa vào CNN.

```
Ảnh gốc (W×H bất kỳ)
  → Scale cạnh dài nhất về 224px
  → Pad viền đen hai bên (không crop, không stretch)
  → Output: 224×224×3
```

> Tránh hiện tượng méo hình học khi xe bus/xe tải bị ép thành hình vuông.

---

### 3.2 Chuỗi Tiền Xử Lý Không Gian (Pipeline Chains)

Hệ thống cho phép kết hợp tuần tự nhiều bộ lọc để giải quyết các vấn đề phức tạp trên ảnh. Toàn bộ chuỗi được áp dụng trực tiếp trên từng kênh R, G, B — không dùng FFT.

Định nghĩa sẵn các chuỗi pipeline (A/B Testing):
1. `baseline_v1`: Giữ nguyên (chỉ chuẩn hóa kích thước).
2. `denoise_sharpen_v1`: Lọc Gaussian khử nhiễu nhẹ -> Highboost làm sắc nét cạnh.
3. `night_vision_v1`: Alpha-trimmed mean khử nhiễu hỗn hợp hạt/đốm ban đêm -> Unsharp Masking tái tạo độ nét khối.
4. `heavy_rescue_v1`: Contraharmonic khử sương/nhiễu muối tiêu -> Median làm sạch -> Laplacian làm nổi bật viền cơ cấu.

**Các bộ lọc cấu thành:**

#### Nhóm A — Làm trơn / Lowpass

| Tên hàm | Bộ lọc | Mô tả |
|---|---|---|
| `apply_box_filter` | Box / Averaging | Trung bình đều, làm mờ nhanh |
| `apply_gaussian_filter` | Gaussian | Trọng số Gaussian, làm mịn tự nhiên, bảo toàn biên tốt hơn Box |

#### Nhóm B — Trung bình khử nhiễu

| Tên hàm | Bộ lọc | Phù hợp với |
|---|---|---|
| `apply_arithmetic_mean_filter` | Arithmetic Mean | Nhiễu phân tán đều |
| `apply_geometric_mean_filter` | Geometric Mean | Giữ chi tiết tinh tế hơn AM |
| `apply_harmonic_mean_filter` | Harmonic Mean | Nhiễu muối (salt noise) |
| `apply_contraharmonic_mean_filter` | Contraharmonic (Q) | Q>0: tiêu; Q<0: muối |

#### Nhóm C — Thống kê thứ tự / Phi tuyến

| Tên hàm | Bộ lọc | Phù hợp với |
|---|---|---|
| `apply_median_filter` | Median | Nhiễu muối-tiêu (khuyến nghị) |
| `apply_max_filter` | Max (Dilation) | Khử nhiễu tiêu |
| `apply_min_filter` | Min (Erosion) | Khử nhiễu muối |
| `apply_midpoint_filter` | Midpoint | Nhiễu phân phối đồng đều |
| `apply_alpha_trimmed_mean_filter` | Alpha-Trimmed Mean | Hỗn hợp nhiều loại nhiễu |

#### Nhóm D — Làm sắc nét / Highpass

| Tên hàm | Bộ lọc | Mô tả |
|---|---|---|
| `apply_laplacian_filter` | Laplacian | Đạo hàm bậc 2, biên mọi hướng |
| `apply_sobel_gradient` | Sobel (x/y/both) | Đạo hàm bậc 1, hướng ngang & dọc |
| `apply_scharr_gradient` | Scharr | Chính xác hơn Sobel tại biên chéo |
| `apply_roberts_gradient` | Roberts | Biên 45°/135°, nhanh & đơn giản |
| `apply_unsharp_masking` | Unsharp Masking | Làm sắc nét bằng mặt nạ biên |
| `apply_highboost_filter` | Highboost (A) | Tăng cường cao hơn Unsharp (A>1) |

#### Gọi bộ lọc qua API

```python
from src.data_prep import apply_filter, preprocess_image

# Áp dụng 1 bộ lọc
img = apply_filter(img_bgr, "gaussian", ksize=5, sigma=1.0)

# Pipeline hoàn chỉnh theo chuỗi đã định nghĩa
img_rgb = preprocess_image("path/to/image.jpg", pipeline_name="denoise_sharpen_v1")
```

---

### 3.3 Custom Split Data (Theo yêu cầu Giảng viên)

```
Tổng N ảnh/lớp
  ├── Test           10%  — Tách biệt hoàn toàn, không bao giờ vào Train/Valid
  └── Còn lại 90%
       ├── Valid Unseen  5%  — Ảnh hoàn toàn MỚI (chưa xuất hiện ở đâu)
       └── Pool (85%)
            ├── Train       80% tổng  — Nguồn học chính
            └── Valid Copy   5% tổng  — COPY ngẫu nhiên từ Train (yêu cầu GV)
```

> **Lưu ý khoa học:** Valid Copy tạo ra Data Leakage có chủ đích theo yêu cầu GV.
> Tập **Test (10%)** mới là thước đo khách quan thực sự của mô hình.

| Tập | Tỷ lệ | Nguồn | Vai trò |
|---|---|---|---|
| **Test** | 10% | Từ tổng dataset | Đánh giá khách quan cuối cùng |
| **Train** | 80% | Phần còn lại | Nguồn học chính |
| **Valid – Unseen** | 5% | Ảnh mới hoàn toàn | Kiểm tra Overfitting |
| **Valid – Copy** | 5% | Copy từ Train | So sánh sai số cũ/mới |

---

### 3.4 Hệ thống Cleanup & Smart Overwrite *(cập nhật mới)*

File manifest `.pipeline_manifest.json` được tạo tự động trong `processed/`, theo dõi trạng thái từng ảnh.

#### 3 chế độ chạy

```powershell
# SMART (mặc định): thông minh, bỏ qua ảnh đã xử lý
python src/data_prep.py --mode smart

# FORCE: xóa sạch processed/ và chạy lại từ đầu
python src/data_prep.py --mode force

# CLEAN: chỉ dọn orphan (ảnh raw đã bị xóa), không xử lý ảnh mới
python src/data_prep.py --mode clean

# Xem trạng thái manifest hiện tại
python src/data_prep.py --show_manifest
```

#### Bảng xử lý theo tình huống (Smart Mode)

| Tình huống | Hành động |
|---|---|
| Chạy lần đầu | Xử lý **tất cả** ảnh |
| Chạy lại, không đổi gì | **Bỏ qua** 100% (tiết kiệm thời gian) |
| Thêm ảnh mới vào `raw/` | Chỉ xử lý ảnh **mới** |
| Xóa ảnh trong `raw/` | **Dọn orphan** trong `processed/` tự động |
| Ảnh raw bị chỉnh sửa | Phát hiện **MD5 khác**, tái xử lý ảnh đó |
| Đổi `--pipeline` hoặc `--target_size` | Cảnh báo, **reset manifest**, tái xử lý toàn bộ |

#### Cấu trúc manifest JSON

```json
{
  "_meta": {
    "pipeline": "baseline_v1",
    "target_size": 224,
    "seed": 42,
    "created_at": "2026-05-25T...",
    "updated_at": "2026-05-25T..."
  },
  "files": {
    "Bus/bus_000001.jpg": {
      "md5": "a3f8...",
      "split": "train",
      "out_name": "bus_000001.jpg",
      "processed_at": "2026-05-25T..."
    }
  }
}
```

---

## 4. Đối sánh kiến trúc mô hình

Transfer Learning trên 3 CNN backbone (PyTorch `torchvision.models`):

| Mô hình | Đặc điểm | Vai trò |
|---|---|---|
| **ResNet-50** | Skip Connections, độ chính xác cao | Baseline chuẩn |
| **MobileNet-V3** | Depthwise Separable Conv, siêu nhẹ | Triển khai thực tế |
| **EfficientNet-B0** | Compound Scaling đa chiều | Cân bằng hiệu năng/tài nguyên |

---

## 5. Ứng dụng Streamlit (`app.py`)

```
Upload ảnh → Resize+Padding → Bộ lọc không gian → CNN → Kết quả + Confidence
```

- Upload ảnh trực tiếp trên giao diện web
- Chọn bộ lọc và điều chỉnh tham số bằng slider
- Hiển thị ảnh gốc vs ảnh đã lọc song song
- Xuất nhãn phương tiện + điểm tin cậy (Confidence Score) thời gian thực

---

## 6. Hướng dẫn cài đặt & chạy

```powershell
# Cài thư viện
pip install opencv-python numpy scikit-learn tqdm python-docx

# Chạy pipeline tiền xử lý (lần đầu)
python src/data_prep.py --pipeline baseline_v1 --mode smart

# Xem trạng thái dữ liệu
python src/data_prep.py --show_manifest

# Chạy ứng dụng Streamlit
streamlit run app.py
```

---

*Tài liệu cập nhật: 2026-05-25 | Phiên bản: 2.0*
