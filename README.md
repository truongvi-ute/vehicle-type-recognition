# Nhận Diện Phương Tiện Giao Thông (Vehicle Type Recognition)

Hệ thống phân loại phương tiện giao thông tiên tiến sử dụng 3 kiến trúc Deep Learning: **ResNet-50** (CNN Baseline), **Vision Transformer (ViT)** (Self-Attention toàn cục), và **YOLO-cls** (Tốc độ cao) kết hợp với chiến lược Data Augmentation Offline/Online toàn diện.

## Tổng quan Dataset

**Dataset Vehicle-10** bao gồm **36,006 ảnh** phương tiện được phân loại vào **10 lớp**:

| Lớp | Tên tiếng Việt | Số ảnh Train | Số ảnh Valid | Số ảnh Test |
|-----|----------------|--------------|--------------|-------------|
| bicycle | Xe đạp | ~1,296 | ~144 | ~162 |
| boat | Thuyền/Tàu | ~7,117 | ~790 | ~890 |
| bus | Xe buýt | ~3,252 | ~361 | ~406 |
| car | Ô tô | ~6,832 | ~759 | ~854 |
| helicopter | Trực thăng | ~534 | ~59 | ~67 |
| minibus | Xe khách nhỏ | ~1,181 | ~131 | ~148 |
| motorcycle | Xe mô tô | ~3,550 | ~394 | ~444 |
| taxi | Taxi | ~726 | ~81 | ~91 |
| train | Tàu hỏa | ~1,346 | ~150 | ~169 |
| truck | Xe tải | ~2,971 | ~330 | ~371 |
| **TỔNG** | | **30,605** | **1,800** | **3,601** |

**Tỷ lệ phân chia:**
- **Train:** 85% (30,605 ảnh)
- **Valid:** 5% (1,800 ảnh) — bao gồm cả _unseen (ảnh mới) và _copy (copy từ train)
- **Test:** 10% (3,601 ảnh) — tập đánh giá độc lập hoàn toàn

**Đặc điểm:**
- Ảnh màu RGB, độ phân giải đa dạng từ 47×36px đến 4101×2651px
- Đã được chuẩn hóa về kích thước 224×224px với padding viền đen (giữ nguyên aspect ratio)
- Áp dụng pipeline tiền xử lý `baseline_v1` (Resize + Padding, không filter thêm)

## Cấu trúc thư mục

```
VehicleTypeRecognition/
├── data/
│   ├── raw/                          # Dữ liệu ảnh gốc (36,006 ảnh)
│   ├── splits/                       # Sau split (85%-5%-10%)
│   │   ├── train/                    # 30,605 ảnh
│   │   ├── valid/                    # 1,800 ảnh
│   │   └── test/                     # 3,601 ảnh
│   ├── balanced/                     # Sau cân bằng class (~70K ảnh train)
│   └── augmented/                    # Sau offline augmentation (4 pipelines)
│       ├── train/<class>/            # ~280K+ ảnh (balanced × 4 pipelines)
│       ├── valid/<class>/            # 1,800 ảnh (không augment)
│       └── test/<class>/             # 3,601 ảnh (không augment)
├── models/                           # Trọng số đã train (.pth, .pt)
├── outputs/                          # Confusion matrix, curves, metrics
├── src/
│   ├── data_split.py                 # Split 85%-5%-10% (Split First)
│   ├── balance_classes.py            # Cân bằng class imbalance
│   ├── augment_offline.py            # 4 pipelines: Base/Night/Rain/Sun
│   ├── dataset.py                    # DataLoader + Online Augmentation
│   ├── train.py                      # ResNet-50 training
│   ├── train_vit.py                  # Vision Transformer training
│   ├── train_yolo.py                 # YOLO-cls training
│   └── evaluate.py                   # Metrics, Confusion Matrix, Curves
├── docs/                             # Tài liệu chi tiết
│   ├── DATA_STRATEGY.md              # Chiến lược dữ liệu mới
│   ├── AUGMENTATION.md               # Chi tiết 4 pipelines + online aug
│   ├── MODEL_COMPARISON.md           # So sánh 3 kiến trúc
│   └── ...
├── app.py                            # Giao diện Streamlit
├── requirements.txt
└── README.md
```

## Hướng dẫn cài đặt & Chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Tải dataset (nếu chưa có)
Link dataset gốc: [Google Drive](https://drive.google.com/file/d/1pNmm9RjcdTJVRl8_uv-Cs5-CahkROKHs/view?usp=sharing)

Giải nén vào thư mục `data/raw/`

### 3. Tiền xử lý & Cân bằng dữ liệu
```bash
# Bước 1: Split dataset TRƯỚC (85%-5%-10%)
python src/data_split.py --raw_dir data/raw --output_dir data/splits --seed 42

# Bước 2: Cân bằng class (nhân bản thiểu số lên ~7K)
python src/balance_classes.py --input_dir data/splits/train --output_dir data/balanced

# Bước 3: Augmentation Offline (4 pipelines: Base, Night, Rain, Sun)
python src/augment_offline.py --input_dir data/balanced --output_dir data/augmented --pipelines all

# Kết quả: data/augmented/ chứa ảnh đã cân bằng + augment
```

### 4. Huấn luyện mô hình
```bash
# ResNet-50 (CNN Baseline)
python src/train.py --model resnet50 --data_dir data/augmented --max_epochs 100 --patience 10

# Vision Transformer (Self-Attention)
python src/train_vit.py --model vit_base --data_dir data/augmented --max_epochs 100 --patience 10

# YOLO-cls (Tốc độ cao)
python src/train_yolo.py --model yolov8n-cls --data_dir data/augmented --epochs 100 --patience 10
```

**Output:**
- Trọng số tốt nhất: `models/<model_name>_best.pth`
- Confusion Matrix: `outputs/confusion_matrix_<model>.png`
- Loss/Acc curves: `outputs/training_curves_<model>.png`
- Metrics report: `outputs/metrics_<model>.json` (Precision/Recall/F1)

### 5. Chạy ứng dụng Demo
```bash
streamlit run app.py
```

Trình duyệt sẽ mở tại `http://localhost:8501`

## Tài liệu chi tiết

Xem thư mục `docs/` để tìm hiểu sâu hơn về từng module:

- **[DATA_STRATEGY.md](docs/DATA_STRATEGY.md)** — Chiến lược Split First + Class Balancing + Offline/Online Augmentation
- **[AUGMENTATION.md](docs/AUGMENTATION.md)** — Chi tiết 4 pipelines (Base/Night/Rain/Sun) + MixUp/CutMix/Mosaic
- **[MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)** — So sánh ResNet-50 vs ViT vs YOLO-cls
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** — Huấn luyện 3 kiến trúc, Early Stopping, Metrics
- **[DATASET_INFO.md](docs/DATASET_INFO.md)** — Thông tin chi tiết Vehicle-10 dataset
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** — Hướng dẫn triển khai Streamlit App, tính năng và lỗi thường gặp


## Kiến trúc mô hình

Dự án so sánh 3 kiến trúc Deep Learning tiên tiến:

| Mô hình | Tham số | Cơ chế chính | Vai trò |
|---------|---------|--------------|---------|
| **ResNet-50** | ~24.5M | Skip Connections (CNN) | Baseline ổn định |
| **Vision Transformer (ViT)** | ~86M | Self-Attention toàn cục | Trích xuất bối cảnh global |
| **YOLO-cls** | ~3-7M | CSPDarknet + SPP | Tốc độ suy luận cực nhanh |

**Chiến lược huấn luyện:**
- **Giới hạn:** Tối đa 100 epochs
- **Early Stopping:** Patience = 10 epochs (tự động dừng khi hội tụ)
- **Transfer Learning:** Pre-trained trên ImageNet (ResNet, ViT) / COCO (YOLO)
- **Fine-tuning:** Đa giai đoạn với Learning Rate Scheduling

## Tính năng nổi bật

### 🎯 Chiến lược Dữ liệu
✅ **Split First Protocol** — Chia tách 85%-5%-10% TRƯỚC khi augment (ngăn Data Leakage)  
✅ **Class Balancing** — Nhân bản class thiểu số lên mức boat (~7K ảnh) bằng thuật toán $K = \lceil N_{target} / N_{current} \rceil$  
✅ **4 Pipeline Offline** — Base, Night, Rain, Sun + biến đổi hình học (sinh file vật lý)  
✅ **Online Augmentation** — MixUp + CutMix (ResNet/ViT), Mosaic (YOLO)

### 🤖 Huấn luyện & Đánh giá
✅ **3 Kiến trúc** — ResNet-50 (CNN), ViT (Transformer), YOLO-cls (Speed)  
✅ **Early Stopping** — Patience=10, tối đa 100 epochs  
✅ **Confusion Matrix** — Biểu đồ nhiệt 10×10 tự động  
✅ **Metrics đầy đủ** — Precision, Recall, F1-Score cho từng class  
✅ **Loss/Acc Curves** — Trực quan hóa quá trình training

### 🚀 Triển khai
✅ **Smart Processing** — Manifest tracking, chỉ xử lý ảnh thay đổi  
✅ **Giao diện Streamlit** — Demo real-time với upload ảnh  
✅ **Multi-model Support** — Chuyển đổi giữa 3 kiến trúc dễ dàng

## Công nghệ sử dụng

- **Deep Learning:** PyTorch, torchvision, timm (ViT), ultralytics (YOLO)
- **Computer Vision:** OpenCV (cv2), Albumentations, PIL
- **Data Science:** NumPy, pandas, scikit-learn
- **Augmentation:** Albumentations (Offline), Mixup/CutMix (Online)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web App:** Streamlit
- **Metrics:** sklearn.metrics (Precision/Recall/F1, Confusion Matrix)

## Liên hệ & Đóng góp

Dự án được phát triển cho môn Xử lý Ảnh Số. Mọi góp ý và cải tiến xin gửi qua Issues hoặc Pull Requests.

---

**Cập nhật:** 26/05/2026 | **Phiên bản:** 2.0
