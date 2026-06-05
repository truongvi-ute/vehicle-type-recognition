# 📊 Thông tin Bộ dữ liệu & Tổng quan Dự án (Dataset Info & Project Overview)

> [!NOTE]  
> Tài liệu này mô tả chi tiết bài toán, cấu trúc thư mục dự án và phân tích thách thức mất cân bằng dữ liệu của bộ dữ liệu **Vehicle-10**.

---

## 1. Mô tả Bài toán (Problem Description)

Dự án tập trung xây dựng hệ thống phân loại hình ảnh (**Image Classification**) nhằm nhận diện chính xác 10 loại phương tiện giao thông khác nhau từ bộ dữ liệu **Vehicle-10** gồm **36,006 ảnh** màu RGB.

Hệ thống được thiết kế để xử lý triệt để hai bài toán thực tế lớn trong lĩnh vực Thị giác máy tính (Computer Vision):
1. **Hiện tượng mất cân bằng lớp (Class Imbalance)**: Sự chênh lệch lớn về số lượng mẫu giữa các lớp phương tiện.
2. **Rủi ro rò rỉ dữ liệu (Data Leakage)**: Ngăn chặn triệt để việc dữ liệu của tập validation/test bị rò rỉ vào tập train thông qua các bước tiền xử lý hoặc tăng cường dữ liệu (augmentation).

---

## 2. Cấu trúc Cây Thư Mục Dự Án Hiện Tại (Project Directory Structure)

Cây thư mục dưới đây thể hiện cách tổ chức mã nguồn và các phân đoạn dữ liệu trong quá trình thực thi:

```plaintext
VehicleTypeRecognition/
├── data/
│   ├── raw/                          # Bộ dữ liệu ảnh gốc ban đầu (36,006 ảnh)
│   ├── splits/                       # Phân chia dữ liệu theo giao thức Split First
│   ├── balanced/                     # Tập Train sau khi chạy cân bằng class K-formula
│   └── augmented/                    # Tập dữ liệu vật lý hoàn chỉnh sau Offline Augmentation
│       ├── train/                    # 284,680 ảnh (71,170 ảnh gốc x 4 pipelines)
│       ├── valid/                    # 1,800 ảnh chuẩn hóa (Không augmentation)
│       └── test/                     # 3,601 ảnh chuẩn hóa (Không augmentation)
├── src/
│   ├── data_split.py                 # Bước 1: Chia tập dữ liệu Train/Valid/Test gốc
│   ├── balance_classes.py            # Bước 2: Đồng đến số lượng mẫu giữa các class
│   ├── augment_offline.py            # Bước 3: Áp dụng 4 pipelines OpenCV vật lý
│   ├── dataset.py                    # PyTorch DataLoader + Online Transform
│   ├── train.py                      # Vòng lặp huấn luyện cho ResNet-50
│   ├── train_vit.py                  # Vòng lặp huấn luyện cho Vision Transformer
│   ├── train_yolo.py                 # Script huấn luyện phân loại YOLO-cls
│   └── evaluate.py                   # Đánh giá metrics độc lập trên tập dữ liệu Test
├── models/                           # Thư mục lưu trữ các file trọng số (*.pth, *.pt)
├── outputs/                          # Lưu ma trận nhầm lẫn, biểu đồ Loss/Acc và JSON log
└── app.py                            # Giao diện web demo thời gian thực bằng Streamlit
```

---

## 3. Phân Phối Dữ Liệu Gốc & Thách Thức Mất Cân Bằng (Data Distribution & Imbalance Challenge)

Bộ dữ liệu **Vehicle-10** thể hiện sự chênh lệch số lượng mẫu cực kỳ lớn giữa các lớp phương tiện giao thông, với tỷ lệ chênh lệch giữa lớp nhiều nhất và ít nhất lên tới **13.3:1**.

### Bảng phân phối chi tiết dữ liệu gốc:

| Thứ tự | Tên Lớp (Class) | Số lượng ảnh | Tỷ lệ phần trăm | Mức độ phổ biến |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **boat** (Lớp tối đa) | 7,117 | 100.0% | Lớp đa số (Majority Class) |
| 2 | **car** | 6,832 | 96.0% | Rất phổ biến |
| 3 | **motorcycle** | 3,550 | 50.0% | Trung bình |
| 4 | **bus** | 3,252 | 46.0% | Trung bình |
| 5 | **truck** | 2,971 | 42.0% | Trung bình |
| 6 | **train** | 1,346 | 19.0% | Ít |
| 7 | **bicycle** | 1,296 | 18.0% | Ít |
| 8 | **minibus** | 1,181 | 17.0% | Ít |
| 9 | **taxi** | 726 | 10.0% | Thiểu số (Minority Class) |
| 10 | **helicopter** (Lớp tối thiểu) | 534 | 7.5% | Lớp thiểu số nghiêm trọng |

> [!WARNING]  
> **Thách thức mất cân bằng lớp (Class Imbalance)**:  
> Nếu huấn luyện trực tiếp trên phân phối này, mô hình sẽ có xu hướng tối ưu hóa độ chính xác trên các lớp đa số như `boat` và `car`, dẫn đến hiệu suất dự đoán kém (nhầm lẫn nhiều) ở các lớp thiểu số như `helicopter` hay `taxi`. Do đó, cần áp dụng chiến lược cân bằng dữ liệu hiệu quả trước khi huấn luyện.
