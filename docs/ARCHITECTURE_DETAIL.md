# Chi Tiết Tiền Xử Lý, Huấn Luyện & Bản Đồ Kiến Trúc (Technical Specs & Architecture Map)

Tài liệu này cung cấp các thông số kỹ thuật chi tiết của giai đoạn tiền xử lý, cấu hình huấn luyện và bản đồ kết nối các tệp tin trong dự án để giúp bạn nắm bắt toàn bộ kiến trúc hệ thống một cách dễ dàng và dễ hiểu nhất.

---

## 1. Chi Tiết Giai Đoạn Tiền Xử Lý (Image Preprocessing & Augmentations)

Tất cả các bộ tiền xử lý được thiết kế độc lập dưới dạng các module trong thư mục `src/image_pipelines/`.

### 1.1. Bộ Tiền xử lý Cơ bản (Base Pipeline) - File: `src/image_pipelines/base.py`
Mọi hình ảnh đầu vào (huấn luyện, đánh giá, dự đoán thực tế) đều bắt buộc đi qua luồng này:
1. **Ép chuẩn màu:** Chuyển đổi ảnh về hệ màu **RGB** 3 kênh màu (loại bỏ kênh Alpha nếu có).
2. **Resize giữ tỷ lệ (Aspect Ratio):** Sử dụng phép nội suy chất lượng cao **PIL Lanczos Resampling** để thay đổi kích thước ảnh theo cạnh lớn nhất sao cho phương tiện không bị méo.
3. **Bù viền đen (Zero-padding):** Dán ảnh đã resize vào chính giữa một khung vẽ đen `(0, 0, 0)` kích thước chuẩn **224x224** pixel.

### 1.2. Các thuật toán tạo ảnh nhiễu ngoại tuyến (Offline Augmentation)

#### Nhóm V1 - Mô phỏng Thời tiết (Weather)
* **Rain (Mưa) - File: `rain.py`:**
  * Vẽ **120 vệt mưa** ngẫu nhiên dạng đoạn thẳng nghiêng nhẹ sang phải (độ dài 5-12px, độ nghiêng xiên tối đa `offset // 4`). Màu vệt mưa được phối hòa sắc (pixel max với độ sáng màu xám `[190, 190, 190]`).
  * Giảm độ phơi sáng tổng thể (nhân hệ số 0.88, trừ đi 8 đơn vị độ sáng).
  * Áp dụng làm mờ **Gaussian Blur bán kính 0.6** bằng thư viện PIL để làm mờ nhẹ các nét vẽ, tạo cảm giác mưa chân thực.
* **Sun (Nắng lóa) - File: `sun.py`:**
  * Mô phỏng vầng sáng mặt trời: Tâm nắng ngẫu nhiên nằm ở nửa trên bức ảnh (trục X trong khoảng `[46, 178]`, trục Y trong `[32, 126]`).
  * Bán kính tán sáng ngẫu nhiên `[36, 64]` pixel.
  * Màu sắc nguồn sáng: Màu vàng ấm dịu `[255, 238, 190]`.
  * Cường độ sáng suy giảm tuyến tính từ tâm ra rìa theo tỷ lệ `1 - distance/radius`. Tăng độ sáng tổng thể của ảnh lên `* 1.08 + 8`.
* **Night (Đêm tối) - File: `night.py`:**
  * Chuẩn hóa pixel về miền số thực `[0.0, 1.0]`.
  * Áp dụng **hàm số mũ (Gamma Correction) lũy thừa 1.75** (`array^1.75`) để làm tối sâu các vùng trung tính và vùng tối.
  * Lệch màu sang tông lạnh (Giảm sắc Đỏ còn 72%, sắc Xanh lá 78%, giữ sắc Xanh lam ở 95%).
  * Sinh **nhiễu muối tiêu hạt mịn (Gaussian Noise)** với độ lệch chuẩn `sigma = 0.025` mô phỏng nhiễu ISO của camera giám sát ban đêm.

#### Nhóm V2 - Làm mờ và Làm nét (Blur/Sharpen)
* **Gaussian Blur (Làm mờ đều) - File: `gaussian_blur.py`:**
  * Chọn ngẫu nhiên kích thước bộ lọc (Kernel size) `3x3` hoặc `5x5`.
  * Chọn ngẫu nhiên độ lệch chuẩn `sigma` trong khoảng `[0.4, 1.4]`.
  * Áp dụng hàm `cv2.GaussianBlur` của OpenCV với chế độ phản xạ đường viền `cv2.BORDER_REFLECT_101` tránh viền đen.
* **Motion Blur (Làm mờ chuyển động) - File: `motion_blur.py`:**
  * Mô phỏng chuyển động của phương tiện: Chọn độ dài vệt mờ (Kernel length) ngẫu nhiên từ `(3, 5, 7)` pixel.
  * Chọn ngẫu nhiên góc chuyển động từ `0.0` đến `180.0` độ.
  * Vẽ đường thẳng mô phỏng hướng chuyển động chống răng cưa (`cv2.LINE_AA`) tạo ma trận lọc, sau đó lọc ảnh bằng `cv2.filter2D`.
* **Unsharp Mask (Làm nét ảnh) - File: `unsharp_mask.py`:**
  * Chọn kích thước kernel `3x3` hoặc `5x5`, `sigma` từ `[0.5, 1.2]`, cường độ làm sắc nét (amount) từ `[0.3, 0.8]`, ngưỡng nhiễu tối thiểu (threshold) từ `[0, 5]`.
  * Lấy ảnh gốc trừ đi ảnh đã làm mờ Gaussian để lọc ra chi tiết cạnh (detail).
  * Lọc nhiễu: Chỉ tăng cường chi tiết nếu độ lệch vượt quá ngưỡng threshold, cộng ngược lại chi tiết vào ảnh gốc: `Sharpened = Original + amount * Detail`.

---

## 2. Chi Tiết Cấu Hình Huấn Luyện (Training Strategy & Specs)

Quá trình huấn luyện được thực hiện bằng 2 file kịch bản chính gửi lên Kaggle.

### 2.1. Đối với ResNet-50 và ViT-B/16 - File: `src/train.py`
Mô hình huấn luyện trải qua chiến lược **Hai Giai Đoạn (2-Phase Training)** để tối ưu hóa Transfer Learning:
* **Giai đoạn 1 (Warm-up Head - 5 Epochs đầu):**
  * Khóa băng (Freeze) toàn bộ các tầng trích xuất đặc trưng của xương sống mô hình (Backbone).
  * Chỉ cập nhật trọng số cho tầng phân loại mới (Classification Head) với tốc độ học cao `lr_head = 1e-3` nhằm ổn định bộ phân loại.
* **Giai đoạn 2 (Fine-tuning - 25 Epochs sau):**
  * Mở băng (Unfreeze) toàn bộ mô hình.
  * Huấn luyện đồng thời với **tốc độ học phân biệt (Differential Learning Rates)**: Tầng phân loại chạy với `lr_head = 1e-3`, các tầng xương sống (Backbone) chạy với tốc độ học siêu nhỏ `lr_backbone = 1e-5` để giữ lại các đặc trưng tổng quát đã học từ ImageNet.
* **Cơ chế dừng sớm (Early Stopping):** Theo dõi độc quyền chỉ số Loss của tập độc lập `valid_unseen`. Nếu Loss không cải thiện sau **7 Epochs** liên tiếp (Patience = 7), quá trình train sẽ tự dừng và lưu checkpoint tốt nhất (`best.pth`).

### 2.2. Đối với YOLOv8n-cls - File: `src/train_yolo.py`
Do YOLO quản lý bởi thư viện Ultralytics, nhóm thiết kế một lớp trung gian (Adapter) tùy chỉnh:
* **Symlink Adapter:** Bản đồ hóa cấu trúc thư mục của nhóm thành cấu trúc YOLO hiểu được (`val` trỏ đến `valid_unseen`).
* **Vô hiệu hóa Online Augmentations mặc định:** 
  * Định nghĩa lớp `FullImageClassificationDataset` kế thừa từ YOLO để bỏ bộ cắt ảnh ngẫu nhiên mặc định (`RandomResizedCrop`) và các phép lật xoay trực tuyến khác. 
  * Ép mô hình chỉ sử dụng phép Resize thông thường về `224x224` để bảo vệ các viền đen Zero-pad đã tiền xử lý offline, giúp so sánh hoàn toàn sòng phẳng với ResNet/ViT.
* **Đồng bộ hóa Chuẩn hóa (ImageNet Normalization):** Bản đồ hóa phân phối dữ liệu đầu vào bằng phép chuẩn hóa ImageNet Mean `[0.485, 0.456, 0.406]` và Std `[0.229, 0.224, 0.225]` thống nhất với ResNet-50 và ViT thay vì giữ nguyên phân phối thô `[0, 1]`.
* **Thông số cấu hình & Bộ tối ưu:** Ép mô hình sử dụng bộ tối ưu **AdamW** (thay cho SGD mặc định của YOLO) với Learning Rate cơ sở `lr0 = 1e-3`, `warmup_epochs = 5` và `mixup = 0.2` nhằm đồng bộ hóa chiến lược huấn luyện và tăng cường online. Train tối đa 30 Epochs, Patience dừng sớm = 7 (đồng bộ với ResNet/ViT), Batch size = 128. Điểm checkpoint được chọn dựa trên độ chính xác Validation Top-1 cao nhất (`best.pt`).

---

## 3. Bản Đồ Truy Xuất Thông Tin Kiến Trúc (Architecture File Map)

Để dễ dàng nắm bắt kiến trúc dự án, dưới đây là sơ đồ chỉ rõ thông tin được Flask Backend và React Dashboard đọc từ các tệp nào:

```
[Màn hình Dashboard] (Vẽ biểu đồ, so sánh hiệu năng 12 phiên bản)
       │
       ▼ (Đọc qua API /api/metrics/experiments)
[Tệp mục lục: outputs/experiment_registry.json] (Định nghĩa ánh xạ ID mô hình với đường dẫn file)
       │
       ├─► [Thông số chi tiết: outputs/<model_run>/metrics_<model>.json]
       │   (Chứa các thông tin tóm tắt: Epoch tốt nhất, Độ chính xác Best, Loss Best)
       │
       ├─► [Biến thiên đồ thị: outputs/<model_run>/history_<model>.json]
       │   (Chứa mảng Loss, Accuracy qua từng epoch phục vụ vẽ biểu đồ đường Line Chart)
       │
       ├─► [Chi tiết lớp học: outputs/<model_run>/evaluation_<model>_best.json]
       │   (Chứa báo cáo F1-score từng lớp, ma trận nhầm lẫn Confusion Matrix dạng mảng số)
       │
       └─► [Cơ cấu tập dữ liệu: outputs/data_<dataset>_<version>_counts.json]
           (Chứa số lượng ảnh thực tế của tập train/val/test phục vụ vẽ biểu đồ phân phối mẫu)
```

### 3.1. Danh sách tệp tin cấu cấu hình Dashboard quan trọng:
* **Tệp Đăng ký Thí nghiệm:** `outputs/experiment_registry.json`
  * Nếu bạn thêm một mô hình mới, bạn phải khai báo ID mô hình, loại dataset, phiên bản (v1/v2) và đường dẫn đến các file metrics tương ứng tại đây.
* **Thông số Phân phối Dữ liệu:**
  * `outputs/data_raw_original_v1_counts.json` (Số lượng ảnh gốc V1)
  * `outputs/data_raw_original_v2_counts.json` (Số lượng ảnh gốc V2)
  * `outputs/data_cleaning_v1_counts.json` (Số lượng ảnh đã lọc nhiễu V1)
  * `outputs/data_cleaning_v2_counts.json` (Số lượng ảnh đã lọc nhiễu V2)

### 3.2. Bản đồ dịch vụ suy luận trực tuyến (Inference File Map):
```
[React UI] 
   │ (Gửi POST /api/predict)
   ▼
[backend/routes/predict.py] (Nhận tệp ảnh, đọc tham số Model & Pipeline gửi từ client)
   │
   ▼ (Gọi hàm predict_image)
[backend/services/inference_service.py] 
   │
   ├─► [backend/services/model_loader.py]
   │   (Nạp động checkpoint PyTorch .pth hoặc YOLO .pt tương ứng từ models/)
   │
   ├─► [backend/utils/preprocessing.py]
   │   (Mở ảnh PIL, gọi các bộ lọc trong src/image_pipelines/ để xử lý ảnh)
   │
   └─► Chạy Forward Pass trên GPU/CPU, đóng gói JSON kết quả trả về UI.
```
