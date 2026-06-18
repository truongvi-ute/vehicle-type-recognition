# ĐỀ XUẤT PHƯƠNG ÁN CẢI TIẾN HỆ THỐNG NHẬN DẠNG PHƯƠNG TIỆN GIAO THÔNG (VEHICLE-10)
*Phiên bản chốt cải tiến: Trích xuất đặc trưng đa quy mô, Đệm gương (Reflective Padding) & Class-Balanced Focal Loss trên tập Clean V2*

---

## I. BỐI CẢNH VÀ SUY LUẬN SAI SỐ ĐỊNH LƯỢNG

Dựa trên số liệu thống kê sai số thực tế từ các mô hình baseline (ResNet-50, ViT, YOLOv8) trên tập dữ liệu ban đầu, nhóm nghiên cứu đã bóc tách được 3 điểm nghẽn nghiêm trọng:
1. **Hiện tượng tiêu biến đặc trưng mào xe taxi**: Do ResNet-50 downsampling ảnh 224x224 về feature map cuối cùng 7x7 (stride 32), đặc trưng mào xe (chiếm dưới 2% diện tích ảnh) bị hòa tan hoàn toàn, dẫn đến YOLO nhầm 36.3% và ResNet nhầm 26.4% taxi sang ô tô con (car).
2. **Hiện tượng mịn hóa quá mức CLS Token (Over-smoothing)**: Ở Vision Transformer (ViT), CLS token tại lớp cuối cùng bị đồng nhất hóa, làm mất thông tin biên dạng và tỷ lệ dài/rộng của xe, khiến F1-score của lớp Minibus bị kẹt ở mức thấp (~80.6%).
3. **Nhiễu tần số cao tại biên ảnh do Zero-Padding**: Việc bù viền đen (zero-pad) tạo ra sự đứt gãy tương phản đột ngột giữa ảnh thực và viền đen, tạo các đường cạnh giả làm rối loạn bộ lọc tích chập ở các tầng nông.
4. **Sự mất cân bằng mẫu độc bản**: Lớp thiểu số như `helicopter` (568 mẫu) và `taxi` (765 mẫu) có mức độ trùng lặp cao sau khi sinh batch offline để cân bằng số lượng với `boat` (7,390 mẫu). Điều này làm lệch không gian vector tối ưu hóa về phía lớp đa số.

---

## II. ĐẶC TẢ PHƯƠNG ÁN CẢI TIẾN CHỐT

### 1. Tiền xử lý: Đệm gương đối xứng (Reflective Padding)
Thay thế toàn bộ zero-padding bằng đệm gương đối xứng (Reflective Padding) khi đưa ảnh về kích thước chuẩn 224x224. Phương pháp này lấy đối xứng phần biên ảnh gốc để lấp đầy các biên trống, giúp triệt tiêu nhiễu tần số cao và đảm bảo tính liên tục của phân phối kích hoạt lớp biên dạng.

### 2. Kiến trúc: Trích xuất đặc trưng đa quy mô và đa lớp
* **Đối với ResNet-50**: Thiết kế Wrapper `MultiScaleResNet50` trích xuất song song đặc trưng từ Layer 3 (14x14, 1024 kênh - giữ chi tiết cục bộ tần số cao) và Layer 4 (7x7, 2048 kênh - giữ ngữ nghĩa toàn cục). Chạy Global Average Pooling trên từng layer rồi ghép nối (concatenate) thành vector 3072 chiều đưa vào bộ phân loại.
* **Đối với Vision Transformer (ViT)**: Thiết kế Wrapper `MultiLayerCLSViT` sử dụng PyTorch Forward Hook để trích xuất token CLS từ 3 block encoder cuối cùng (Layer 10, Layer 11, Layer 12), ghép nối chúng thành vector 2304 chiều trước khi đưa vào classifier, bảo toàn cấu trúc hình thái của xe khách minibus.

### 3. Tối ưu hóa: Hàm lỗi Class-Balanced Focal Loss
Thay thế Cross-Entropy truyền thống bằng Class-Balanced Focal Loss với công thức:
\[L(\text{CB-Focal}) = - W_y \cdot (1 - p_y)^\gamma \cdot \log(p_y)\]

Trong đó, hệ số trọng số \(W_i\) được tính nghịch đảo theo số lượng mẫu thực tế độc bản gốc trong tập train:
\[W_i = \frac{1 - \beta}{1 - \beta^{n_i}}\]

Với tham số \(\beta = 0.999\) và \(\gamma = 2.0\). Trọng số này được chuẩn hóa để tổng bằng 10 (số lượng lớp). Mẫu số lượng gốc \(n_i\) được định cấu hình chính xác theo phân phối thực tế của project:

| Chỉ số lớp | Tên Lớp (Class) | Số mẫu độc bản gốc (\(n_i\)) |
| :---: | :--- | :---: |
| 0 | bicycle | 1,334 |
| 1 | boat | 7,390 |
| 2 | bus | 3,395 |
| 3 | car | 7,111 |
| 4 | helicopter | 568 |
| 5 | minibus | 1,251 |
| 6 | motorcycle | 3,703 |
| 7 | taxi | 765 |
| 8 | train | 1,403 |
| 9 | truck | 3,125 |

---

## III. BẢNG ĐỐI CHIẾU PHƯƠNG ÁN CŨ VÀ CẢI TIẾN

Dưới đây là các bảng đối chiếu chi tiết giữa kiến trúc Baseline (Phiên bản cũ) và kiến trúc Cải tiến đề xuất (Phiên bản mới) nhằm làm nổi bật sự khác biệt về giải pháp kỹ thuật cũng như hiệu năng thực nghiệm.

### 1. Bảng đối chiếu giải pháp kỹ thuật và thuật toán

| Thành phần so sánh | Phiên bản cũ (Baseline / V1) | Phiên bản mới cải tiến (V2 đề xuất) |
| :--- | :--- | :--- |
| **Tiền xử lý biên ảnh** *(Image Preprocessing)* | Sử dụng **Zero-Padding** (đệm viền đen) khi resize ảnh về 224x224.<br><br>**Hạn chế**: Tạo độ đứt gãy tương phản đột ngột ở rìa ảnh, sinh ra các đường biên giả (nhiễu tần số cao) gây nhiễu cho các lớp tích chập nông. | Sử dụng **Reflective Padding** (đệm gương đối xứng biên).<br><br>**Ưu điểm**: Lấy đối xứng vùng biên ảnh để điền đầy khoảng trống, triệt tiêu nhiễu tần số cao, duy trì tính liên tục và trơn tru của phân phối không gian. |
| **Trích xuất đặc trưng ResNet-50** *(Feature Extraction)* | Đặc trưng **đơn quy mô (Single-scale)**. Chỉ lấy output từ Layer 4 cuối cùng (7x7, 2048 kênh) qua Global Average Pooling (GAP).<br><br>**Hạn chế**: Đặc trưng hạt mịn, kích thước nhỏ (như mào xe taxi) bị tiêu biến hoàn toàn sau nhiều lớp pooling. | Đặc trưng **đa quy mô (Multi-scale fusion)**. Ghép nối song song đặc trưng cục bộ Layer 3 (14x14, 1024 kênh) và đặc trưng ngữ nghĩa Layer 4 (7x7, 2048 kênh) thành vector 3072 chiều.<br><br>**Ưu điểm**: Bảo toàn các đặc trưng nhỏ nhạy cảm. |
| **Trích xuất đặc trưng ViT** *(Feature Extraction)* | Sử dụng **CLS Token duy nhất** của block cuối cùng (Layer 12, 768 kênh) để phân loại.<br><br>**Hạn chế**: CLS token bị hiện tượng mịn hóa quá mức (Over-smoothing) qua 12 lớp self-attention, làm mất chi tiết hình học đặc thù của minibus. | Sử dụng **Multi-Layer CLS Fusion**. Trích xuất và ghép nối CLS token từ cả 3 block cuối cùng (Layer 10, 11 và 12) thành vector 2304 chiều.<br><br>**Ưu điểm**: Giữ nguyên thông tin hình học và hình thái xe khách. |
| **Hàm tối ưu hóa** *(Loss Function)* | **Cross-Entropy Loss** tiêu chuẩn.<br><br>**Hạn chế**: Phân phối lỗi đồng đều cho mọi mẫu, khiến mô hình bị lệch không gian tối ưu về các lớp đa số (boat, car) và bỏ qua lớp thiểu số (helicopter, taxi). | **Class-Balanced Focal Loss**.<br><br>**Ưu điểm**: Tích hợp trọng số dựa trên số lượng mẫu thực tế độc bản gốc (\(W_i\)) kết hợp Focal Loss để giảm thiểu nhiễu từ các mẫu dễ, tập trung học các mẫu khó thuộc lớp thiểu số. |
| **Tập dữ liệu huấn luyện** *(Training Dataset)* | Tăng cường dữ liệu (Augmentation) offline cơ bản tập trung vào điều kiện thời tiết (Rain, Sun, Night, Normal).<br><br>**Hạn chế**: Chỉ giải quyết nhiễu ánh sáng, chưa bền bỉ với camera bị mờ, nhòe hoặc rung lắc. | Tăng cường dữ liệu offline nâng cao tập trung vào độ nét và mờ (Gaussian, Motion, Unsharp) trên tập **Clean V2**.<br><br>**Ưu điểm**: Tăng khả năng chống nhiễu mờ/nhòe thực tế của camera giám sát. |

### 2. Bảng đối chiếu hiệu năng thực nghiệm định lượng (Chờ đối chiếu kết quả mới)

> [!NOTE]
> Các cột dữ liệu của phiên bản **Cải tiến (Mới)** dưới đây được để ở trạng thái **`[Chờ chạy train]`**. Sau khi tiến hành chạy huấn luyện các mô hình cải tiến trên tập dữ liệu `Clean V2`, bạn chỉ cần điền kết quả đo lường thực tế vào đây để thực hiện so sánh delta.

| Mô hình | Phiên bản | Accuracy (Test) | Macro F1 (Test) | F1-Score Taxi | F1-Score Minibus | F1-Score Helicopter | Độ trễ (Latency) | Trạng thái kiểm chứng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | Baseline (Cũ) | 93.27% | 91.38% | 79.04% | 81.36% | 95.31% | ~12.5 ms | Đã xác thực |
| **ResNet-50** | Cải tiến (Mới) | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ đo]` | Sắp triển khai |
| **ViT-B/16** | Baseline (Cũ) | 94.79% | 97.33% | 97.73% | 97.22% | 96.97% | ~28.4 ms | Đã xác thực |
| **ViT-B/16** | Cải tiến (Mới) | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ đo]` | Sắp triển khai |
| **YOLOv8n-cls** | Baseline (Cũ) | 91.71% | 89.49% | 75.74% | 79.87% | 93.02% | ~3.8 ms | Đã xác thực |
| **YOLOv8n-cls** | Cải tiến (Mới) | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ chạy train]` | `[Chờ đo]` | Sắp triển khai |

---

## IV. LỘ TRÌNH VÀ KẾ HOẠCH TRIỂN KHAI CHI TIẾT

* **Bước 1: Cải tiến Tiền xử lý (Reflective Padding)**: Chỉnh sửa tệp `src/image_pipelines/base.py` để thay thế `zero_pad_to_square` bằng `reflective_pad_to_square` sử dụng OpenCV. Cập nhật `app.py` để đồng bộ hóa tuyệt đối luồng suy luận trực tuyến.
* **Bước 2: Xây dựng Wrapper Trích xuất đặc trưng**: Chỉnh sửa `src/model.py` để triển khai `MultiScaleResNet50` và `MultiLayerCLSViT`. Đảm bảo cấu trúc wrapper có các thuộc tính tương thích để chạy unfreeze đa giai đoạn bằng `switch_strategy` mà không gây lỗi.
* **Bước 3: Tích hợp Class-Balanced Focal Loss**: Chỉnh sửa `src/train.py` để cài đặt hàm loss tùy chỉnh. Khai báo mảng số lượng mẫu gốc \(n_i\) và triển khai giải thuật Focal Loss xử lý được cả nhãn mềm MixUp/CutMix.
* **Bước 4: Chạy Thử nghiệm xác thực**: Thực hiện smoke test mô hình bằng `python src/model.py`. Chạy train 2 epochs trên tập dữ liệu Clean V2 để đảm bảo loss hội tụ ổn định và không gặp lỗi CUDA out of memory.

---
*Dự án được phê duyệt bởi Project Leader*
*Ngày cập nhật bảng so sánh: 18 tháng 06, 2026*
