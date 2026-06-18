# Đầu Vào, Đầu Ra & Cơ Chế Suy Luận (Input, Output & Inference Mechanics)

Tài liệu này giải thích chi tiết về định dạng đầu vào (Input), đầu ra (Output), loại hình dự đoán và cơ chế suy luận của dự án **VehicleTypeRecognition** để bạn nắm vững bản chất học sâu của đề tài.

---

## 1. Đầu Vào và Đầu Ra của Hệ Thống (Input & Output)

Dự án xử lý dữ liệu qua hai giai đoạn: **Huấn luyện (Training)** và **Suy luận trực tuyến (Inference/Demo)**.

### 1.1. Giai đoạn Huấn luyện (Training Phase)
* **Đầu vào (Input):**
  * Tập dữ liệu ảnh phương tiện giao thông đã được chia và tiền xử lý offline (V1 Weather hoặc V2 Blur/Sharpen).
  * Đối với các mô hình PyTorch tiêu chuẩn (**ResNet-50, ViT**):
    * Một Batch ảnh biểu diễn dưới dạng Tensor kích thước `(B, 3, 224, 224)` với `B` là batch size.
    * Các giá trị pixel được chuẩn hóa: Chia cho 255.0 đưa về miền `[0.0, 1.0]` và áp dụng công thức chuẩn hóa ImageNet:
      $$\text{Normalized} = \frac{\text{Pixel} - \text{Mean}}{\text{Std}}$$
      (Trong đó: \(\text{Mean} = [0.485, 0.456, 0.406]\), \(\text{Std} = [0.229, 0.224, 0.225]\)).
  * Đối với mô hình **YOLOv8n-cls**:
    * Đầu vào là tệp tin ảnh hoặc đối tượng ảnh PIL. Trình tải dữ liệu của Ultralytics tự động chuyển đổi thành Tensor kích thước `(B, 3, 224, 224)` và chuẩn hóa `[0, 1]` ở bên trong card đồ họa.
* **Đầu ra (Output):**
  * Một vector dự đoán (Logits) kích thước `(B, 10)` chứa điểm số phân loại chưa chuẩn hóa của 10 lớp phương tiện giao thông.
  * Vector này đi qua hàm Loss **CrossEntropyLoss** để tính toán sai lệch so với nhãn đúng (Ground Truth index) nhằm thực hiện cập nhật trọng số qua lan truyền ngược (Backpropagation).

### 1.2. Giai đoạn Suy luận trực tuyến (Inference/Demo Phase)
* **Đầu vào (Input):**
  * 1 tệp ảnh phương tiện do người dùng tải lên giao diện React UI (định dạng `.jpg`, `.jpeg`, `.png`, dung lượng `<10MB`).
  * Tham số cấu hình gửi kèm: Tên mô hình (1 trong 12 phiên bản) và loại bộ lọc tiền xử lý (Base / Rain / Sun / Night / Gaussian / Motion / Unsharp).
* **Đầu ra (Output) - Phản hồi JSON từ Flask:**
  * `model_name`: Tên của mô hình đang thực hiện suy luận.
  * `pipeline`: Tên thuật toán tiền xử lý đã áp dụng.
  * `processed_image`: Ảnh sau khi đi qua pipeline tiền xử lý, được mã hóa dưới dạng chuỗi **Base64** (`data:image/png;base64,...`) để React UI hiển thị trực tiếp lên màn hình.
  * `predictions`: Mảng chứa thông tin phân loại của **tất cả 10 lớp phương tiện giao thông**, mỗi phần tử gồm `class_name` (tên lớp) và `confidence` (độ tự tin/xác suất - số thực từ `0.0` đến `1.0` làm tròn đến 6 chữ số thập phân), được sắp xếp theo thứ tự giảm dần.
  * `processing_time_ms`: Thời gian trễ suy luận thực tế (ms) đo bằng CPU/GPU.

---

## 2. Kiểu Dự Đoán & Cơ Chế Hoạt Động (Prediction Type & Mechanics)

### 2.1. Dự đoán theo kiểu gì? (Prediction Type)
Đề tài của chúng ta thuộc kiểu bài toán **Phân loại ảnh đa lớp (Multi-class Image Classification)**:
* **Mục tiêu:** Nhận diện và gán duy nhất **1 nhãn lớp phân loại chính xác nhất** trong số 10 lớp phương tiện giao thông đã định nghĩa cho toàn bộ bức ảnh đầu vào.
* **Phân biệt:** 
  * Đây *không phải* là bài toán Phát hiện đối tượng (Object Detection - không sinh tọa độ khung bao Bounding Box).
  * Đây *không phải* là bài toán Phân loại đa nhãn (Multi-label - một ảnh chỉ thuộc về duy nhất 1 phương tiện chủ thể chính, không gán nhiều nhãn cùng lúc).

### 2.2. Dự đoán như thế nào? (How the Prediction is Done)
Khi ảnh đi qua mô hình AI đã nạp, luồng tính toán diễn ra như sau:

#### A. Kiến trúc ResNet-50 (Mô hình CNN)
1. **Trích xuất đặc trưng cục bộ:** Ảnh Tensor `(1, 3, 224, 224)` đi qua 4 nhóm khối Residual (Residual Blocks) chứa các tầng tích chập (Convolutional Layers). Phép toán tích chập quét qua các vùng ảnh để học các đặc trưng hình học cục bộ từ đơn giản (cạnh, góc, màu sắc) đến phức tạp (bánh xe, mào taxi, cabin).
2. **Kênh kết nối FC:** Đi qua tầng Average Pooling đưa về kích thước phẳng và tầng Fully Connected (FC) cuối cùng để ánh xạ các đặc trưng thành vector điểm số Logits thô gồm 10 phần tử.
3. **Tính xác suất Softmax:** Flask Backend nhận Logits và chạy hàm **Softmax** để chuyển đổi điểm số thô thành phân phối xác suất thực tế:
   $$P(y = i | x) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$
   (Trong đó \(z_i\) là điểm logit thứ \(i\), tổng xác suất của cả 10 lớp luôn bằng 1.0).

#### B. Kiến trúc Vision Transformer (ViT-B/16 - Mô hình Attention)
1. **Phân mảnh ảnh (Patching):** Ảnh Tensor `224x224` được cắt nhỏ thành **196 mảnh ảnh (Patches)** kích thước `16x16` pixel.
2. **Chiếu tuyến tính & Nhúng vị trí (Projection & Positional Embedding):** Mỗi mảnh được duỗi phẳng và chiếu tuyến tính thành một vector 768 chiều. Nhóm cộng thêm vector nhúng vị trí (Positional Embedding) để mô hình biết được vị trí của mảnh đó trong bức ảnh tổng thể.
3. **Token phân loại `[CLS]`:** Thêm một token phân loại đặc biệt `[CLS]` vào đầu chuỗi mảnh nhúng.
4. **Cơ chế Tự chú ý (Self-Attention):** Chuỗi token đi qua 12 tầng Transformer Encoder. Cơ chế Multi-Head Self-Attention cho phép các mảnh ảnh "nhìn thấy" và tính toán mối liên quan ngữ cảnh với toàn bộ các mảnh khác (Contextual Modeling). Điều này giúp ViT nhận diện được các đặc trưng dài hạn liên quan đến tỷ lệ hình học toàn cục của xe.
5. **Trích xuất phân loại:** Tầng MLP Head cuối cùng chỉ trích xuất thông tin đầu ra tại token `[CLS]` để tạo ra vector Logits 10 lớp và áp dụng hàm Softmax tương tự ResNet.

#### C. Kiến trúc YOLOv8n-cls (Mô hình CNN tối ưu)
* Hoạt động tương tự ResNet nhưng sử dụng xương sống **DarkNet** cải tiến siêu nhẹ.
* Khác biệt duy nhất ở suy luận: Thư viện `ultralytics` chạy Softmax nội bộ và trả về kết quả xác suất trực tiếp qua đối tượng `.probs` mà không cần Flask chạy Softmax thủ công.
