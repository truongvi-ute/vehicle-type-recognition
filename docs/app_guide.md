# `app.py` — Ứng dụng Giao diện Web (Streamlit)

Giao diện đồ họa người dùng (GUI) cho đồ án **Nhận dạng Phương tiện Giao thông**. Ứng dụng được xây dựng bằng **Streamlit**, cho phép kiểm thử trực tiếp mô hình AI trên nền web một cách trực quan.

> **Yêu cầu tiên quyết:** Đã cài đặt thư viện Streamlit (`pip install streamlit pandas`) và đã chạy file `train.py` để có ít nhất 1 file trọng số mô hình `.pth`.

---

## 1. Khởi động Ứng dụng

Mở Terminal / Command Prompt tại thư mục gốc của dự án và chạy:

```powershell
streamlit run app.py
```

Một tab mới trên trình duyệt (thường là `http://localhost:8501`) sẽ tự động mở ra.

---

## 2. Các Tính năng Chính

### 2.1 Cấu hình AI ở Menu Trái (Sidebar)

*   **Chọn Mô Hình:** Lựa chọn 1 trong 3 kiến trúc đã thiết kế (`resnet50`, `mobilenet_v3`, `efficientnet_b0`).
*   **Đường dẫn Checkpoint:** Ứng dụng sẽ tự động nội suy file tốt nhất dựa trên kiến trúc (ví dụ: `models/resnet50_best.pth`). Bạn có thể tự gõ đường dẫn khác nếu muốn.
*   **Pipeline Tự động khóa:** Hệ thống sẽ tự động quét file `_config.json` đi kèm checkpoint để tự động thiết lập chuỗi tiền xử lý (Pipeline) chuẩn. Cơ chế này loại bỏ hoàn toàn rủi ro Training-Serving Skew (lệch pha huấn luyện - phục vụ).

### 2.2 Tải Ảnh Lên (Upload)

Kéo thả hoặc click để chọn 1 tấm ảnh phương tiện (định dạng JPG, PNG, WEBP).
> *Mẹo: Bạn có thể lấy trực tiếp một ảnh bất kỳ trong thư mục `data/raw/` hoặc tải ảnh xe cộ từ Internet về để kiểm tra mô hình.*

### 2.3 Xem Trực Quan (Visual Comparison)

Ngay khi tải lên, bức ảnh sẽ đi qua quy trình tiền xử lý tương tự như lúc tạo tập Dataset:
1.  Resize giữ nguyên tỷ lệ + đệm viền đen để đạt kích thước chuẩn (224x224).
2.  Đi qua chuỗi bộ lọc (Pipeline) đã được tự động khoá từ thông tin của Checkpoint.

Màn hình sẽ hiển thị song song 2 ảnh: **Ảnh Gốc** bên trái, **Ảnh Đã Tiền Xử Lý** bên phải.

### 2.4 Kết Quả Nhận Diện (AI Inference)

Ảnh sau khi lọc sẽ được chuẩn hoá (`ToTensor`, `Normalize(ImageNet)`) và đưa vào mô hình để suy luận (Inference).
*   In ra lớp phương tiện có xác suất cao nhất.
*   Hiển thị biểu đồ cột mô tả độ tin cậy (Confidence Score %) của tất cả các lớp.

### 2.5 Giải Thích Thuật Toán (Education)

Mục **"Xem cấu hình Pipeline đang áp dụng"** ở dưới cùng sẽ trích xuất tự động JSON của chuỗi các bước biến đổi được mô hình sử dụng, giúp bất kỳ ai sử dụng app hiểu rõ dữ liệu đã đi qua những biến đổi không gian nào.

---

## 3. Khắc phục sự cố

| Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Không tìm thấy file trọng số** | Chưa chạy `train.py` hoặc điền sai đường dẫn | Mở Terminal chạy `python src/train.py` để train một mô hình mẫu, hoặc trỏ đúng đường dẫn trong app. |
| **Lỗi thư viện thiếu** | Chưa cài đặt đủ Dependency | Chạy `pip install streamlit pandas opencv-python pillow torch torchvision` |
| **Lỗi `class_to_idx` sai** | Tên lớp từ thư mục `data/raw` không khớp lúc train | Đảm bảo thư mục `data/raw/` chứa đúng các thư mục (Bike, Bus, Car...), vì app sẽ tự động quét thư mục này để lấy tên lớp. |
