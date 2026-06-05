# 🚀 Triển Khai Giao Diện & Ứng Dụng (Deployment)

> [!NOTE]  
> Tài liệu này hướng dẫn cách khởi động giao diện Dashboard kiểm thử thời gian thực bằng Streamlit, các tính năng chính của ứng dụng và cách khắc phục một số lỗi môi trường phổ biến.

---

## 1. Khởi Động Giao Diện

Chúng ta sử dụng thư viện **Streamlit** để tạo một Dashboard kiểm thử trực quan và nhanh chóng. Để khởi chạy ứng dụng, hãy mở cửa sổ terminal tại thư mục gốc của dự án và thực thi lệnh sau:

```powershell
streamlit run app.py
```

Sau khi chạy lệnh, giao diện web sẽ tự động được khởi tạo và mở trên trình duyệt mặc định của bạn tại địa chỉ:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 2. Tính Năng Giao Diện Dashboard

Giao diện ứng dụng được thiết kế tối giản, trực quan và chia làm các khu vực chức năng chính:

* **Cấu hình AI (Thanh Sidebar bên trái)**:
  * Cho phép người dùng linh hoạt chọn một trong ba kiến trúc mạng đã huấn luyện để chạy suy luận (`ResNet-50`, `Vision Transformer`, hoặc `YOLO-cls`).
  * **Cơ chế bảo vệ Training-Serving Skew**: Ứng dụng tự động quét và áp dụng đúng cấu hình tiền xử lý/pipeline lọc mà mô hình đó đã được sử dụng trong giai đoạn huấn luyện (đọc từ file cấu hình best checkpoint tương ứng).
* **Khu vực tải ảnh & Đối sánh trực quan**:
  * Người dùng có thể kéo thả hoặc chọn tải lên một ảnh phương tiện bất kỳ từ máy tính.
  * Màn hình sẽ hiển thị trực tiếp hai khung hình song song:
    * **Ảnh gốc** ở phía bên trái.
    * **Ảnh sau khi đi qua pipeline tiền xử lý OpenCV** ở phía bên phải để người dùng quan sát rõ sự khác biệt của bộ lọc miền không gian.
* **Kết quả Nhận diện**:
  * Hiển thị bảng xếp hạng **Top-3 dự đoán** có độ tin cậy cao nhất từ mô hình AI đã chọn.
  * Trực quan hóa phần trăm điểm số tin cậy (Confidence Score) dưới dạng biểu đồ cột động.

---

## 3. Lỗi Môi Trường Thường Gặp & Cách Khắc Phục

* **Không tìm thấy file trọng số (Model Weight Not Found)**:
  * *Nguyên nhân*: Ứng dụng Streamlit không tìm thấy file trọng số mô hình `.pth` hoặc `.pt` trong thư mục `models/`.
  * *Khắc phục*: Đảm bảo bạn đã hoàn thành việc huấn luyện mô hình (chạy file `src/train.py`, `src/train_vit.py`, hoặc `src/train_yolo.py`) để sinh ra checkpoint tốt nhất trước khi khởi động ứng dụng demo.
* **Lỗi chỉ mục phân lớp (Class Index Mismatch)**:
  * *Nguyên nhân*: Số lượng hoặc tên lớp của ảnh đầu vào suy luận không tương thích với cấu hình đầu ra của mô hình.
  * *Khắc phục*: Đảm bảo rằng cấu trúc thư mục con trong `data/raw/` duy trì chuẩn xác tên viết thường (lowercase) của 10 lớp phương tiện gốc (`bicycle`, `boat`, `bus`, `car`, `helicopter`, `minibus`, `motorcycle`, `taxi`, `train`, `truck`) vì hệ thống sử dụng tên thư mục con này để định danh các nhãn lớp logic.
