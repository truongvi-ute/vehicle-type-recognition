# 🤖 Đối Sánh Lý Thuyết Giữa 3 Kiến Trúc Mô Hình (Model Comparison)

> [!NOTE]  
> Dự án thực hiện đánh giá chéo giữa 3 trường phái kiến trúc mạng nơ-ron tiêu biểu: Mạng tích chập truyền thống (CNN), Cơ chế tự chú ý (Transformer), và Mô hình phát hiện/phân loại thời gian thực tối ưu hóa cực cao (YOLO-cls).

---

## 1. Bảng So Sánh Chi Tiết

| Tiêu chí | ResNet-50 | Vision Transformer (ViT) | YOLO-cls |
| :--- | :--- | :--- | :--- |
| **Bản chất kiến trúc** | CNN truyền thống với mạng dư thừa (Skip Connections) | Cơ chế Self-Attention nắm bắt bối cảnh toàn cục | Mạng CSPDarknet tối ưu hóa tốc độ suy luận |
| **Số lượng tham số** | ~24.5M | ~86M (Bản ViT-Base) | ~3M - 7M (Bản Nano/Small) |
| **Tốc độ suy luận** | ~100 FPS | ~50 FPS | ~200+ FPS |
| **Độ nhạy dữ liệu** | Hoạt động tốt với data quy mô vừa | Data-hungry, dễ overfit nếu data ít | Hội tụ nhanh, tối ưu tốt trên tập dữ liệu |
| **Vai trò dự án** | Làm Baseline đối sánh chuẩn | Kiểm thử giới hạn độ chính xác SOTA | Triển khai thực tế trên Production |

---

## 2. Chi Tiết Kiến Trúc & Đặc Tính

### 1. ResNet-50 (Residual Network)
* **Nguyên lý chính**: Sử dụng các liên kết tắt (**Skip Connections** hay **Residual Connections**) để truyền trực tiếp tín hiệu qua các tầng nơ-ron mà không bị suy giảm. Điều này giải quyết triệt để lỗi tiêu biến gradient (vanishing gradient) khi huấn luyện mạng rất sâu.
* **Ưu điểm**: Ổn định, dễ huấn luyện, độ chính xác cao trên các tập dữ liệu kích thước trung bình và là tiêu chuẩn đối sánh (Baseline) vững chắc cho mọi hệ thống phân loại ảnh.

### 2. Vision Transformer (ViT)
* **Nguyên lý chính**: Kế thừa cơ chế tự chú ý (**Self-Attention**) từ các mô hình ngôn ngữ lớn (Transformer). Ảnh được cắt thành các mảnh nhỏ (Patches), biến đổi thành vector nhúng và xử lý tuần tự để nắm bắt mối tương quan không gian toàn cục trực tiếp ngay từ các lớp đầu tiên.
* **Ưu điểm**: Cho phép mô hình hiểu được bối cảnh rộng lớn của toàn bộ ảnh (không bị giới hạn bởi trường thụ cảm cục bộ như CNN), đạt độ chính xác SOTA (State-of-the-Art) khi có đủ lượng dữ liệu lớn và chiến lược tiền xử lý phù hợp.

### 3. YOLO-cls (YOLO Classification)
* **Nguyên lý chính**: Dựa trên kiến trúc xương sống **CSPDarknet** kết hợp với các kỹ thuật tối ưu phần cứng của Ultralytics. Mô hình tập trung vào việc giảm thiểu khối lượng tính toán mà vẫn duy trì khả năng trích xuất đặc trưng mạnh mẽ.
* **Ưu điểm**: Tốc độ xử lý cực kỳ nhanh (thích hợp cho các luồng video trực tiếp thời gian thực), dung lượng mô hình siêu nhẹ, dễ dàng nhúng vào các thiết bị biên (Edge Devices) có cấu hình thấp.
