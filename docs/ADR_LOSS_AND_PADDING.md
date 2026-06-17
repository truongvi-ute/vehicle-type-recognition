# Nhật Ký Quyết Định Kiến Trúc (ADR): Tiền Xử Lý & Tối Ưu Hóa Loss

## Trạng thái
Đang đề xuất (Chờ phê duyệt)

## Bối cảnh
Các mô hình học sâu của chúng ta (ResNet-50 và Vision Transformer) khi huấn luyện trên tập dữ liệu Vehicle-10 gặp phải các điểm nghẽn nghiêm trọng về hiệu năng phân loại:
1. **Biến dạng hình học do kích thước:** Phép thay đổi kích thước trực tiếp (direct resize) làm nén dẹt tỷ lệ thực tế của xe. Mặc dù zero-padding giúp bảo toàn tỷ lệ, nó lại đưa vào các đường biên đen đột ngột, tạo ra các cạnh sắc nhọn nhân tạo (xung nhiễu tần số cao) làm rối loạn bộ lọc tích chập và các cơ chế self-attention.
2. **Sự loãng thông tin lớp thiểu số:** Tập dữ liệu gốc bị mất cân bằng nghiêm trọng (ví dụ: Taxi chỉ có 765 ảnh train, trong khi Boat có tới 7,390 ảnh). Việc tăng cường ngoại tuyến (offline augmentation) nhân bản ảnh thiểu số tạo ra tính lặp lại cao. Hàm lỗi Cross Entropy tiêu chuẩn đối xử đồng đều với mọi mẫu, làm mô hình bị thiên lệch (bias) nặng nề về phía lớp đa số.
3. **Tiêu biến đặc trưng tần số cao:** Phép downsampling tích lũy của ResNet-50 (stride 32) làm tiêu biến các đặc trưng nhỏ (như mào xe taxi chiếm diện tích < 2% ảnh) ở feature map cuối cùng ($7 \times 7$). Trong khi đó, các lớp cuối của ViT gặp hiện tượng mịn hóa quá mức (over-smoothing) token CLS, làm mất đặc trưng hình thái chi tiết của xe (như minibus).

## Quyết định
1. **Đệm gương đối xứng (Reflective Padding):** Thay thế zero-padding trong hàm `apply_base_pipeline` bằng **Reflective Padding** (`cv2.BORDER_REFLECT_101`). Thuật toán sẽ lấy đối xứng biên ảnh qua gương để lấp đầy các khoảng trống thừa sau khi resize giữ nguyên tỷ lệ, triệt tiêu hoàn toàn nhiễu biên đen.
2. **Hàm lỗi Class-Balanced Focal Loss:** Thay thế hoàn toàn `nn.CrossEntropyLoss` bằng hàm lỗi Class-Balanced Focal Loss tự định nghĩa:
   - Tính toán trọng số dựa trên số lượng mẫu huấn luyện thực tế gốc ($W_i = \frac{1 - \beta}{1 - \beta^{n_i}}$ với hệ số mẫu hiệu dụng $\beta = 0.999$).
   - Kết hợp Focal Loss với số mũ $\gamma = 2.0$ để hạ trọng số của các mẫu dễ học (Boat, Car) và tập trung tài nguyên gradient vào các mẫu khó bị phân loại sai.
3. **Ghép nối đặc trưng đa quy mô và đa lớp:**
   - Đối với **ResNet-50**, trích xuất song song và ghép nối (concatenate) vector đặc trưng từ Layer 3 (kích thước $14 \times 14$, stride 16) và Layer 4 (kích thước $7 \times 7$, stride 32) trước khi đưa vào bộ phân loại.
   - Đối với **ViT**, sử dụng cơ chế Forward Hook để trích xuất và ghép nối CLS Tokens từ 3 block encoder cuối cùng (Layer 10, Layer 11, Layer 12).

## Hệ quả

### Tích cực
*   **Bảo toàn hình học & Mịn biên:** Reflective Padding vừa giữ nguyên tỷ lệ xe thực tế vừa loại bỏ các cạnh viền đen giả tạo, giúp activation map của mô hình hội tụ tốt hơn vào xe.
*   **Trọng số Gradient tối ưu:** Class-Balanced Focal Loss định hình lại không gian tối ưu hóa, buộc mô hình phải học sâu hơn các lớp khó mà không bị áp đảo bởi các mẫu trùng lặp của lớp đa số, trực tiếp tăng Recall của lớp Taxi và Helicopter.
*   **Giữ đặc trưng cục bộ:** Giữ lại các đặc trưng cục bộ tần số cao (như mào taxi) nhờ Layer 3 và các đặc trưng hình thái trung cấp (như hông xe minibus) nhờ các lớp trung gian của ViT.

### Tiêu cực
*   **Tăng kích thước Vector phân loại:** Số chiều đầu vào classifier tăng lên (ResNet: từ $2048 \rightarrow 3072$ chiều; ViT: từ $768 \rightarrow 2304$ chiều), làm tăng nhẹ số lượng tham số trong Classification Head.
*   **Độ trễ huấn luyện tăng nhẹ:** Việc tính toán hàm lỗi tự định nghĩa và kiểm tra điều kiện đầu vào (MixUp soft labels) làm tăng nhẹ khối lượng tính toán trên mỗi batch huấn luyện, tuy nhiên không đáng kể khi chạy trên GPU T4.
