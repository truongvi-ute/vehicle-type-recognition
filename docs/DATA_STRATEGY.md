# 📉 Chiến Lược Xử Lý & Cân Bằng Dữ Liệu (Data Strategy)

> [!NOTE]  
> Tài liệu này chi tiết hóa giải pháp giải quyết hai bài toán cốt lõi: ngăn chặn lỗi rò rỉ dữ liệu (**Data Leakage**) và khắc phục hiện tượng mất cân bằng lớp dữ liệu (**Class Imbalance**) nghiêm trọng của bộ dữ liệu Vehicle-10.

---

## 1. Giao Thức Chia Tập "Split First Protocol"

Để chặn đứng lỗi **Data Leakage (rò rỉ dữ liệu)** — hiện tượng dữ liệu kiểm thử hoặc thông tin từ tập xác thực xuất hiện gián tiếp trong quá trình huấn luyện do các phép biến đổi hình học hoặc tăng cường mẫu chung — toàn bộ dữ liệu gốc buộc phải phân chia tập trước khi xử lý:

* **Tập Train (85% - 30,605 ảnh)**: Duy nhất tập này được đi qua luồng cân bằng (balancing) và tăng cường dữ liệu vật lý (augmentation).
* **Tập Valid (5% - 1,800 ảnh) & Tập Test (10% - 3,601 ảnh)**: Giữ nguyên bản gốc. Chỉ chạy duy nhất qua luồng tiền xử lý cơ bản (Base normalization - chuẩn hóa kích thước 224x224 và bù padding), tuyệt đối không áp dụng các bộ lọc tăng cường, xoay, lật hay nhiễu để đảm bảo tính khách quan khi đánh giá.

---

## 2. Chiến Lược Cân Bằng Dữ Liệu (Class Balancing)

Vì bộ dữ liệu gốc mất cân bằng nặng (tỷ lệ 13.3:1), chúng ta áp dụng phương pháp cân bằng lại số lượng mẫu giữa các lớp phương tiện trên tập **Train**.

Mọi lớp phương tiện được nâng quy mô mẫu (Oversampling) đồng đều về con số của lớp lớn nhất ($N_{target} = 7,117$ ảnh của lớp `boat`) thông qua hệ số nhân bản $K$:

$$K = \left\lceil \frac{N_{target}}{N_{current}} \right\rceil$$

### Quy trình chi tiết:
1. Xác định số lượng mẫu mục tiêu $N_{target}$ (trong trường hợp này là lớp `boat` với 7,117 mẫu).
2. Với mỗi lớp $C$, tính hệ số nhân bản $K$ bằng cách lấy trần phép chia mẫu mục tiêu cho số lượng mẫu hiện tại.
3. Nhân bản dữ liệu của lớp đó lên $K$ lần.
4. Lấy mẫu ngẫu nhiên (down-sample) không hoàn lại đúng $N_{target}$ ảnh từ tập dữ liệu vừa được nhân bản.

**Kết quả:** Sau khi cân bằng, tổng số lượng mẫu tập Train đạt **71,170 ảnh** chia đều tuyệt đối cho 10 lớp (mỗi lớp 7,117 ảnh).

---

## 3. Tăng Cường Dữ Liệu Trực Tuyến (Online Augmentation)

Bên cạnh việc tăng cường offline trên đĩa cứng, hệ thống tích hợp các kỹ thuật tăng cường trực tuyến (Online Augmentation) trong lúc huấn luyện nhằm tăng độ tổng quát hóa của mô hình:

### A. Đối với ResNet-50 & Vision Transformer (ViT)
Sử dụng hai kỹ thuật trộn ảnh hiện đại cấu hình qua thư viện `torchvision.transforms.v2`:
* **MixUp**: Trộn hai ảnh ngẫu nhiên trong batch đè lên nhau theo một tỷ lệ $\lambda$ ngẫu nhiên (lấy từ phân phối Beta), nhãn của ảnh mới cũng được làm mịn theo tỷ lệ tương ứng. Giúp mô hình học được ranh giới quyết định liên tục.
* **CutMix**: Cắt một phần vùng chữ nhật từ bức ảnh này dán chéo sang bức ảnh kia. Giúp mô hình không chỉ tập trung vào một đặc trưng cục bộ dễ nhận thấy mà học cách quét toàn bộ cấu trúc vật thể.

### B. Đối với YOLO-cls
* Kích hoạt cơ chế **Mosaic** (`mosaic=1.0`). Cơ chế này ghép 4 bức ảnh ngẫu nhiên thành 1 khung hình với tỉ lệ thu phóng khác nhau, tạo ra bối cảnh nền phong phú và đa dạng kích thước vật thể, tối ưu hóa mạnh mẽ khả năng bao quát không gian của mạng YOLO.
