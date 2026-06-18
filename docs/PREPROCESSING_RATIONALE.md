# Lý Do Lựa Chọn Các Phương Pháp Tiền Xử Lý (Preprocessing Rationale)

Tài liệu này tổng hợp lập luận khoa học và thực tiễn để giải thích tại sao nhóm lại lựa chọn các phương pháp tiền xử lý này thay vì các phương pháp khác. Đây là nội dung quan trọng giúp bạn trả lời thuyết phục trước câu hỏi chất vấn: *"Tại sao nhóm lại tiền xử lý theo cách này?"*.

---

## 1. Tại sao lại chọn "Resize giữ tỷ lệ + Zero-padding" thay vì "Resize trực tiếp"?

### 1.1. Tác hại của Resize trực tiếp (Direct Resizing/Stretching)
* Khi nén một bức ảnh hình chữ nhật (ví dụ tỷ lệ 16:9 hoặc 4:3 từ camera giám sát) trực tiếp về khung vuông 224x224 của mạng nơ-ron, hình ảnh sẽ bị **bóp méo hình học (spatial distortion)**. 
* Ví dụ: Một chiếc xe khách (bus) dài sẽ bị co ngắn lại trông giống xe tải (truck) hoặc minibus; một chiếc xe máy (motorcycle) sẽ bị nén dẹt. 
* Sự bóp méo này làm thay đổi các tỷ lệ đặc trưng hình học quan trọng (như tỷ lệ chiều dài/chiều cao, tỷ lệ bánh xe, độ dốc đầu xe) mà mô hình cần học để phân biệt các lớp phương tiện tương đồng cao.

### 1.2. Lợi ích của Resize giữ tỷ lệ kết hợp bù viền đen (Zero-padding)
* **Bảo toàn đặc trưng hình học:** Giúp xe giữ nguyên hình dạng vật lý thực tế trên ảnh vuông 224x224.
* **Đồng nhất vị trí:** Phép paste ảnh vào giữa canvas đen đảm bảo chủ thể xe luôn nằm ở vị trí trung tâm, giúp mạng nơ-ron tích chập (CNN) và Transformer dễ dàng hội tụ và tập trung vùng chú ý (attention) vào chủ thể chính.

### 1.3. Tại sao không chọn bộ cắt ngẫu nhiên (RandomResizedCrop)?
* *RandomResizedCrop* là phép tăng cường trực tuyến phổ biến trong huấn luyện phân loại ảnh ImageNet.
* Tuy nhiên, trong ảnh giao thông thực tế, phương tiện giao thông (chủ thể) có thể chỉ chiếm một phần nhỏ trong khung cảnh rộng. Nếu cắt ngẫu nhiên, mô hình có thể cắt mất xe và chỉ lấy phần background (mặt đường, hàng cây, bầu trời), dẫn đến nhãn bị sai lệch (label noise) trong quá trình huấn luyện.
* Việc bảo toàn toàn bộ ảnh đã zero-pad giúp mô hình luôn học được trọn vẹn hình dáng của xe.

---

## 2. Tại sao chọn Xử lý ảnh số truyền thống để chèn nhiễu ngoại tuyến (Offline Augmentation)?

Thay vì sử dụng các công nghệ tạo ảnh mới như GANs (Generative Adversarial Networks) hay Diffusion Models để sinh ảnh thời tiết/nhiễu, nhóm quyết định chọn các thuật toán xử lý ảnh số truyền thống vì:

### 2.1. Kiểm soát định lượng và tính nhất quán (Controllability & Determinism)
* Các bộ lọc nhiễu truyền thống (vẽ vệt mưa toán học, chèn vầng sáng mặt trời tuyến tính, biến đổi gamma đêm tối) là **deterministic (xác định)** thông qua một Seed ngẫu nhiên.
* Điều này giúp nhóm kiểm soát chính xác 100% tỷ lệ phân phối mẫu (luôn cố định 70% Normal, 10% Rain, 10% Sun, 10% Night hoặc 10% Blur/Sharpen cho từng lớp xe).
* Các mô hình sinh ảnh (Generative AI) thường không kiểm soát được nội dung sinh ra, dễ tạo ra hiện tượng "ảo ảnh" (hallucination) làm biến đổi hoặc méo mó các chi tiết quan trọng của xe (như biển số, mào xe, logo), gây nhiễu cho mô hình học sâu.

### 2.2. Chi phí tính toán cực kỳ thấp (Resource Efficiency)
* Các thuật toán xử lý ảnh số truyền thống chạy bằng CPU chỉ mất vài mili-giây trên mỗi ảnh.
* Ngược lại, việc huấn luyện và suy luận các mô hình sinh ảnh đòi hỏi tài nguyên GPU khổng lồ và thời gian xử lý rất lâu, không khả thi trong môi trường sản xuất thực tế hoặc các hệ thống giám sát biên (Edge Devices).

### 3. Ý nghĩa thực tiễn của các nhóm nhiễu được chọn (V1 & V2)

Nhóm lựa chọn chèn các loại nhiễu này vì chúng mô phỏng chính xác các thách thức vật lý thực tế của camera giám sát giao thông:

* **V1 (Weather - Thời tiết):** Camera giám sát hoạt động ngoài trời 24/7, do đó phải đối mặt trực tiếp với các hiện tượng thiên nhiên làm giảm chất lượng ảnh:
  * **Rain (Mưa):** Làm nhòe và xuất hiện các vệt cản trở tầm nhìn.
  * **Sun (Nắng lóa):** Gây hiện tượng lóa camera làm mất chi tiết vùng sáng.
  * **Night (Đêm tối):** Gây hiện tượng thiếu sáng, giảm độ tương phản và xuất hiện nhiễu hạt do cảm biến ISO tăng cao.
* **V2 (Blur & Sharpen - Mờ/Nét):** Mô phỏng lỗi phần cứng vật lý của ống kính camera và chuyển động:
  * **Motion Blur (Mờ chuyển động):** Xe di chuyển ở tốc độ cao trong khi tốc độ màn trập camera chậm sẽ tạo ra vệt mờ chuyển động tuyến tính.
  * **Gaussian Blur (Mờ tiêu cự):** Do camera bị bám bụi, hơi nước đọng trên kính hoặc lấy nét sai tiêu cự.
  * **Unsharp Mask (Làm nét):** Mô phỏng bước tăng cường cạnh sắc thường được tích hợp sẵn trong chip xử lý DSP của camera trước khi gửi dữ liệu về server.

---

## 4. Sự kết hợp hoàn hảo giữa Xử lý ảnh truyền thống và Học sâu (Deep Learning)

* Lựa chọn phương pháp này giúp đề tài thể hiện một **luồng kỹ thuật lai (Hybrid Pipeline)** hoàn chỉnh. 
* Nhóm không phụ thuộc hoàn toàn vào mạng nơ-ron một cách "hộp đen" (Black-box), mà chủ động sử dụng kiến thức môn học **Xử lý ảnh số** (Lọc tuyến tính, biến đổi phi tuyến, lọc tần số, xử lý hình thái học) để kiểm soát chất lượng dữ liệu đầu vào.
* Việc dữ liệu được chuẩn bị tốt từ bước tiền xử lý đã trực tiếp giúp mô hình học sâu (ResNet, ViT, YOLO) tăng đáng kể độ chính xác (Accuracy tăng tới **1.5% - 2%** trên tập dữ liệu đã lọc sạch nhiễu `Raw Cleaning`).
