# 🏋️ Huấn Luyện Mô Hỏi & Chiến Lược Đánh Giá (Training & Evaluation Guide)

> [!NOTE]  
> Tài liệu này mô tả chi tiết chiến lược huấn luyện đa giai đoạn, cú pháp lệnh thực thi và cách đánh giá, xử lý sự cố trong quá trình huấn luyện các mô hình.

---

## 1. Phương Pháp Tiếp Cận Đa Giai Đoạn (Multi-Phase Fine-Tuning)

Để tận dụng tối đa sức mạnh của Transfer Learning từ các bộ trọng số pre-trained lớn mà không phá vỡ cấu trúc đặc trưng cơ bản đã học được, dự án áp dụng chiến lược huấn luyện hai giai đoạn (**Multi-Phase Freeze/Unfreeze Strategy**):

* **Giai đoạn 1 (Phase 1 - Epoch 1 đến 5)**: 
  * **Đóng băng Backbone**: Khóa toàn bộ các trọng số của mạng tích chập/Transformer nền tảng.
  * **Chỉ huấn luyện Classification Head**: Tập trung cập nhật các trọng số ở tầng phân loại cuối cùng (FC layer) để khớp với 10 lớp của Vehicle-10.
  * **Learning Rate (LR)**: Đặt ở mức lớn (`1e-3`) để đầu ra nhanh chóng hội tụ sơ bộ.
* **Giai đoạn 2 (Phase 2 - Epoch 6 đến 100)**:
  * **Mở khóa một phần Backbone**: Giải phóng 1/3 số tầng sâu của mạng để tinh chỉnh sâu hơn (Fine-tuning) các đường nét đặc thù của xe cộ (như bánh xe, cabin, cánh quạt trực thăng).
  * **Hạ Learning Rate (LR)**: Giảm đi 10 lần (`1e-4` hoặc thấp hơn) nhằm tránh phá hỏng các đặc trưng tổng quát đã học tốt ở Phase 1.
  * **Tích hợp Early Stopping**: Áp dụng cơ chế dừng sớm với số epoch kiên nhẫn (`patience = 10` đến `15`) dựa trên độ lỗi của tập Validation nhằm chống Overfitting.

---

## 2. Cấu Hình Lệnh Huấn Luyện

Dưới đây là các câu lệnh chính để bắt đầu quá trình huấn luyện cho từng mô hình:

### A. Huấn luyện ResNet-50
```bash
python src/train.py --model resnet50 --data_dir data/augmented --batch_size 64 --max_epochs 100 --patience 10 --lr 1e-3
```

### B. Huấn luyện Vision Transformer (ViT)
> [!IMPORTANT]  
> Đối với ViT, bắt buộc sử dụng thuật toán tối ưu hóa **AdamW** (để phạt trọng số đúng cách) kết hợp kỹ thuật **Warmup Learning Rate** trong những epoch đầu để tránh sụp đổ gradient.

```bash
python src/train_vit.py --model vit_base_patch16_224 --data_dir data/augmented --batch_size 32 --max_epochs 100 --patience 15 --lr 1e-4
```

### C. Huấn luyện YOLO-cls
```bash
yolo classify train model=yolov8n-cls.pt data=data/augmented epochs=100 patience=10 batch=128 imgsz=224
```

---

## 3. Tiêu Chí Đánh Giá Xuất Ra (Evaluation Metrics)

Sau khi quá trình huấn luyện hoàn tất, file mô hình tốt nhất (lưu dưới dạng `*_best.pth` hoặc `*_best.pt`) sẽ tự động được kiểm thử độc lập trên tập **Test** để đánh giá chéo hiệu năng thực tế. Các kết quả xuất ra thư mục `outputs/` bao gồm:

1. **Ma trận nhầm lẫn (Confusion Matrix)**: 
   * Được kết xuất dưới dạng file ảnh heatmap `.png` (kích thước 10x10).
   * Giúp trực quan hóa và phân tích cụ thể mô hình thường xuyên nhầm lẫn giữa các lớp nào nhất (Ví dụ: sự tương đồng hình học cao giữa `Minibus` và `Bus`, hoặc `Car` và `Taxi`).
2. **Báo cáo phân loại (Classification Report)**:
   * File `.json` lưu chi tiết các chỉ số chất lượng cho từng lớp và trung bình toàn bộ hệ thống:
     * **Precision** (Độ chính xác dự đoán): Tỷ lệ các mẫu dự đoán đúng trên tổng số mẫu được dự báo thuộc lớp đó.
     * **Recall** (Độ phủ thực tế): Tỷ lệ các mẫu dự đoán đúng trên tổng số mẫu thực tế của lớp đó.
     * **F1-Score**: Trung bình điều hòa giữa Precision và Recall.

---

## 4. Xử Lý Sự Cố Kỹ Thuật (Troubleshooting)

Trong quá trình huấn luyện mô hình sâu, có thể xảy ra một số lỗi hệ thống phổ biến:

* **Lỗi Out of Memory (OOM) - Tràn bộ nhớ VRAM GPU**:
  * *Hiện tượng*: PyTorch báo lỗi `CUDA out of memory`.
  * *Giải pháp*:
    1. Hạ tham số kích thước batch (`--batch_size`) từ 64 xuống 32, 16 hoặc 8.
    2. Áp dụng kỹ thuật tích lũy gradient (**Gradient Accumulation**) để duy trì kích thước batch ảo lớn mà không gây quá tải bộ nhớ vật lý.
* **Lỗi NaN Loss (Nổ Gradient - Gradient Explosion)**:
  * *Hiện tượng*: Hàm Loss trả về giá trị `NaN` (Not a Number) sau một vài lượt lặp.
  * *Giải pháp*:
    1. Giảm tốc độ học ban đầu (`--lr`) xuống 5 đến 10 lần.
    2. Kích hoạt tính năng kẹp biên độ gradient (**Gradient Clipping**) bằng hàm `torch.nn.utils.clip_grad_norm_` trước bước cập nhật optimizer.
