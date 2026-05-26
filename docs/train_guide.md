# `src/train.py` — Hướng dẫn Huấn luyện Mô hình (Training Loop)

Module thực thi vòng lặp huấn luyện chính cho đồ án **Nhận dạng Phương tiện Giao thông**. Nó kết hợp `dataset.py` và `model.py` để tạo ra quy trình Fine-tuning hoàn chỉnh.

> **Yêu cầu tiên quyết:** Đã chạy `data_prep.py` để chuẩn bị dữ liệu trong `data/processed/`.

---

## 1. Tính năng nổi bật

1.  **Huấn luyện Đa giai đoạn (Multi-phase Training):**
    *   **Phase 1 (Head Only):** Mặc định 5 epochs đầu tiên chỉ huấn luyện Classification Head với Learning Rate lớn (`1e-3`), đóng băng hoàn toàn phần Backbone để bảo vệ các đặc trưng ImageNet.
    *   **Phase 2 (Partial Fine-tuning):** Mở khoá 1/3 các tầng cuối cùng của Backbone, tự động giảm Learning Rate đi 10 lần (còn `1e-4`) để học tinh chỉnh các đặc trưng hình học đặc thù của phương tiện.
2.  **Đánh giá chi tiết (Detailed Validation):** Tính toán độ chính xác chung (`Val Acc`), đồng thời phân tách rành mạch:
    *   `Val(Unseen)`: Độ chính xác trên 5% ảnh mới hoàn toàn.
    *   `Val(Copy)`: Độ chính xác trên 5% ảnh rò rỉ từ Train (theo yêu cầu Giảng viên).
3.  **Tự động Tối ưu:**
    *   **Early Stopping:** Dừng sớm nếu mô hình không cải thiện sau $N$ epochs (`--patience`).
    *   **LR Scheduler:** Giảm Learning Rate nếu Loss đi ngang (Plateau).
4.  **Lưu trữ tự động (Logging & Checkpointing):**
    *   Lưu lịch sử Loss/Acc dưới định dạng `JSON`.
    *   Tự động ghi đè mô hình tốt nhất vào `models/<model_name>_best.pth` và kèm cấu hình `config.json`.
5.  **Mixup Augmentation:** Áp dụng kỹ thuật nội suy ảnh và nhãn (Mixup) trực tiếp trên GPU, trộn 2 mẫu ngẫu nhiên trong batch để giảm thiểu tối đa hiện tượng Overfitting.

---

## 2. Cách thức chạy huấn luyện

### Huấn luyện Cơ bản

Chạy nhanh với ResNet-50 (Mặc định: Batch 32, Phase1 = 5 epochs, Phase2 = 10 epochs):

```powershell
python src/train.py
```

### Huấn luyện Tuỳ chỉnh

Chạy MobileNet-V3, tăng số batch size, giảm số epoch phase 2:

```powershell
python src/train.py --model mobilenet_v3 --batch_size 64 --phase1_epochs 3 --phase2_epochs 7
```

---

## 3. Các tham số cấu hình (CLI Arguments)

| Tham số | Giá trị mặc định | Giải thích |
| :--- | :--- | :--- |
| `--processed_dir` | `data/processed` | Thư mục chứa dữ liệu đã tiền xử lý. |
| `--checkpoint_dir` | `models` | Thư mục lưu file `.pth` và file `.json` log. |
| `--model` | `resnet50` | Kiến trúc mạng: `resnet50`, `mobilenet_v3`, `efficientnet_b0`. |
| `--batch_size` | `32` | Số lượng ảnh xử lý trong một lượt. Tăng nếu GPU mạnh. |
| `--lr` | `1e-3` | Learning Rate gốc khởi điểm (Phase 1). |
| `--num_workers` | `0` | Luồng xử lý dữ liệu song song (Để `0` trên Windows). |
| `--phase1_epochs`| `5` | Số epoch chỉ huấn luyện head. |
| `--phase2_epochs`| `10` | Số epoch fine-tune 1/3 backbone cuối. |
| `--patience` | `5` | Số epoch liên tiếp không cải thiện thì kích hoạt Early Stopping. |

---

## 4. Diễn giải Kết quả (Console Output)

Trong quá trình huấn luyện, màn hình sẽ hiển thị cấu trúc sau:

```text
======================================================================
BẮT ĐẦU HUẤN LUYỆN: RESNET50 | Tổng Epoch: 15
======================================================================

Epoch 1/15 [Phase 1 (Head Only)] | LR: 1.00e-03
--------------------------------------------------
  Train : Loss 0.8521 | Acc 0.7250
  Valid : Loss 0.6510 | Acc 0.8010  (Unseen: 0.7900, Copy: 0.8120)
  Time  : 00:03:15
  [SAVE] Checkpoint: models\resnet50_epoch001.pth  (epoch=1, val_acc=0.8010, val_loss=0.6510)
  [BEST] Lưu best model → models\resnet50_best.pth

... (Sau 5 Epoch) ...

[CHUYỂN GIAO] Bắt đầu Phase 2: Fine-tune (Partial Freeze)
  [switch_strategy] → 'partial' | Trainable: 24,335,365/24,560,709 (99.1%)

Epoch 6/15 [Phase 2 (Partial)] | LR: 1.00e-04
--------------------------------------------------
```

**Quan sát kỹ sự chênh lệch (Unseen) và (Copy):**
Do đặc thù của đồ án này, tập `Copy` chắc chắn sẽ có Acc rất cao (gần như 100% nếu mô hình Overfit), trong khi tập `Unseen` phản ánh chính xác năng lực thực tế. 

## 5. Đánh giá tự động khi kết thúc

Khi kết thúc toàn bộ số Epoch (hoặc bị ngắt bởi Early Stopping), hệ thống sẽ tự động Load lại file `_best.pth` tốt nhất và chạy thử nghiệm một lần duy nhất trên tập `Test` (10% bộ dữ liệu không bao giờ được học). 

Kết quả Test này là con số chính xác và khách quan nhất báo cáo cho Giảng viên.

```text
Đánh giá trên tập TEST độc lập với Best Model...
  [LOAD] models\resnet50_best.pth
         epoch=14 | val_loss=0.2104, val_acc=0.9102, val_unseen_acc=0.9001, val_copy_acc=0.9203
  Test Loss: 0.2210 | Test Acc: 0.8950
======================================================================
```
