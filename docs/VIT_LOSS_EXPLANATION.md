# Hàm Mất Mát của Vision Transformer (ViT Loss Explanation)

Tài liệu này giải thích chi tiết về kiểu hàm mất mát (Loss function), công thức toán học và lý do kỹ thuật tại sao mô hình **Vision Transformer (ViT)** trong dự án lại sử dụng cấu hình Loss này.

---

## 1. Kiểu Loss Sử Dụng trong Mã Nguồn

Trong tệp kịch bản huấn luyện chính `src/train.py` (dòng 290), hàm mất mát được định nghĩa như sau:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
```

* **Tên hàm mất mát:** Cross Entropy Loss with Label Smoothing (Hàm mất mát Entropy chéo kết hợp Làm mịn nhãn).
* **Hệ số làm mịn (Smoothing factor - $\alpha$):** `0.1` (10%).
* **Số lượng lớp ($K$):** `10` lớp phương tiện giao thông.

---

## 2. Công Thức Toán Học Chi Tiết

### 2.1. Hàm Softmax (Tính xác suất dự đoán)
Với đầu ra Logits thô của mô hình là vector $z = [z_1, z_2, ..., z_{10}]$, xác suất dự đoán $\hat{y}_k$ của lớp thứ $k$ được tính bằng hàm Softmax:
$$\hat{y}_k = \text{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}$$

### 2.2. Kỹ thuật làm mịn nhãn (Label Smoothing)
Trong bài toán phân loại thông thường (Standard Cross Entropy), nhãn đúng là một vector One-hot $q$:
* Lớp chính xác $q_{correct} = 1$.
* Các lớp sai $q_{incorrect} = 0$.

Kỹ thuật **Label Smoothing ($\alpha = 0.1$)** thay đổi phân phối nhãn mục tiêu này thành vector phân phối xác suất mịn $y$:
* Nhãn của lớp đúng: 
  $$y_{correct} = (1 - \alpha) + \frac{\alpha}{K} = (1 - 0.1) + \frac{0.1}{10} = 0.9 + 0.01 = 0.91$$
* Nhãn của các lớp sai: 
  $$y_{incorrect} = \frac{\alpha}{K} = \frac{0.1}{10} = 0.01$$

### 2.3. Công thức tính Loss
Hàm mất mát Entropy chéo với nhãn mịn được tính theo công thức:
$$\mathcal{L}_{LS} = -\sum_{k=1}^{10} y_k \log(\hat{y}_k)$$

Công thức này có thể khai triển thành sự kết hợp tuyến tính giữa hàm Cross Entropy truyền thống ($\mathcal{L}_{CE}$) và hàm mất mát phân phối đều ($\mathcal{L}_{uni}$):
$$\mathcal{L}_{LS} = (1 - \alpha) \cdot \mathcal{L}_{CE}(q, \hat{y}) + \alpha \cdot \mathcal{L}_{uni}(u, \hat{y})$$
Trong đó:
* $\mathcal{L}_{CE}(q, \hat{y}) = -\log(\hat{y}_{correct})$: Chỉ tập trung phạt sai số của lớp đúng.
* $\mathcal{L}_{uni}(u, \hat{y}) = -\frac{1}{10} \sum_{k=1}^{10} \log(\hat{y}_k)$: Phạt sai số đồng đều trên tất cả các lớp, khuyến khích mô hình duy trì khoảng cách Logits hợp lý.

---

## 3. Lý Do Khoa Học Lựa Chọn Label Smoothing cho ViT

Việc cấu hình thêm tham số `label_smoothing=0.1` là một quyết định kỹ thuật cực kỳ quan trọng đối với Vision Transformer vì các lý do sau:

### 3.1. Vision Transformer thiếu tính thiên vị quy nạp (Inductive Bias)
* Các mô hình CNN (như ResNet-50) có sẵn các giả định kiến trúc (inductive bias) như **tính bất biến dịch chuyển (translation invariance)** và **tính cục bộ không gian (locality)** (tức là các pixel gần nhau thì liên quan đến nhau).
* ViT không có các giả định này; nó coi ảnh như một chuỗi token phẳng và phải tự học toàn bộ các quan hệ không gian từ đầu. Do đó, ViT cực kỳ **nhạy cảm và dễ bị Overfitting (quá khớp)** khi huấn luyện trên các tập dữ liệu có quy mô vừa và nhỏ (như tập dữ liệu của nhóm).

### 3.2. Ngăn ngừa sự quá tự tin (Overconfidence) của mô hình
* Hàm Cross Entropy tiêu chuẩn sẽ phạt mô hình vô hạn nếu nó không đưa ra xác suất lớp đúng tiến sát về $1.0$ (tức logit $z_{correct} \rightarrow \infty$ và các logit khác tiến về $-\infty$). Điều này khiến mô hình học các trọng số cực lớn để tối đa hóa khoảng cách logits, dẫn đến việc mô hình trở nên quá tự tin vào dự đoán của mình.
* Khi mô hình quá tự tin, nó sẽ mất đi sự linh hoạt khi gặp các bức ảnh thực tế bị nhiễu (mưa, lóa nắng) trên tập Test.
* **Label Smoothing** giới hạn mục tiêu của lớp đúng ở mức **0.91**, ngăn không cho logits tiến tới vô hạn, giúp làm mượt mặt phẳng tối ưu (optimization landscape) và tăng khả năng tổng quát hóa (generalization) của ViT trên tập dữ liệu chưa từng thấy (Test set).
