# 🎨 Tăng Cường Dữ Liệu Ngoại Tuyến (Offline Augmentation)

> [!NOTE]  
> Sau bước cân bằng dữ liệu, tập Train đạt 71,170 ảnh. Để nâng cao khả năng chống chịu nhiễu môi trường thực tế, tập này được đưa qua 4 pipelines biến đổi vật lý lưu trực tiếp lên ổ đĩa cứng, sinh ra tổng số **284,680 ảnh** huấn luyện thực tế.

---

## Các Pipeline Tăng Cường Vật Lý

### A. Pipeline Base (`_base.jpg`)
* **Mô tả**: Chuẩn hóa ảnh về độ phân giải vuông 224x224, sử dụng thuật toán **Zero Padding** để bảo toàn tỷ lệ khung hình gốc (Aspect Ratio) của phương tiện giao thông (tránh làm méo mó xe tải hoặc xe khách). Sau đó, áp dụng các toán tử hình thái học đóng (**Closing**) và mở (**Opening**) để làm sạch nhiễu hạt nền.

```python
def apply_pipeline_base(img):
    h, w = img.shape[:2]
    scale = 224 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224 - new_h) // 2, (224 - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)
```

---

### B. Pipeline Night (`_night.jpg`)
* **Mô tả**: Giả lập môi trường thiếu sáng hoặc ban đêm bằng cách điều chỉnh hệ số Gamma chiếu sáng, đồng thời chèn thêm nhiễu phân phối chuẩn (**Gaussian Noise/ISO Noise**) và làm suy giảm độ bão hòa màu sắc (Saturation) xuống còn 50%.

```python
def apply_pipeline_night(img_base):
    gamma = random.uniform(0.4, 0.7)
    img_dark = np.clip(np.power(img_base / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    
    noise = np.random.normal(0, 15, img_dark.shape)
    img_noisy = np.clip(img_dark + noise, 0, 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = (hsv[:, :, 1] * 0.5).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

---

### C. Pipeline Rain (`_rain.jpg`)
* **Mô tả**: Mô phỏng hiệu ứng mưa rơi bằng cách xoay góc và lọc tích chập chuyển động tạo vệt mờ (**Motion Blur**), kết hợp các đốm nước đọng trên ống kính bằng bộ lọc trung vị (**Median Blur**). Cuối cùng, vẽ các nét vẽ thẳng màu xám sáng ngẫu nhiên giả lập hạt mưa rơi.

```python
def apply_pipeline_rain(img_base):
    angle, kernel_size = random.uniform(-10, 10), random.randint(7, 15)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, kernel_size // 2] = 1 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    img_rain = cv2.medianBlur(cv2.filter2D(img_base, -1, kernel), 3)
    
    for _ in range(random.randint(100, 300)):
        x, y, length = random.randint(0, 223), random.randint(0, 223), random.randint(3, 8)
        cv2.line(img_rain, (x, y), (x, y + length), (200, 200, 200), 1)
    return img_rain
```

---

### D. Pipeline Sun (`_sun.jpg`)
* **Mô tả**: Giả lập hiệu ứng lóa nắng gắt bằng cách vẽ chèn luồng sáng tròn đồng tâm (**Sun Flare**) màu trắng đỏ, tạo bóng đổ hình học che khuất vật lý ngẫu nhiên thông qua đa giác màu tối và tăng độ tương phản kết hợp độ sáng tổng thể của bức ảnh.

```python
def apply_pipeline_sun(img_base):
    flare_x, flare_y = random.randint(50, 174), random.randint(50, 174)
    flare_radius = random.randint(30, 60)
    img_flare = img_base.copy()
    for r in range(flare_radius, 0, -5):
        intensity = int(255 * (r / flare_radius))
        cv2.circle(img_flare, (flare_x, flare_y), r, (intensity, intensity, 255), -1)
    img_blend = cv2.addWeighted(img_base, 0.7, img_flare, 0.3, 0)
    
    shadow_mask = np.ones_like(img_blend) * 255
    pts = np.array([[random.randint(0, 224), random.randint(0, 224)] for _ in range(3)], np.int32)
    cv2.fillPoly(shadow_mask, [pts], (50, 50, 50))
    return cv2.convertScaleAbs(cv2.addWeighted(img_blend, 0.8, shadow_mask, 0.2, 0), alpha=1.2, beta=10)
```
