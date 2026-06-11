# Geometric Augmentation

Tài liệu này mô tả geometric augmentation hiện đang được sử dụng trong quá trình tạo dữ liệu offline. Phiên bản hiện tại được gọi là **Geo Version 1** để làm mốc trước khi thiết kế và triển khai **Geo Version 2**.

## 1. Mục đích

Geo Version 1 được dùng để tạo thêm biến thể từ ảnh train khi số ảnh của một lớp chưa đủ quota. Nó không được áp dụng cho `valid_unseen`, `valid_traincopy` hoặc `test`.

Mã nguồn chính:

- `src/augment_offline.py`: phân bổ quota và thực hiện geometric augmentation.
- `src/image_pipelines/base.py`: resize giữ tỷ lệ và zero-padding ảnh về `224 x 224`.

## 2. Các phép biến đổi trong Geo Version 1

### 2.1. Lật ngang

- Xác suất áp dụng: `50%`.
- Thực hiện bằng `cv2.flip(image, 1)`.
- Đảo vị trí trái và phải của toàn bộ ảnh.

### 2.2. Affine transform

Affine transform luôn được áp dụng cho ảnh được đánh dấu geometric. Các tham số được lấy ngẫu nhiên như sau:

| Tham số | Phạm vi |
|---|---:|
| Góc xoay | `-15` đến `+15` độ |
| Tỷ lệ scale | `0.9` đến `1.1` |
| Dịch ngang | `-6%` đến `+6%` chiều rộng |
| Dịch dọc | `-6%` đến `+6%` chiều cao |

Phép biến đổi được thực hiện quanh tâm ảnh bằng `cv2.getRotationMatrix2D` và `cv2.warpAffine`. Những vùng nằm ngoài ảnh sau biến đổi được tô màu đen.

### 2.3. Perspective transform

- Xác suất áp dụng: `35%`.
- Mỗi góc ảnh được dịch ngẫu nhiên tối đa `6%` chiều rộng hoặc chiều cao.
- Thực hiện bằng `cv2.getPerspectiveTransform` và `cv2.warpPerspective`.
- Những vùng trống sau biến đổi được tô màu đen.

Phép biến đổi này mô phỏng thay đổi nhẹ về góc quan sát.

### 2.4. Điều chỉnh độ tương phản và độ sáng

Phép điều chỉnh này luôn được áp dụng:

| Tham số | Phạm vi | Ý nghĩa |
|---|---:|---|
| `alpha` | `0.85` đến `1.2` | Điều chỉnh contrast |
| `beta` | `-20` đến `+20` | Điều chỉnh brightness |

Công thức tổng quát:

```text
output = abs(alpha * image + beta)
```

Phần này là content/photometric augmentation, không phải geometric augmentation theo nghĩa chặt chẽ, nhưng hiện đang được đặt trong hàm `apply_geometric_augmentation()`.

### 2.5. Random erasing

- Xác suất áp dụng: `35%`.
- Chọn một vùng chữ nhật ngẫu nhiên và tô vùng đó thành màu đen.
- Chiều rộng vùng xóa nằm trong khoảng `1/14` đến `1/5` chiều rộng ảnh.
- Chiều cao vùng xóa nằm trong khoảng `1/14` đến `1/5` chiều cao ảnh.

Phần này mô phỏng che khuất một phần phương tiện. Nó cũng không phải geometric augmentation theo nghĩa chặt chẽ nhưng hiện nằm chung trong Geo Version 1.

## 3. Thứ tự xử lý ảnh

Một ảnh geometric được xử lý theo thứ tự:

```text
Ảnh gốc
  -> chuyển sang RGB
  -> resize giữ nguyên tỷ lệ
  -> zero-padding thành 224 x 224
  -> Geo Version 1
  -> resize và zero-padding lại thành 224 x 224
  -> áp pipeline normal, rain, sun hoặc night
  -> lưu ảnh JPEG
```

Việc chạy base pipeline lần thứ hai bảo đảm ảnh đầu ra vẫn có kích thước `224 x 224` sau geometric augmentation.

## 4. Quan hệ với quota fill policy

Mỗi lớp có phân phối mục tiêu:

| Bucket | Tỷ lệ |
|---|---:|
| `normal` | `70%` |
| `rain` | `10%` |
| `sun` | `10%` |
| `night` | `10%` |

### 4.1. Lớp đã đạt hoặc vượt target

- Chỉ lấy số ảnh bằng target.
- `70%` đầu được đưa vào bucket `normal` dưới dạng ảnh gốc.
- `30%` còn lại được dùng làm ảnh nguồn gốc cho các bucket thời tiết.
- Không tạo dư thêm ảnh geometric.

### 4.2. Lớp có từ 70% đến dưới 100% target

- Dùng ảnh gốc để lấp đầy bucket `normal` đến `70%`.
- Các ảnh gốc còn lại được ưu tiên phân bổ cho `rain`, `sun` và `night`.
- Những slot thời tiết còn thiếu sẽ chọn lại ngẫu nhiên ảnh nguồn và áp Geo Version 1 trước khi áp hiệu ứng thời tiết.

### 4.3. Lớp có dưới 70% target

- Toàn bộ ảnh gốc được đưa vào bucket `normal`.
- Chọn lặp ngẫu nhiên ảnh nguồn và áp Geo Version 1 để lấp bucket `normal` đến `70%`.
- Các bucket `rain`, `sun` và `night` tiếp tục chọn ảnh nguồn ngẫu nhiên, áp Geo Version 1, rồi mới áp hiệu ứng môi trường.

## 5. Nhận biết ảnh đầu ra

Tên ảnh có cấu trúc:

```text
<source>_<bucket>_<orig|geo>_<index>.jpg
```

Ví dụ:

```text
car_001_normal_orig_00001.jpg
car_001_normal_geo_00002.jpg
car_001_rain_geo_00003.jpg
```

- `_orig_`: ảnh nguồn không đi qua geometric augmentation.
- `_geo_`: ảnh nguồn đã đi qua Geo Version 1.

## 6. Tính tái lập

Quá trình augmentation sử dụng `random.Random` với seed mặc định là `42`. Với cùng dữ liệu đầu vào, cùng target và cùng seed, việc chọn ảnh cũng như các tham số biến đổi sẽ có thể tái lập.

## 7. Giới hạn của Geo Version 1

- Gộp geometric, brightness/contrast và random erasing vào cùng một hàm.
- Mỗi ảnh geometric luôn phải trải qua affine transform; không có xác suất bỏ qua riêng cho affine.
- Viền đen có thể xuất hiện sau affine và perspective transform.
- Horizontal flip có thể không phù hợp với một số ảnh chứa chữ hoặc biển báo có ý nghĩa định hướng.
- Random erasing dùng vùng đen hoàn toàn, có thể tạo đặc trưng nhân tạo không giống che khuất thực tế.
- Chưa có cấu hình riêng theo loại phương tiện hoặc theo bucket môi trường.
- Chưa ghi metadata chi tiết về các phép biến đổi đã áp dụng cho từng ảnh.

## 8. Geo Version 2

Đặc tả Geo Version 2 nằm trong [GEO_AUGMENTATION_V2.md](GEO_AUGMENTATION_V2.md).

Version 2 giữ nguyên Geo Version 1, quota fill policy và luồng xử lý hiện tại. Khác biệt duy nhất là `30%` dữ liệu thuộc các bucket thời tiết được thay thế như sau:

| Version 1 | Version 2 | Tỷ lệ |
|---|---|---:|
| `rain` | `gaussian_blur` | `10%` |
| `sun` | `motion_blur` | `10%` |
| `night` | `unsharp_mask` | `10%` |
