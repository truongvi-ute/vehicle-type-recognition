# Geo Augmentation Version 2

## 1. Tổng quan

Geo Augmentation Version 2 giữ nguyên toàn bộ cách phân bổ quota, cách chọn ảnh nguồn và Geo Version 1 đang dùng trong pipeline hiện tại. Khác biệt duy nhất là `30%` dữ liệu trước đây thuộc ba pipeline môi trường `rain`, `sun` và `night` được thay bằng ba pipeline mô phỏng chất lượng lấy nét của ảnh:

- `gaussian_blur`
- `motion_blur`
- `unsharp_mask`

Phân phối mục tiêu cho mỗi lớp:

| Bucket | Tỷ lệ |
|---|---:|
| `normal` | `70%` |
| `gaussian_blur` | `10%` |
| `motion_blur` | `10%` |
| `unsharp_mask` | `10%` |

Tổng số ảnh sau augmentation của mỗi lớp vẫn bằng `target_per_class`.

Ánh xạ trực tiếp từ Version 1 sang Version 2:

| Version 1 | Version 2 | Tỷ lệ |
|---|---|---:|
| `normal` | `normal` | `70%` |
| `rain` | `gaussian_blur` | `10%` |
| `sun` | `motion_blur` | `10%` |
| `night` | `unsharp_mask` | `10%` |

> Ghi chú: Gaussian blur, motion blur và unsharp mask là các phép biến đổi chất lượng ảnh, không phải geometric augmentation theo nghĩa chặt chẽ. Tên "Geo Version 2" được giữ để phù hợp với cách đặt tên hiện tại của dự án.

## 2. Mục tiêu

Version 2 hướng đến khả năng nhận dạng phương tiện ổn định hơn trong các trường hợp:

- Camera bị mất nét nhẹ.
- Phương tiện hoặc camera đang chuyển động.
- Ảnh đã được camera hoặc phần mềm tăng độ sắc nét.
- Chất lượng ảnh đầu vào khác với ảnh train gốc.

Version 2 không mô phỏng điều kiện thời tiết và không kết hợp với `rain`, `sun` hoặc `night`.

## 3. Nguyên tắc chung

- Giữ nguyên Geo Version 1: lật ngang, affine transform, perspective transform, brightness/contrast và random erasing với cùng tham số, xác suất và seed.
- Giữ nguyên quota fill policy của Version 1; chỉ thay pipeline được áp dụng sau bước Geo Version 1 cho `30%` dữ liệu ngoài bucket `normal`.
- Mỗi ảnh đầu ra chỉ thuộc một bucket.
- Một ảnh chỉ nhận tối đa một phép xử lý V2.
- Không áp blur và sharpen liên tiếp trên cùng một ảnh.
- Không áp các pipeline V2 lên ảnh đã qua `rain`, `sun` hoặc `night`.
- Ảnh phải giữ nguyên kích thước đầu ra `224 x 224`.
- Mức biến đổi phải nhẹ hoặc trung bình để phương tiện vẫn nhận diện được.
- Các tham số ngẫu nhiên phải sử dụng seed để có thể tái lập kết quả.
- Validation và test không được augmentation.

## 4. Pipeline Normal

Tỷ lệ mục tiêu: `70%`.

Pipeline `normal` chỉ thực hiện base preprocessing:

```text
Ảnh gốc
  -> chuyển sang RGB
  -> resize giữ nguyên tỷ lệ
  -> zero-padding thành 224 x 224
  -> lưu ảnh
```

Không áp blur, sharpen hoặc hiệu ứng thời tiết.

Với ảnh gốc được phân vào `normal`, pipeline trên được dùng trực tiếp. Nếu lớp thiếu ảnh và cần lấp thêm quota `normal`, ảnh nguồn được áp Geo Version 1 giống chính sách hiện tại của Version 1; sau đó không áp thêm Gaussian blur, motion blur hoặc unsharp mask.

## 5. Gaussian Blur

Tỷ lệ mục tiêu: `10%`.

Gaussian blur làm mờ ảnh theo phân phối Gaussian. Pipeline này mô phỏng mất nét nhẹ, độ phân giải quang học thấp hoặc quá trình resize/compression làm giảm chi tiết.

### Tham số khởi đầu đề xuất

| Tham số | Giá trị |
|---|---:|
| Kernel size | chọn ngẫu nhiên `3 x 3` hoặc `5 x 5` |
| Sigma | `0.4` đến `1.4` |
| Border mode | reflect hoặc replicate |

Ví dụ triển khai bằng OpenCV:

```python
output = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
```

Không nên dùng kernel quá lớn hoặc sigma quá cao vì có thể xóa mất các đặc trưng phân biệt giữa những loại phương tiện gần giống nhau.

## 6. Motion Blur

Tỷ lệ mục tiêu: `10%`.

Motion blur mô phỏng chuyển động tương đối giữa phương tiện và camera. Đây là tình huống phổ biến trong ảnh giao thông.

### Tham số khởi đầu đề xuất

| Tham số | Giá trị |
|---|---:|
| Kernel length | số lẻ từ `3` đến `7` |
| Góc chuyển động | `0` đến `180` độ |
| Cường độ | chuẩn hóa tổng kernel bằng `1.0` |
| Border mode | reflect hoặc replicate |

Quy trình tổng quát:

```text
Tạo kernel đường thẳng
  -> xoay kernel theo góc ngẫu nhiên
  -> chuẩn hóa kernel
  -> convolution với ảnh
```

Kernel không nên dài quá mức vì ảnh `224 x 224` có thể mất gần hết chi tiết của phương tiện.

## 7. Unsharp Mask

Tỷ lệ mục tiêu: `10%`.

Unsharp mask tăng độ sắc nét bằng cách lấy ảnh gốc cộng với phần chi tiết chênh lệch giữa ảnh gốc và ảnh đã làm mờ.

Công thức:

```text
blurred = GaussianBlur(image)
detail = image - blurred
output = image + amount * detail
```

### Tham số khởi đầu đề xuất

| Tham số | Giá trị |
|---|---:|
| Gaussian kernel | `3 x 3` hoặc `5 x 5` |
| Sigma | `0.5` đến `1.2` |
| Amount | `0.3` đến `0.8` |
| Threshold tùy chọn | `0` đến `5` |

Giá trị đầu ra phải được giới hạn trong khoảng `[0, 255]` trước khi chuyển về `uint8`.

Không nên dùng amount quá lớn vì có thể tạo halo quanh cạnh, khuếch đại nhiễu và làm ảnh khác xa dữ liệu thực tế.

## 8. Thứ tự xử lý

Ảnh Version 2 được xử lý theo thứ tự:

```text
Ảnh nguồn
  -> chuyển sang RGB
  -> resize giữ nguyên tỷ lệ
  -> zero-padding thành 224 x 224
  -> nếu slot cần geometric augmentation: áp Geo Version 1
  -> resize và zero-padding lại thành 224 x 224
  -> chọn đúng một bucket
       -> normal: không áp thêm pipeline V2
       -> gaussian_blur: áp Gaussian blur
       -> motion_blur: áp motion blur
       -> unsharp_mask: áp unsharp mask
  -> kiểm tra kích thước và kiểu dữ liệu
  -> lưu ảnh JPEG
```

Không có bước áp pipeline thời tiết sau V2.

Luồng này tương ứng trực tiếp với Version 1: bước cuối `rain`, `sun` hoặc `night` được thay bằng `gaussian_blur`, `motion_blur` hoặc `unsharp_mask`; các bước còn lại không đổi.

## 9. Quota Fill Policy

Giả sử target của mỗi lớp là `T`:

```text
normal_target         = round(T * 0.70)
gaussian_blur_target  = round(T * 0.10)
motion_blur_target    = round(T * 0.10)
unsharp_mask_target   = T - các bucket còn lại
```

Cách tính bucket cuối bằng phần còn lại giúp tổng số ảnh luôn đúng bằng target dù có sai số làm tròn.

### 9.1. Lớp đã đạt hoặc vượt target

- Trộn ngẫu nhiên danh sách ảnh nguồn.
- Giới hạn số ảnh được sử dụng bằng target.
- Phân bổ `70%` ảnh vào `normal`.
- Phân bổ `10%` vào mỗi bucket V2.
- Không tạo thêm ảnh ngoài target.

### 9.2. Lớp có từ 70% đến dưới 100% target

- Dùng ảnh gốc để lấp bucket `normal` đến `70%` target.
- Các ảnh gốc còn lại được ưu tiên dùng làm nguồn cho ba bucket V2.
- Nếu bucket V2 còn thiếu, chọn lặp ngẫu nhiên từ ảnh nguồn của lớp, áp Geo Version 1, rồi áp đúng pipeline V2 của bucket đó.

### 9.3. Lớp có dưới 70% target

- Đưa toàn bộ ảnh gốc vào bucket `normal`.
- Chọn lặp ngẫu nhiên ảnh nguồn, áp Geo Version 1 để lấp bucket `normal` đến `70%` target.
- Với `gaussian_blur`, `motion_blur` và `unsharp_mask`, chọn lặp ngẫu nhiên ảnh nguồn, áp Geo Version 1, rồi áp pipeline V2 tương ứng đến khi đủ quota.

## 10. Quy tắc đặt tên file

Cấu trúc giữ nguyên như Version 1:

```text
<source>_<bucket>_<orig|geo>_<index>.jpg
```

Ví dụ:

```text
car_001_normal_orig_00001.jpg
car_001_gaussian_blur_geo_00001.jpg
car_001_motion_blur_geo_00001.jpg
car_001_unsharp_mask_geo_00001.jpg
```

- `_orig_`: ảnh nguồn không đi qua Geo Version 1 trước pipeline bucket.
- `_geo_`: ảnh nguồn đã đi qua Geo Version 1 trước pipeline bucket.

Nếu cần phân biệt rõ phiên bản augmentation:

```text
<source>_v2_<bucket>_<index>.jpg
```

## 11. Metadata đề xuất

Mỗi ảnh được tạo nên có metadata trong file JSON hoặc JSON Lines:

```json
{
  "source": "car_001.jpg",
  "output": "car_001_v2_motion_blur_00001.jpg",
  "version": "v2",
  "bucket": "motion_blur",
  "parameters": {
    "kernel_length": 5,
    "angle": 37.5
  },
  "seed": 42
}
```

Metadata giúp kiểm tra ảnh lỗi, tái lập dữ liệu và so sánh ảnh hưởng của từng thuật toán.

## 12. Kiểm tra chất lượng

Trước khi dùng dữ liệu V2 để train, cần kiểm tra:

- Mỗi lớp có đúng tổng target và đúng tỷ lệ `70/10/10/10`.
- Không có ảnh V2 nằm trong validation hoặc test.
- Ảnh đầu ra có kích thước `224 x 224`, ba kênh màu và kiểu `uint8`.
- Gaussian blur không xóa mất hình dạng chính của phương tiện.
- Motion blur không quá dài hoặc quá mạnh.
- Unsharp mask không tạo halo lớn và không làm cháy vùng sáng.
- Không có ảnh nào đồng thời nhận nhiều pipeline V2.
- Cùng seed và dữ liệu đầu vào phải tạo lại cùng kết quả.

Nên xuất một contact sheet ngẫu nhiên cho từng lớp và từng bucket để kiểm tra trực quan trước khi train.

## 13. Đánh giá Version 2

Version 2 nên được so sánh với Version 1 trong cùng điều kiện:

- Cùng train/validation/test split.
- Cùng target mỗi lớp.
- Cùng kiến trúc mô hình và pretrained weights.
- Cùng số epoch, optimizer và learning rate schedule.
- So sánh accuracy, macro F1 và F1 theo từng lớp trên `valid_unseen` và `test`.

Không dùng `valid_traincopy` làm tiêu chí chính để kết luận Version 2 tốt hơn.

## 14. Trạng thái triển khai

Geo Augmentation Version 2 đã có ba pipeline xử lý ảnh và script tạo dữ liệu riêng trong mã nguồn. Bộ dữ liệu `data/augmented_v2` chỉ được tạo khi chạy `src/augment_offline_v2.py`.

Version 2 phải được triển khai tách biệt với Version 1 để có thể train và so sánh hai bộ dữ liệu trong cùng điều kiện. Không sửa nội dung của `data/splits` và không ghi đè `data/augmented`.

## 15. Cấu trúc mã nguồn cần bổ sung

Thêm ba file xử lý ảnh vào package dùng chung `src/image_pipelines`:

```text
src/image_pipelines/
  __init__.py
  base.py
  rain.py
  sun.py
  night.py
  gaussian_blur.py       # mới cho Geo Version 2
  motion_blur.py         # mới cho Geo Version 2
  unsharp_mask.py        # mới cho Geo Version 2
```

Thêm một script augmentation riêng cho Version 2:

```text
src/augment_offline_v2.py
```

Không thay thế hoặc đổi hành vi của `src/augment_offline.py`. File hiện tại tiếp tục là implementation của Version 1 với các bucket `normal`, `rain`, `sun` và `night`.

### 15.1. `gaussian_blur.py`

Trách nhiệm:

- Nhận một `PIL.Image.Image` ở mode `RGB`.
- Chọn kernel và sigma bằng RNG có seed.
- Áp Gaussian blur với mức nhẹ hoặc trung bình.
- Trả về ảnh `RGB` có cùng kích thước đầu vào.

API dự kiến:

```python
def apply_gaussian_blur_pipeline(
    image: Image.Image,
    seed: int = 42,
) -> Image.Image:
    ...
```

### 15.2. `motion_blur.py`

Trách nhiệm:

- Nhận một `PIL.Image.Image` ở mode `RGB`.
- Tạo motion kernel có chiều dài lẻ từ `3` đến `7`.
- Xoay kernel theo góc ngẫu nhiên từ `0` đến `180` độ.
- Chuẩn hóa kernel trước khi convolution.
- Trả về ảnh `RGB` có cùng kích thước đầu vào.

API dự kiến:

```python
def apply_motion_blur_pipeline(
    image: Image.Image,
    seed: int = 42,
) -> Image.Image:
    ...
```

### 15.3. `unsharp_mask.py`

Trách nhiệm:

- Nhận một `PIL.Image.Image` ở mode `RGB`.
- Tạo ảnh Gaussian blur để lấy thành phần chi tiết.
- Tăng độ sắc nét bằng amount ngẫu nhiên trong phạm vi đã định nghĩa.
- Giới hạn giá trị pixel trong `[0, 255]`.
- Trả về ảnh `RGB` có cùng kích thước đầu vào.

API dự kiến:

```python
def apply_unsharp_mask_pipeline(
    image: Image.Image,
    seed: int = 42,
) -> Image.Image:
    ...
```

## 16. Registry pipeline Version 2

Cập nhật `src/image_pipelines/__init__.py` để export thêm ba hàm mới, nhưng không thay đổi `SUPPORTED_PIPELINES` hiện tại của Version 1 và backend.

Thêm registry riêng cho Version 2:

```python
SUPPORTED_V2_PIPELINES = [
    "normal",
    "gaussian_blur",
    "motion_blur",
    "unsharp_mask",
]

def apply_v2_pipeline(
    image: Image.Image,
    pipeline: str | None,
    seed: int = 42,
) -> tuple[Image.Image, str]:
    ...
```

Quy tắc dispatch:

| Bucket | Hàm xử lý |
|---|---|
| `normal` | trả nguyên ảnh sau Base Pipeline |
| `gaussian_blur` | `apply_gaussian_blur_pipeline` |
| `motion_blur` | `apply_motion_blur_pipeline` |
| `unsharp_mask` | `apply_unsharp_mask_pipeline` |

Việc dùng registry riêng tránh làm frontend/backend hiện tại tự động nhận các pipeline V2 trước khi phần giao diện được thiết kế và kiểm thử.

## 17. Script `augment_offline_v2.py`

`src/augment_offline_v2.py` là entry point tạo bộ dữ liệu Geo Version 2. Script nên tái sử dụng các hàm ổn định từ Version 1 khi phù hợp, đặc biệt là:

- Khám phá class và đọc danh sách ảnh.
- Base Pipeline `224 x 224`.
- `apply_geometric_augmentation()` và toàn bộ tham số Geo Version 1.
- Cách tính `target_per_class`.
- Quota fill policy.
- Cách chọn ảnh gốc và ảnh cần geometric augmentation.
- Seed mặc định `42`.

Khác biệt duy nhất trong kế hoạch tạo ảnh là danh sách bucket và pipeline cuối:

```python
BUCKET_RATIOS = {
    "normal": 0.70,
    "gaussian_blur": 0.10,
    "motion_blur": 0.10,
    "unsharp_mask": 0.10,
}
```

Script không được gọi `data_split.py`. Nguồn đầu vào bắt buộc là kết quả split đã tồn tại:

```text
data/splits/train
data/splits/valid_unseen
data/splits/valid_traincopy
data/splits/test
```

CLI dự kiến:

```bash
python src/augment_offline_v2.py \
  --input_dir data/splits/train \
  --splits_dir data/splits \
  --output_dir data/augmented_v2 \
  --seed 42
```

Các tham số nên tương thích với Version 1:

```text
--input_dir
--output_dir
--splits_dir
--target_per_class
--seed
--stats_json
```

## 18. Cấu trúc dữ liệu đầu ra

Version 2 tạo một dataset root riêng:

```text
data/augmented_v2/
  train/
    bicycle/
    boat/
    bus/
    car/
    helicopter/
    minibus/
    motorcycle/
    taxi/
    train/
    truck/
  valid_unseen/
  valid_traincopy/
  test/
```

Quy tắc dữ liệu:

- `train`: được tạo lại bằng quota policy `70/10/10/10` của Version 2.
- `valid_unseen`: sao chép nguyên trạng từ `data/splits/valid_unseen`.
- `valid_traincopy`: sao chép nguyên trạng từ `data/splits/valid_traincopy`.
- `test`: sao chép nguyên trạng từ `data/splits/test`.
- Không augmentation validation và test.
- Không đọc ảnh train từ `data/augmented` của Version 1.
- Không ghi file vào `data/splits`.

Nhờ vậy hai thí nghiệm dùng chính xác cùng một phép chia dữ liệu:

```text
data/splits
  |-> augment_offline.py    -> data/augmented     (Geo Version 1)
  |-> augment_offline_v2.py -> data/augmented_v2  (Geo Version 2)
```

## 19. Yêu cầu tương đương giữa V1 và V2

Để phép so sánh có ý nghĩa, hai script phải giữ giống nhau các yếu tố sau:

| Thành phần | Version 1 | Version 2 |
|---|---|---|
| Nguồn split | `data/splits` | `data/splits` |
| Train source | `data/splits/train` | `data/splits/train` |
| Target mỗi class | giống nhau | giống nhau |
| Base Pipeline | resize + zero-pad `224 x 224` | giống V1 |
| Geo augmentation | Geo Version 1 | giữ nguyên Geo Version 1 |
| Normal quota | `70%` | `70%` |
| Ba bucket còn lại | rain, sun, night | Gaussian, motion, unsharp |
| Seed mặc định | `42` | `42` |
| Validation/test | không augmentation | không augmentation |

Nếu truyền cùng `target_per_class` và seed, số lượng ảnh theo class của hai dataset phải bằng nhau. Chỉ loại biến đổi trên `30%` ảnh ngoài bucket `normal` được phép khác nhau.

## 20. Kiểm tra sau triển khai

Sau khi tạo `data/augmented_v2`, cần bổ sung hoặc mở rộng công cụ kiểm tra để xác nhận:

1. Mỗi class có đúng `target_per_class` ảnh train.
2. Mỗi class có tỷ lệ `70% normal`, `10% gaussian_blur`, `10% motion_blur`, `10% unsharp_mask`.
3. Tổng ảnh của từng class bằng đúng bộ dữ liệu V1 tương ứng.
4. Tên ảnh chỉ chứa bucket V2, không chứa `rain`, `sun` hoặc `night`.
5. `valid_unseen`, `valid_traincopy` và `test` giống `data/splits` về tên file và số lượng.
6. Không có file nào trong `data/splits` bị sửa trong quá trình chạy.
7. Tất cả ảnh đầu ra đọc được, có mode RGB và kích thước `224 x 224`.
8. Chạy lại cùng seed tạo cùng kế hoạch chọn ảnh và cùng tham số augmentation.
9. Contact sheet cho từng bucket không có ảnh mất hoàn toàn đặc trưng phương tiện.

Nên tạo report riêng, ví dụ:

```text
outputs/geo_v2/data_prep_counts.json
```

## 21. Phạm vi thay đổi

Giai đoạn triển khai dữ liệu Geo Version 2 chỉ cần thay đổi hoặc thêm các file sau:

```text
src/image_pipelines/gaussian_blur.py
src/image_pipelines/motion_blur.py
src/image_pipelines/unsharp_mask.py
src/image_pipelines/__init__.py
src/augment_offline_v2.py
docs/GEO_AUGMENTATION_V2.md
```

Chưa cần thay đổi frontend, backend, training loop hoặc model architecture. Khi train thử nghiệm, chỉ cần đổi `--data_dir` từ `data/augmented` sang `data/augmented_v2` và giữ nguyên toàn bộ cấu hình huấn luyện còn lại.
