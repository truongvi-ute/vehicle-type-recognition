"""
src/data_prep.py
================
Pipeline tiền xử lý hình ảnh và phân chia dữ liệu cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Gồm 4 phần chính:
  1. Chuẩn hóa hình học: Resize + Padding viền đen (bảo toàn Aspect Ratio)
  2. Bộ lọc Miền Không Gian (Spatial Domain Filters):
       - Nhóm Làm trơn / Lọc thông thấp (Smoothing / Lowpass)
       - Nhóm Trung bình khử nhiễu (Mean Filters)
       - Nhóm Thống kê thứ tự / Phi tuyến (Order-Statistic / Nonlinear)
       - Nhóm Làm sắc nét / Lọc thông cao (Sharpening / Highpass)
  3. Custom Split Data theo yêu cầu đặc thù của Giảng viên:
       - Test  : 10% (hoàn toàn độc lập)
       - Train : 80%
       - Valid (Phần 1 – 5%): ảnh UNSEEN (không có trong Train/Test)
       - Valid (Phần 2 – 5%): COPY 5% ngẫu nhiên từ tập Train
  4. Hệ thống Dọn dẹp & Ghi đè Thông minh (Cleanup / Smart Overwrite):
       - Manifest JSON theo dõi toàn bộ ảnh đã xử lý
       - Bỏ qua ảnh đã xử lý đúng cấu hình (tiết kiệm thời gian)
       - Phát hiện ảnh thay đổi nội dung (MD5 hash)
       - Phát hiện khi đổi bộ lọc / kích thước ảnh → xử lý lại
       - Dọn sạch processed/ khi ảnh raw bị xóa (tránh rác)
       - Hỗ trợ 3 chế độ: smart | force | clean

Thư viện cần thiết:
  pip install opencv-python numpy scikit-learn tqdm
"""

import os
import shutil
import random
import hashlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SIZE   = 224        # Kích thước ảnh đầu ra (224×224)
RANDOM_SEED   = 42         # Seed để đảm bảo tái tạo kết quả
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FILE = ".pipeline_manifest.json"  # Tên file manifest lưu trong processed_dir

# Tỷ lệ phân chia
TEST_RATIO         = 0.10  # 10% toàn bộ tập dữ liệu
VALID_UNSEEN_RATIO = 0.05  # 5% ảnh hoàn toàn mới (không xuất hiện trong Train/Test)
VALID_COPY_RATIO   = 0.05  # 5% copy từ tập Train
# → Train = 80% còn lại sau khi trích Test + phần Unseen Valid


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 – CHUẨN HÓA HÌNH HỌC
# ─────────────────────────────────────────────────────────────────────────────

def resize_with_padding(img: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """
    Resize ảnh về kích thước (target × target) bằng cách:
      1. Scale ảnh gốc để cạnh dài nhất bằng `target`, giữ nguyên tỷ lệ.
      2. Pad phần còn lại bằng màu đen để tạo ảnh vuông.

    Tránh hiện tượng méo / biến dạng hình học của xe (xe bus, xe tải...).
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tạo canvas đen kích thước target × target
    canvas = np.zeros((target, target, 3), dtype=np.uint8)

    # Dán ảnh đã resize vào giữa canvas
    top  = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2 – CÁC BỘ LỌC MIỀN KHÔNG GIAN
# ─────────────────────────────────────────────────────────────────────────────

# ── 2A. Nhóm Làm trơn / Lọc thông thấp ──────────────────────────────────────

def apply_box_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc hộp (Box Filter / Averaging Filter).
    Thay thế giá trị pixel bằng trung bình cộng các pixel lân cận trong cửa sổ ksize×ksize.
    Làm mờ nhanh nhưng không phân biệt biên cạnh.
    """
    return cv2.blur(img, (ksize, ksize))


def apply_gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Lọc Gaussian (Gaussian Filter).
    Sử dụng trọng số phân phối chuẩn Gaussian — làm mịn tự nhiên hơn lọc hộp,
    loại bỏ nhiễu tần số cao hiệu quả, bảo toàn cấu trúc biên tốt hơn.
    sigma=0 → OpenCV tự tính từ ksize.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # ksize phải lẻ
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ── 2B. Nhóm Trung bình khử nhiễu (Mean Filters) ────────────────────────────

def _apply_per_channel(func, img: np.ndarray, **kwargs) -> np.ndarray:
    """Helper: áp dụng hàm lọc lên từng kênh màu R, G, B rồi gộp lại."""
    channels = cv2.split(img)
    processed = [func(ch, **kwargs) for ch in channels]
    return cv2.merge(processed)


def apply_arithmetic_mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc trung bình số học (Arithmetic Mean Filter).
    Làm mịn các biến động nhiễu cục bộ bằng cách tính trung bình vùng.
    Tương đương với Lọc hộp — thực hiện bằng cv2.blur.
    """
    return cv2.blur(img, (ksize, ksize))


def apply_geometric_mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc trung bình hình học (Geometric Mean Filter).
    Giữ lại nhiều chi tiết tinh tế hơn trung bình số học.
    Công thức: pixel = exp( mean(log(pixel+eps)) ) trong vùng lân cận.
    """
    def _geom_mean_channel(ch: np.ndarray, ksize: int) -> np.ndarray:
        log_ch = np.log(ch.astype(np.float64) + 1e-6)
        blurred = cv2.blur(log_ch, (ksize, ksize))
        result = np.exp(blurred)
        return np.clip(result, 0, 255).astype(np.uint8)

    return _apply_per_channel(_geom_mean_channel, img, ksize=ksize)


def apply_harmonic_mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc trung bình điều hòa (Harmonic Mean Filter).
    Tốt với nhiễu muối (salt noise), không hiệu quả với nhiễu tiêu (pepper noise).
    Công thức: pixel = N / sum(1 / (pixel+eps)) trong vùng lân cận.
    """
    def _harmonic_channel(ch: np.ndarray, ksize: int) -> np.ndarray:
        f = ch.astype(np.float64)
        inv = 1.0 / (f + 1e-6)
        inv_sum = cv2.blur(inv, (ksize, ksize)) * (ksize * ksize)
        result = (ksize * ksize) / (inv_sum + 1e-6)
        return np.clip(result, 0, 255).astype(np.uint8)

    return _apply_per_channel(_harmonic_channel, img, ksize=ksize)


def apply_contraharmonic_mean_filter(
    img: np.ndarray, ksize: int = 5, Q: float = 1.5
) -> np.ndarray:
    """
    Lọc trung bình Contraharmonic (Contraharmonic Mean Filter).
    - Q > 0: khử nhiễu tiêu (pepper noise)
    - Q < 0: khử nhiễu muối (salt noise)
    Công thức: pixel = sum(f^(Q+1)) / sum(f^Q) trong vùng lân cận.
    """
    def _contra_channel(ch: np.ndarray, ksize: int, Q: float) -> np.ndarray:
        f = ch.astype(np.float64) + 1e-6
        num = cv2.blur(np.power(f, Q + 1), (ksize, ksize))
        den = cv2.blur(np.power(f, Q), (ksize, ksize))
        result = num / (den + 1e-6)
        return np.clip(result, 0, 255).astype(np.uint8)

    return _apply_per_channel(_contra_channel, img, ksize=ksize, Q=Q)


# ── 2C. Nhóm Thống kê thứ tự / Phi tuyến (Order-Statistic / Nonlinear) ──────

def apply_median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc trung vị (Median Filter).
    Bộ lọc "quốc dân" loại bỏ nhiễu muối-tiêu (salt-and-pepper) cực hiệu quả
    mà không làm mờ các đường biên góc cạnh của xe.
    ksize phải là số lẻ.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)


def _sliding_window_stat(img: np.ndarray, ksize: int, stat_func) -> np.ndarray:
    """
    Helper: áp dụng hàm thống kê (max/min/midpoint) theo cửa sổ trượt
    lên từng kênh màu bằng dilate / erode.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    channels = cv2.split(img)
    processed = [stat_func(ch, kernel) for ch in channels]
    return cv2.merge(processed)


def apply_max_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc Max (Max Filter / Dilation).
    Chọn giá trị lớn nhất trong cửa sổ — tìm các điểm sáng cục bộ, khử nhiễu tiêu.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    return _sliding_window_stat(img, ksize, lambda ch, k: cv2.dilate(ch, k))


def apply_min_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc Min (Min Filter / Erosion).
    Chọn giá trị nhỏ nhất trong cửa sổ — tìm các điểm tối cục bộ, khử nhiễu muối.
    """
    return _sliding_window_stat(img, ksize, lambda ch, k: cv2.erode(ch, k))


def apply_midpoint_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Lọc điểm giữa (Midpoint Filter).
    Tính trung bình cộng giữa Max và Min trong cửa sổ.
    Kết hợp tối ưu cho nhiễu phân phối đồng đều (uniform noise).
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    channels = cv2.split(img)
    processed = []
    for ch in channels:
        mx = cv2.dilate(ch, kernel)
        mn = cv2.erode(ch, kernel)
        mid = ((mx.astype(np.float32) + mn.astype(np.float32)) / 2.0)
        processed.append(np.clip(mid, 0, 255).astype(np.uint8))
    return cv2.merge(processed)


def apply_alpha_trimmed_mean_filter(
    img: np.ndarray, ksize: int = 5, d: int = 2
) -> np.ndarray:
    """
    Lọc trung bình xén Alpha (Alpha-Trimmed Mean Filter).
    Loại bỏ d//2 pixel nhỏ nhất và lớn nhất ở hai đầu chuỗi đã sắp xếp
    trong cửa sổ rồi tính trung bình phần còn lại.
    Hiệu quả cao khi ảnh bị nhiễm hỗn hợp nhiều loại nhiễu.
    d phải chẵn, 0 <= d < ksize*ksize.
    """
    d = max(0, min(d, ksize * ksize - 1))
    trim = d // 2  # số pixel bị loại ở mỗi đầu

    def _alpha_trim_channel(ch: np.ndarray) -> np.ndarray:
        h, w = ch.shape
        pad = ksize // 2
        padded = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        out = np.zeros_like(ch, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                window = padded[i:i + ksize, j:j + ksize].flatten().astype(np.float64)
                window.sort()
                if trim > 0:
                    window = window[trim:-trim]
                out[i, j] = window.mean()

        return np.clip(out, 0, 255).astype(np.uint8)

    channels = cv2.split(img)
    processed = [_alpha_trim_channel(ch) for ch in channels]
    return cv2.merge(processed)


# ── 2D. Nhóm Làm sắc nét / Lọc thông cao (Sharpening / Highpass) ─────────────

def apply_laplacian_filter(img: np.ndarray) -> np.ndarray:
    """
    Toán tử Laplacian.
    Sử dụng đạo hàm bậc hai để phát hiện các thay đổi độ sáng đột ngột (tất cả hướng).
    Kết quả = ảnh gốc + mặt nạ Laplacian (tăng cường đường biên).
    """
    # Tính Laplacian trên từng kênh (ảnh xám hóa từng channel)
    channels = cv2.split(img)
    sharpened = []
    for ch in channels:
        lap = cv2.Laplacian(ch.astype(np.float64), cv2.CV_64F, ksize=3)
        result = np.clip(ch.astype(np.float64) - lap, 0, 255).astype(np.uint8)
        sharpened.append(result)
    return cv2.merge(sharpened)


def apply_sobel_gradient(
    img: np.ndarray, direction: str = "both"
) -> np.ndarray:
    """
    Toán tử Gradient Sobel.
    Sử dụng đạo hàm bậc nhất để tính độ dốc theo hướng ngang (X) và dọc (Y).
    direction: 'x' | 'y' | 'both'
    """
    channels = cv2.split(img)
    result_channels = []
    for ch in channels:
        f = ch.astype(np.float64)
        if direction in ("x", "both"):
            gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
        if direction in ("y", "both"):
            gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)

        if direction == "x":
            mag = np.abs(gx)
        elif direction == "y":
            mag = np.abs(gy)
        else:
            mag = np.sqrt(gx ** 2 + gy ** 2)

        result_channels.append(np.clip(mag, 0, 255).astype(np.uint8))
    return cv2.merge(result_channels)


def apply_scharr_gradient(img: np.ndarray) -> np.ndarray:
    """
    Toán tử Gradient Scharr.
    Cung cấp độ chính xác xấp xỉ đạo hàm tốt hơn Sobel ở các vùng biên chéo.
    """
    channels = cv2.split(img)
    result_channels = []
    for ch in channels:
        f = ch.astype(np.float64)
        gx = cv2.Scharr(f, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(f, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        result_channels.append(np.clip(mag, 0, 255).astype(np.uint8))
    return cv2.merge(result_channels)


def apply_roberts_gradient(img: np.ndarray) -> np.ndarray:
    """
    Toán tử Gradient Roberts.
    Sử dụng kernel 2×2 chéo — phát hiện biên theo hướng 45° và 135°.
    Nhạy cảm với nhiễu nhưng nhanh và đơn giản.
    """
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    channels = cv2.split(img)
    result_channels = []
    for ch in channels:
        f = ch.astype(np.float64)
        gx = cv2.filter2D(f, cv2.CV_64F, kernel_x)
        gy = cv2.filter2D(f, cv2.CV_64F, kernel_y)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        result_channels.append(np.clip(mag, 0, 255).astype(np.uint8))
    return cv2.merge(result_channels)


def apply_unsharp_masking(
    img: np.ndarray, ksize: int = 5, sigma: float = 1.0, amount: float = 1.0
) -> np.ndarray:
    """
    Mặt nạ làm mờ (Unsharp Masking).
    Trừ phiên bản làm mờ khỏi ảnh gốc để tạo mặt nạ biên,
    rồi cộng ngược lại → ảnh sắc nét rõ rệt.
    Công thức: sharpened = original + amount * (original - blurred)
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(img.astype(np.float64), (ksize, ksize), sigma)
    mask = img.astype(np.float64) - blurred
    sharpened = img.astype(np.float64) + amount * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_highboost_filter(
    img: np.ndarray, ksize: int = 5, sigma: float = 1.0, A: float = 2.0
) -> np.ndarray:
    """
    Lọc tăng cường cao (Highboost Filter).
    Biến thể nâng cao của Unsharp Masking.
    Nhân thêm hệ số A cho ảnh gốc → giữ lại nền tốt hơn, tăng mạnh độ nét biên.
    Công thức: highboost = A * original - blurred = (A-1)*original + unsharp_mask
    A > 1: tăng cường tất cả tần số; A = 1 → tương đương Unsharp Masking.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(img.astype(np.float64), (ksize, ksize), sigma)
    highboost = A * img.astype(np.float64) - blurred
    return np.clip(highboost, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# API CÔNG KHAI: Áp dụng bộ lọc theo tên
# ─────────────────────────────────────────────────────────────────────────────

FILTER_REGISTRY = {
    # Lowpass / Smoothing
    "box":              apply_box_filter,
    "gaussian":         apply_gaussian_filter,
    # Mean
    "arithmetic_mean":  apply_arithmetic_mean_filter,
    "geometric_mean":   apply_geometric_mean_filter,
    "harmonic_mean":    apply_harmonic_mean_filter,
    "contraharmonic":   apply_contraharmonic_mean_filter,
    # Order-Statistic / Nonlinear
    "median":           apply_median_filter,
    "max":              apply_max_filter,
    "min":              apply_min_filter,
    "midpoint":         apply_midpoint_filter,
    "alpha_trimmed":    apply_alpha_trimmed_mean_filter,
    # Sharpening / Highpass
    "laplacian":        apply_laplacian_filter,
    "sobel":            apply_sobel_gradient,
    "scharr":           apply_scharr_gradient,
    "roberts":          apply_roberts_gradient,
    "unsharp":          apply_unsharp_masking,
    "highboost":        apply_highboost_filter,
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CHAINS (Chuỗi Tiền Xử Lý)
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_CHAINS = {
    # 1. Baseline: Giữ nguyên bản chất gốc, chỉ chuẩn hóa kích thước (Khuyên dùng mặc định)
    "baseline_v1": [],
    
    # 2. Xử lý nhiễu hạt nhẹ + Làm nét (Thích hợp cho ảnh camera giao thông mờ)
    "denoise_sharpen_v1": [
        {"name": "median", "kwargs": {"ksize": 3}},
        {"name": "unsharp", "kwargs": {"ksize": 5, "amount": 1.2}}
    ],
    
    # 3. Phục hồi ảnh tối/chói + Giảm nhiễu (Dành cho bộ dữ liệu có nhiều ảnh ban đêm/ngược sáng)
    "night_vision_v1": [
        {"name": "gaussian", "kwargs": {"ksize": 3}},
        {"name": "highboost", "kwargs": {"A": 1.5, "sigma": 1.0}}
    ],
    
    # 4. Cứu hộ nhiễu nặng (Dành cho ảnh chất lượng cực thấp)
    "heavy_rescue_v1": [
        {"name": "alpha_trimmed", "kwargs": {"ksize": 3, "d": 2}},
        {"name": "laplacian", "kwargs": {}}
    ]
}


def apply_filter(img: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """
    Áp dụng bộ lọc theo tên từ FILTER_REGISTRY.

    Ví dụ:
        filtered = apply_filter(img, "gaussian", ksize=5, sigma=1.0)
        filtered = apply_filter(img, "median", ksize=3)
        filtered = apply_filter(img, "highboost", A=2.0)
    """
    if filter_name not in FILTER_REGISTRY:
        available = list(FILTER_REGISTRY.keys())
        raise ValueError(
            f"Bộ lọc '{filter_name}' không tồn tại.\n"
            f"Các bộ lọc khả dụng: {available}"
        )
    return FILTER_REGISTRY[filter_name](img, **kwargs)


def preprocess_image(
    img_path: str,
    pipeline_name: str = "baseline_v1",
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """
    Pipeline tiền xử lý hoàn chỉnh áp dụng Chuỗi Bộ Lọc (Chaining):
      1. Đọc ảnh (BGR)
      2. Resize + Padding viền đen (bảo toàn Aspect Ratio)
      3. Chạy qua chuỗi các bộ lọc đã định nghĩa trong PIPELINE_CHAINS
      4. Trả về ảnh RGB numpy array
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

    # Bước 1: Chuẩn hóa hình học
    current_img = resize_with_padding(img_bgr, target=target_size)

    # Bước 2: Chạy qua băng chuyền Pipeline
    if pipeline_name not in PIPELINE_CHAINS:
        raise ValueError(f"Pipeline '{pipeline_name}' không tồn tại. Chọn từ: {list(PIPELINE_CHAINS.keys())}")
        
    chain = PIPELINE_CHAINS[pipeline_name]
    for step in chain:
        filter_name = step["name"]
        kwargs = step.get("kwargs", {})
        # Đầu ra của bộ lọc trước là đầu vào của bộ lọc sau
        current_img = apply_filter(current_img, filter_name, **kwargs)

    # Bước 3: Chuyển BGR → RGB
    img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)

    return img_rgb


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3 – CUSTOM SPLIT DATA ĐẶC THÙ
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4 – HỆ THỐNG DỌN DẸP & GHI ĐÈ THÔNG MINH
# ─────────────────────────────────────────────────────────────────────────────

def _file_md5(path: str, chunk_size: int = 65536) -> str:
    """
    Tính MD5 hash của file để phát hiện thay đổi nội dung ảnh.
    Đọc theo chunk để xử lý file lớn mà không tốn RAM.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class ManifestManager:
    """
    Quản lý file manifest JSON để theo dõi trạng thái xử lý của toàn bộ tập dữ liệu.

    Cấu trúc manifest:
    {
      "_meta": {
        "filter": "gaussian",       # Bộ lọc đã dùng
        "target_size": 224,         # Kích thước ảnh
        "seed": 42,                 # Random seed
        "created_at": "...",        # Lần đầu tạo
        "updated_at": "..."         # Lần cập nhật gần nhất
      },
      "files": {
        "Bus/bus_000001.jpg": {
          "md5": "abc123...",         # Hash nội dung file raw
          "split": "train",           # Thuộc tập nào
          "out_name": "bus_000001.jpg",# Tên file đầu ra
          "processed_at": "..."       # Thời điểm xử lý
        },
        ...
      }
    }
    """

    def __init__(self, processed_dir: str):
        self.manifest_path = os.path.join(processed_dir, MANIFEST_FILE)
        self._data: dict = self._load()

    # ── I/O ────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Đọc manifest từ disk, trả về dict rỗng nếu chưa tồn tại."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print("  [WARN] Manifest bị lỗi, tạo lại từ đầu.")
        return {"_meta": {}, "files": {}}

    def save(self) -> None:
        """Ghi manifest xuống disk (atomic write qua file tạm)."""
        self._data["_meta"]["updated_at"] = datetime.now().isoformat()
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.manifest_path)  # atomic

    # ── Metadata config ────────────────────────────────────────────────────

    def get_meta(self) -> dict:
        return self._data.get("_meta", {})

    def set_meta(self, pipeline_name: str, target_size: int, seed: int) -> None:
        meta = self._data["_meta"]
        if not meta.get("created_at"):
            meta["created_at"] = datetime.now().isoformat()
        meta["pipeline"]    = pipeline_name
        meta["target_size"] = target_size
        meta["seed"]        = seed

    def config_changed(self, pipeline_name: str, target_size: int, seed: int) -> bool:
        """
        Kiểm tra xem cấu hình pipeline có thay đổi so với lần chạy trước không.
        Nếu có → toàn bộ ảnh cần xử lý lại.
        """
        meta = self.get_meta()
        return (
            meta.get("pipeline")    != pipeline_name  or
            meta.get("target_size") != target_size  or
            meta.get("seed")        != seed
        )

    # ── File tracking ──────────────────────────────────────────────────────

    def get_file_record(self, rel_key: str) -> dict | None:
        """Lấy bản ghi của 1 file raw (key = 'ClassName/filename.jpg')."""
        return self._data["files"].get(rel_key)

    def set_file_record(
        self,
        rel_key: str,
        md5: str,
        split: str,
        out_name: str,
    ) -> None:
        """Ghi / cập nhật bản ghi cho 1 file."""
        self._data["files"][rel_key] = {
            "md5":          md5,
            "split":        split,
            "out_name":     out_name,
            "processed_at": datetime.now().isoformat(),
        }

    def all_keys(self) -> set:
        """Tập hợp tất cả rel_key đang được theo dõi."""
        return set(self._data["files"].keys())

    def remove_key(self, rel_key: str) -> None:
        """Xóa bản ghi khỏi manifest."""
        self._data["files"].pop(rel_key, None)

    def needs_processing(
        self,
        rel_key: str,
        raw_path: str,
        processed_dir: str,
    ) -> tuple[bool, str]:
        """
        Kiểm tra xem file có cần xử lý lại không.

        Returns:
            (True, lý_do) nếu cần xử lý lại
            (False, "")   nếu đã xử lý và còn nguyên vẹn
        """
        record = self.get_file_record(rel_key)
        if record is None:
            return True, "mới"

        # Kiểm tra file processed còn tồn tại không
        split    = record["split"]
        out_name = record["out_name"]
        cls_name = rel_key.split("/")[0]
        out_path = os.path.join(processed_dir, split, cls_name, out_name)
        if not os.path.exists(out_path):
            return True, "file processed bị mất"

        # Kiểm tra nội dung file raw có thay đổi không
        current_md5 = _file_md5(raw_path)
        if record["md5"] != current_md5:
            return True, "nội dung thay đổi (MD5 khác)"

        return False, ""


def cleanup_orphans(
    manifest: ManifestManager,
    raw_dir: str,
    processed_dir: str,
    dry_run: bool = False,
) -> int:
    """
    Dọn dẹp (Cleanup): Xóa các file processed và bản ghi manifest
    của những ảnh raw đã bị xóa khỏi raw_dir.

    Ví dụ: Bạn xóa bus_000999.jpg khỏi raw/Bus/ →
           processed/train/Bus/bus_000999.jpg và bản ghi manifest sẽ bị xóa.

    Args:
        manifest    : ManifestManager instance
        raw_dir     : Thư mục dữ liệu thô
        processed_dir: Thư mục đầu ra
        dry_run     : True → chỉ in ra, không xóa thực sự

    Returns:
        Số file orphan đã xóa (hoặc sẽ xóa nếu dry_run)
    """
    orphan_keys = []
    for rel_key in manifest.all_keys():
        raw_path = os.path.join(raw_dir, rel_key.replace("/", os.sep))
        if not os.path.exists(raw_path):
            orphan_keys.append(rel_key)

    if not orphan_keys:
        print("  [CLEANUP] Không có file orphan nào. Sạch sẽ!")
        return 0

    prefix = "  [DRY-RUN]" if dry_run else "  [CLEANUP]"
    print(f"{prefix} Phát hiện {len(orphan_keys)} file orphan:")

    removed = 0
    for rel_key in orphan_keys:
        record = manifest.get_file_record(rel_key)
        if record:
            split    = record["split"]
            out_name = record["out_name"]
            cls_name = rel_key.split("/")[0]
            out_path = os.path.join(processed_dir, split, cls_name, out_name)
            print(f"    → {rel_key}  [{split}/{cls_name}/{out_name}]")
            if not dry_run:
                if os.path.exists(out_path):
                    os.remove(out_path)
                manifest.remove_key(rel_key)
                removed += 1
        else:
            print(f"    → {rel_key} (không có record processed)")
            if not dry_run:
                manifest.remove_key(rel_key)
                removed += 1

    if not dry_run:
        manifest.save()
        print(f"  [CLEANUP] Đã xóa {removed} file orphan và cập nhật manifest.")
    else:
        print(f"  [DRY-RUN] Sẽ xóa {len(orphan_keys)} file (chạy thật để áp dụng).")
        removed = len(orphan_keys)

    return removed


def purge_all(
    processed_dir: str,
    confirm: bool = False,
) -> None:
    """
    Xóa toàn bộ thư mục processed/ và manifest để reset sạch hoàn toàn.
    Sử dụng khi muốn đổi bộ lọc / tham số và tái xử lý từ đầu.

    Args:
        processed_dir: Thư mục cần xóa
        confirm      : Phải True mới thực hiện (tránh vô tình xóa)
    """
    if not confirm:
        raise RuntimeError(
            "purge_all() yêu cầu confirm=True để tránh xóa nhầm. "
            "Truyền confirm=True hoặc dùng --mode force từ CLI."
        )
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        print(f"  [PURGE] Đã xóa toàn bộ: {processed_dir}")
    else:
        print(f"  [PURGE] Thư mục không tồn tại (OK): {processed_dir}")

def custom_split_and_process(
    raw_dir: str,
    processed_dir: str,
    pipeline_name: str = "baseline_v1",
    target_size: int = TARGET_SIZE,
    seed: int = RANDOM_SEED,
    mode: str = "smart",
) -> dict:
    """
    Phân chia và xử lý toàn bộ tập dữ liệu theo yêu cầu của Giảng viên.

    Thuật toán phân rã:
      ┌─────────────────────────────────────────────────────────────┐
      │  Tổng dữ liệu (N ảnh/lớp)                                  │
      │  ├─ Test    (10%) ← tách biệt hoàn toàn                    │
      │  └─ Còn lại (90%)                                           │
      │      ├─ Valid Unseen (5/90 ≈ 5.56% phần còn lại)           │
      │      └─ Pool Train+ValidCopy (84.44% phần còn lại)         │
      │          ├─ Train          (80% tổng ≈ 94% pool)           │
      │          └─ Valid Copy     (5% tổng ≈ 6% pool) ← copy Train│
      └─────────────────────────────────────────────────────────────┘

    Cấu trúc thư mục đầu ra:
      processed_dir/
        ├── train/<class>/
        ├── valid/<class>/   ← Valid_Unseen (5%) + Valid_Copy (5%)
        ├── test/<class>/
        └── .pipeline_manifest.json   ← Manifest theo dõi trạng thái

    Chế độ chạy (`mode`):
      'smart'  (mặc định) — Ghi đè thông minh:
                 • Bỏ qua ảnh đã xử lý đúng cấu hình
                 • Tái xử lý ảnh thay đổi nội dung (MD5 khác)
                 • Dọn orphan (ảnh raw bị xóa)
                 • Nếu đổi filter/size/seed → tái xử lý toàn bộ
      'force'  — Xóa sạch processed/ và chạy lại từ đầu hoàn toàn
      'clean'  — Chỉ dọn orphan, không xử lý ảnh mới

    Args:
        raw_dir        : Thư mục chứa dữ liệu thô, mỗi lớp là 1 thư mục con
        processed_dir  : Thư mục đầu ra sau xử lý
        filter_name    : Bộ lọc áp dụng (tên trong FILTER_REGISTRY)
        target_size    : Kích thước ảnh đầu ra
        seed           : Random seed
        mode           : 'smart' | 'force' | 'clean'
        **filter_kwargs: Tham số cho bộ lọc

    Returns:
        dict: Thống kê số lượng ảnh mỗi split theo từng lớp
    """
    if mode not in ("smart", "force", "clean"):
        raise ValueError(f"mode phải là 'smart', 'force' hoặc 'clean'. Nhận: '{mode}'")

    # ── Chế độ FORCE: xóa sạch và bắt đầu lại ────────────────────────────
    if mode == "force":
        print("  [MODE=force] Đang xóa toàn bộ processed/ để chạy lại từ đầu...")
        purge_all(processed_dir, confirm=True)

    random.seed(seed)
    np.random.seed(seed)

    splits = ["train", "valid", "test"]

    # Lấy danh sách các lớp (thư mục con trong raw_dir)
    class_names = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    if not class_names:
        raise ValueError(f"Không tìm thấy lớp nào trong: {raw_dir}")

    print(f"\n{'='*60}")
    print(f"  Vehicle Type Recognition — Custom Data Pipeline")
    print(f"{'='*60}")
    print(f"  Chế độ          : {mode.upper()}")
    print(f"  Thư mục raw     : {raw_dir}")
    print(f"  Thư mục output  : {processed_dir}")
    print(f"  Pipeline        : {pipeline_name}")
    print(f"  Kích thước ảnh  : {target_size}×{target_size}")
    print(f"  Các lớp phát hiện: {class_names}")
    print(f"{'='*60}\n")

    # Tạo thư mục đầu ra
    for split in splits:
        for cls in class_names:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    # ── Khởi tạo Manifest ─────────────────────────────────────────────────
    manifest = ManifestManager(processed_dir)

    # Nếu cấu hình thay đổi trong chế độ smart → cảnh báo và force reprocess
    if mode == "smart" and manifest.get_meta() and manifest.config_changed(
        pipeline_name, target_size, seed
    ):
        old_meta = manifest.get_meta()
        print("  [WARN] Phát hiện thay đổi cấu hình pipeline so với lần chạy trước:")
        print(f"    Pipeline : {old_meta.get('pipeline')} → {pipeline_name}")
        print(f"    Size     : {old_meta.get('target_size')} → {target_size}")
        print(f"    Seed     : {old_meta.get('seed')} → {seed}")
        print("  → Tất cả ảnh sẽ được tái xử lý (toàn bộ manifest bị reset).")
        # Reset manifest để buộc xử lý lại tất cả
        manifest._data = {"_meta": {}, "files": {}}

    manifest.set_meta(pipeline_name, target_size, seed)

    # ── Chế độ CLEAN: chỉ dọn orphan, không xử lý ảnh mới ────────────────
    if mode == "clean":
        print("  [MODE=clean] Chỉ dọn dẹp file orphan (ảnh raw đã bị xóa)...")
        cleanup_orphans(manifest, raw_dir, processed_dir)
        manifest.save()
        return {}

    stats = {}
    total_skipped  = 0
    total_new      = 0
    total_updated  = 0

    # ── Dọn orphan trước khi xử lý (chỉ trong smart mode) ───────────────
    if mode == "smart":
        n_orphans = cleanup_orphans(manifest, raw_dir, processed_dir)
        if n_orphans > 0:
            manifest.save()  # Lưu ngay sau cleanup

    for cls in class_names:
        cls_raw_dir = os.path.join(raw_dir, cls)

        # Lấy tất cả file ảnh hợp lệ
        all_files = sorted([
            f for f in os.listdir(cls_raw_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])

        if not all_files:
            print(f"  [WARN] Lớp '{cls}' không có ảnh nào. Bỏ qua.")
            continue

        n = len(all_files)
        random.shuffle(all_files)

        # ── Bước 1: Tách TEST (10% tổng) ──────────────────────────
        n_test = max(1, round(n * TEST_RATIO))
        test_files    = all_files[:n_test]
        remaining     = all_files[n_test:]

        # ── Bước 2: Tách VALID UNSEEN (5% tổng ≈ từ phần còn lại) ─
        n_valid_unseen = max(1, round(n * VALID_UNSEEN_RATIO))
        valid_unseen_files = remaining[:n_valid_unseen]
        pool           = remaining[n_valid_unseen:]   # Pool cho Train + ValidCopy

        # ── Bước 3: Tách VALID COPY (5% tổng từ pool) ─────────────
        n_valid_copy = max(1, round(n * VALID_COPY_RATIO))
        # Chọn ngẫu nhiên từ pool (không xóa khỏi pool — copy từ Train)
        valid_copy_files = random.sample(pool, min(n_valid_copy, len(pool)))

        # ── Bước 4: Phần còn lại của pool → TRAIN ─────────────────
        train_files = pool  # 80% tổng (xấp xỉ)

        print(f"  [{cls.upper()}] | Tổng: {n} | "
              f"Train: {len(train_files)} | "
              f"Valid(Unseen): {len(valid_unseen_files)} | "
              f"Valid(Copy): {len(valid_copy_files)} | "
              f"Test: {len(test_files)}")

        # ── Xử lý và lưu ảnh (với Smart Overwrite) ────────────────
        cls_skipped = 0
        cls_new     = 0
        cls_updated = 0

        def process_and_save(file_list, split_name, suffix=""):
            nonlocal cls_skipped, cls_new, cls_updated
            out_dir = os.path.join(processed_dir, split_name, cls)
            for fname in tqdm(
                file_list,
                desc=f"    {split_name}/{cls}{suffix}",
                leave=False,
                ncols=80,
            ):
                src_path = os.path.join(cls_raw_dir, fname)
                base     = os.path.splitext(fname)[0]
                out_name = f"{base}{suffix}.jpg"
                rel_key  = f"{cls}/{fname}"

                # ── Smart Overwrite check ──────────────────────────
                if mode == "smart":
                    needs, reason = manifest.needs_processing(
                        rel_key, src_path, processed_dir
                    )
                    if not needs:
                        cls_skipped += 1
                        continue  # Bỏ qua — đã xử lý và không thay đổi
                    if reason in ("nội dung thay đổi (MD5 khác)", "file processed bị mất"):
                        cls_updated += 1
                    else:
                        cls_new += 1
                else:
                    cls_new += 1

                # ── Xử lý ảnh ─────────────────────────────────────
                try:
                    current_md5 = _file_md5(src_path)
                    img_rgb = preprocess_image(
                        src_path, pipeline_name, target_size
                    )
                    # Chuyển lại sang BGR để lưu bằng cv2.imwrite
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(out_dir, out_name), img_bgr)

                    # Cập nhật manifest
                    manifest.set_file_record(
                        rel_key, current_md5, split_name, out_name
                    )
                except Exception as e:
                    print(f"\n    [ERROR] {fname}: {e}")

        # Train
        process_and_save(train_files, "train")
        # Test
        process_and_save(test_files, "test")
        # Valid Unseen (ảnh mới chưa từng có trong Train/Test)
        process_and_save(valid_unseen_files, "valid", suffix="_unseen")
        # Valid Copy (copy từ tập Train — theo yêu cầu của GV)
        process_and_save(valid_copy_files, "valid", suffix="_copy")

        # Lưu manifest sau mỗi lớp (tránh mất dữ liệu nếu bị ngắt giữa chừng)
        manifest.save()

        total_skipped += cls_skipped
        total_new     += cls_new
        total_updated += cls_updated

        print(f"    → Mới: {cls_new} | Cập nhật: {cls_updated} | Bỏ qua: {cls_skipped}")

        stats[cls] = {
            "total":         n,
            "train":         len(train_files),
            "valid_unseen":  len(valid_unseen_files),
            "valid_copy":    len(valid_copy_files),
            "valid_total":   len(valid_unseen_files) + len(valid_copy_files),
            "test":          len(test_files),
            "new":           cls_new,
            "updated":       cls_updated,
            "skipped":       cls_skipped,
        }

    # ── In bảng tổng kết ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  KẾT QUẢ PHÂN CHIA DỮ LIỆU")
    print(f"{'='*65}")
    print(f"  {'Lớp':<15} {'Tổng':>6} {'Train':>7} {'Valid':>6} {'Test':>6} "
          f"{'Mới':>5} {'Cập nhật':>9} {'Bỏ qua':>7}")
    print(f"  {'-'*60}")
    for cls, s in stats.items():
        print(f"  {cls:<15} {s['total']:>6} {s['train']:>7} "
              f"{s['valid_total']:>6} {s['test']:>6} "
              f"{s.get('new',0):>5} {s.get('updated',0):>9} {s.get('skipped',0):>7}")
    print(f"  {'-'*60}")
    print(f"  Tổng cộng: Mới={total_new} | Cập nhật={total_updated} | Bỏ qua={total_skipped}")
    print(f"  Manifest : {os.path.join(processed_dir, MANIFEST_FILE)}")
    print(f"{'='*65}\n")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (chạy trực tiếp)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vehicle Type Recognition — Data Preprocessing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"g:\Data\Projects\VehicleTypeRecognition\data\raw",
        help="Thư mục dữ liệu thô (mỗi lớp là 1 thư mục con)",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=r"g:\Data\Projects\VehicleTypeRecognition\data\processed",
        help="Thư mục đầu ra sau xử lý",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="baseline_v1",
        choices=list(PIPELINE_CHAINS.keys()),
        help="Chuỗi tiền xử lý áp dụng (mặc định: baseline_v1)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=TARGET_SIZE,
        help=f"Kích thước ảnh đầu ra (mặc định: {TARGET_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (mặc định: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smart",
        choices=["smart", "force", "clean"],
        help=(
            "Chế độ chạy pipeline:\n"
            "  smart  (mặc định): Ghi đè thông minh — bỏ qua ảnh đã xử lý,\n"
            "                     tái xử lý nếu nội dung thay đổi hoặc đổi filter,\n"
            "                     tự động dọn orphan (ảnh raw bị xóa).\n"
            "  force            : Xóa sạch processed/ và chạy lại từ đầu hoàn toàn.\n"
            "  clean            : Chỉ dọn dẹp file orphan, không xử lý ảnh mới."
        ),
    )
    parser.add_argument(
        "--show_manifest",
        action="store_true",
        help="In nội dung manifest hiện tại rồi thoát (không chạy pipeline).",
    )

    args = parser.parse_args()

    # ── Xem manifest ──────────────────────────────────────────────────────
    if args.show_manifest:
        m = ManifestManager(args.processed_dir)
        meta = m.get_meta()
        files = m._data.get("files", {})
        print(f"\nManifest: {os.path.join(args.processed_dir, MANIFEST_FILE)}")
        print(f"  Tạo lúc    : {meta.get('created_at', 'N/A')}")
        print(f"  Cập nhật   : {meta.get('updated_at', 'N/A')}")
        print(f"  Pipeline   : {meta.get('pipeline', 'N/A')}")
        print(f"  Target size: {meta.get('target_size', 'N/A')}")
        print(f"  Seed       : {meta.get('seed', 'N/A')}")
        print(f"  Số ảnh theo dõi: {len(files)}")
        # Thống kê theo split
        from collections import Counter
        split_counter = Counter(v["split"] for v in files.values())
        for sp, cnt in sorted(split_counter.items()):
            print(f"    {sp}: {cnt} ảnh")
        import sys; sys.exit(0)

    # ── Chạy pipeline ─────────────────────────────────────────────────────
    custom_split_and_process(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        pipeline_name=args.pipeline,
        target_size=args.target_size,
        seed=args.seed,
        mode=args.mode,
    )
