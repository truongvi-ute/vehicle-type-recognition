"""
src/data_prep.py
================
Pipeline xử lý dữ liệu 3 giai đoạn cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Giai đoạn 1 — Base Processing & Split  (data/raw/       → data/splits/)
Giai đoạn 2 — Class Balancing           (data/splits/train/ → data/balanced/)
Giai đoạn 3 — Offline Augmentation     (data/balanced/  → data/augmented/)

Thư viện cần thiết:
    pip install opencv-python numpy tqdm

Cách chạy:
    # Chạy cả 3 giai đoạn liên tiếp
    python src/data_prep.py --all

    # Chạy từng giai đoạn riêng lẻ
    python src/data_prep.py --step 1
    python src/data_prep.py --step 2
    python src/data_prep.py --step 3

    # Tuỳ chỉnh đường dẫn và seed
    python src/data_prep.py --all --raw_dir data/raw --seed 42 --workers 4
"""

import os
import sys
import math
import shutil
import random
import argparse
import uuid
import logging
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS: set = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Tỷ lệ phân chia
TRAIN_RATIO: float = 0.80   # 80 % dữ liệu cho tập Train
VALID_RATIO: float = 0.05   # 5  % độc lập cho Valid-Unseen
TEST_RATIO:  float = 0.10   # 10 % hoàn toàn độc lập cho Test
# Valid-Copy = 5 % copy từ Train (được lấy sau khi chia Train)
VALID_COPY_RATIO: float = 0.05

RANDOM_SEED:   int = 42
NUM_WORKERS:   int = max(1, multiprocessing.cpu_count() - 1)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logger(name: str = "data_prep") -> logging.Logger:
    """Cấu hình logger ghi ra console với định dạng rõ ràng."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


log = _setup_logger()


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1 — CÁC HÀM OPENCV (KHÔNG ĐƯỢC CHỈNH SỬA NỘI DUNG)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pipeline_base(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa ảnh về 224×224 bằng Zero-Padding bảo toàn Aspect Ratio.
    Áp dụng Morphological Closing → Opening để xóa nhiễu hạt nhỏ trên nền.

    Args:
        img: Ảnh BGR đọc bằng cv2.imread (np.ndarray, shape HxWx3).

    Returns:
        Ảnh BGR đã chuẩn hóa, shape = (224, 224, 3), dtype = uint8.
    """
    h, w = img.shape[:2]
    scale = 224 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224 - new_h) // 2, (224 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel)


def apply_pipeline_night(img_base: np.ndarray) -> np.ndarray:
    """
    Giả lập ảnh chụp ban đêm / thiếu sáng:
      - Gamma correction làm tối ảnh (γ ∈ [0.4, 0.7])
      - Chèn nhiễu Gaussian mô phỏng ISO cao
      - Giảm bão hòa màu (Saturation) xuống 50 %

    Args:
        img_base: Ảnh BGR đã qua pipeline Base.

    Returns:
        Ảnh BGR giả lập cảnh tối, dtype = uint8.
    """
    gamma = random.uniform(0.4, 0.7)
    img_dark = np.clip(
        np.power(img_base / 255.0, gamma) * 255, 0, 255
    ).astype(np.uint8)

    noise = np.random.normal(0, 15, img_dark.shape)
    img_noisy = np.clip(img_dark + noise, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = (hsv[:, :, 1] * 0.5).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_pipeline_rain(img_base: np.ndarray) -> np.ndarray:
    """
    Mô phỏng hiệu ứng mưa trên ảnh:
      - Motion Blur theo góc ngẫu nhiên giả lập vệt mờ do mưa
      - Median Blur giả lập đốm nước đọng trên ống kính
      - Vẽ 100–300 nét thẳng màu xám trắng giả lập hạt mưa rơi

    Args:
        img_base: Ảnh BGR đã qua pipeline Base.

    Returns:
        Ảnh BGR giả lập cảnh mưa, dtype = uint8.
    """
    angle       = random.uniform(-10, 10)
    kernel_size = random.randint(7, 15)

    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, kernel_size // 2] = 1 / kernel_size
    M = cv2.getRotationMatrix2D(
        (kernel_size / 2, kernel_size / 2), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    img_rain = cv2.medianBlur(cv2.filter2D(img_base, -1, kernel), 3)

    for _ in range(random.randint(100, 300)):
        x      = random.randint(0, 223)
        y      = random.randint(0, 223)
        length = random.randint(3, 8)
        cv2.line(img_rain, (x, y), (x, y + length), (200, 200, 200), 1)

    return img_rain


def apply_pipeline_sun(img_base: np.ndarray) -> np.ndarray:
    """
    Tạo hiệu ứng nắng gắt và bóng đổ hình học:
      - Vẽ Sun Flare hình tròn đồng tâm màu trắng-vàng
      - Đổ bóng ngẫu nhiên hình tam giác che khuất
      - Tăng tương phản (alpha=1.2) và độ sáng (beta=10)

    Args:
        img_base: Ảnh BGR đã qua pipeline Base.

    Returns:
        Ảnh BGR giả lập cảnh nắng, dtype = uint8.
    """
    flare_x      = random.randint(50, 174)
    flare_y      = random.randint(50, 174)
    flare_radius = random.randint(30, 60)

    img_flare = img_base.copy()
    for r in range(flare_radius, 0, -5):
        intensity = int(255 * (r / flare_radius))
        cv2.circle(img_flare, (flare_x, flare_y), r,
                   (intensity, intensity, 255), -1)

    img_blend   = cv2.addWeighted(img_base, 0.7, img_flare, 0.3, 0)
    shadow_mask = np.ones_like(img_blend) * 255
    pts = np.array(
        [[random.randint(0, 224), random.randint(0, 224)] for _ in range(3)],
        np.int32,
    )
    cv2.fillPoly(shadow_mask, [pts], (50, 50, 50))

    return cv2.convertScaleAbs(
        cv2.addWeighted(img_blend, 0.8, shadow_mask, 0.2, 0),
        alpha=1.2, beta=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH DÙNG CHUNG
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(class_dir: str) -> List[str]:
    """
    Thu thập toàn bộ đường dẫn ảnh hợp lệ (theo IMG_EXTS) bên trong thư mục.
    Chỉ quét cấp thư mục hiện tại (không đệ quy).
    """
    result: List[str] = []
    if not os.path.isdir(class_dir):
        return result
    for fname in sorted(os.listdir(class_dir)):
        if Path(fname).suffix.lower() in IMG_EXTS:
            result.append(os.path.join(class_dir, fname))
    return result


def _discover_classes(root_dir: str) -> List[str]:
    """
    Lấy danh sách tên lớp phương tiện từ các thư mục con của `root_dir`.
    Chỉ lấy thư mục, bỏ qua file.
    """
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
        and not d.startswith(".")
    ])
    if not classes:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục lớp nào trong: {root_dir}"
        )
    return classes


def _makedirs(*paths: str) -> None:
    """Tạo nhiều thư mục (bao gồm cha) nếu chưa tồn tại."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _save_image(img: np.ndarray, out_path: str) -> bool:
    """
    Lưu ảnh BGR xuống đĩa.
    Trả về True nếu thành công, False nếu lỗi (in cảnh báo nhưng không dừng).
    """
    try:
        ok = cv2.imwrite(out_path, img)
        if not ok:
            log.warning(f"cv2.imwrite thất bại: {out_path}")
            return False
        return True
    except Exception as exc:
        log.warning(f"Lỗi khi lưu ảnh {out_path}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GIAI ĐOẠN 1 — BASE PROCESSING & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_image_split(args_tuple) -> Tuple[str, bool]:
    """
    Worker function dùng với multiprocessing.Pool.
    Đọc 1 ảnh, áp dụng pipeline Base, lưu vào thư mục đích.

    Args:
        args_tuple: (src_path, dst_path)

    Returns:
        (dst_path, success_flag)
    """
    src_path, dst_path = args_tuple
    try:
        img = cv2.imread(src_path)
        if img is None:
            return dst_path, False
        processed = apply_pipeline_base(img)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        ok = cv2.imwrite(dst_path, processed)
        return dst_path, ok
    except Exception:
        return dst_path, False


def process_and_split(
    raw_dir:    str,
    splits_dir: str,
    seed:       int = RANDOM_SEED,
    workers:    int = NUM_WORKERS,
) -> Dict[str, Dict[str, int]]:
    """
    Giai đoạn 1: Base Processing & Split.

    Đọc ảnh từng lớp trong `raw_dir`, áp dụng ngay apply_pipeline_base(),
    rồi phân chia và lưu vào `splits_dir` với cấu trúc:

        splits_dir/
          train/<class>/      — 80 % mỗi lớp
          valid/<class>/      — 5 % Unseen + 5 % Copy từ Train = ~10 % tổng
          test/<class>/       — 10 % mỗi lớp (hoàn toàn độc lập)

    Lưu ý:
        - Tập Test được tách ra *trước tiên* để đảm bảo hoàn toàn độc lập.
        - Valid-Unseen (5 %) không xuất hiện trong Train.
        - Valid-Copy (5 %) được copy ngẫu nhiên từ ảnh Train (có thể trùng
          nội dung nhưng khác split — mô phỏng yêu cầu đặc thù).
        - Ảnh lưu ra có tên gốc, không đổi tên để dễ theo dõi.

    Args:
        raw_dir    : Thư mục nguồn chứa các thư mục lớp (data/raw/).
        splits_dir : Thư mục đích (data/splits/).
        seed       : Random seed.
        workers    : Số process song song cho bước xử lý ảnh.

    Returns:
        stats — dict {class_name: {"train": N, "valid": N, "test": N}}
    """
    log.info("=" * 60)
    log.info("GIAI ĐOẠN 1 — BASE PROCESSING & SPLIT")
    log.info(f"  Nguồn : {raw_dir}")
    log.info(f"  Đích   : {splits_dir}")
    log.info(f"  Seed   : {seed}  |  Workers: {workers}")
    log.info("=" * 60)

    random.seed(seed)
    classes = _discover_classes(raw_dir)
    log.info(f"Phát hiện {len(classes)} lớp: {classes}")

    stats: Dict[str, Dict[str, int]] = {}
    t_start = time.time()

    for cls in classes:
        cls_raw_dir = os.path.join(raw_dir, cls)
        all_imgs    = _collect_images(cls_raw_dir)

        if not all_imgs:
            log.warning(f"  [{cls}] Không có ảnh — bỏ qua.")
            continue

        # ── Xáo trộn ngẫu nhiên ──────────────────────────────────────────
        random.shuffle(all_imgs)
        n_total = len(all_imgs)

        # ── Tách Test (10 %) trước — hoàn toàn độc lập ───────────────────
        n_test  = max(1, round(n_total * TEST_RATIO))
        test_imgs  = all_imgs[:n_test]
        remaining  = all_imgs[n_test:]          # 90 % còn lại

        # ── Tách Valid-Unseen (5 %) từ phần còn lại ──────────────────────
        n_unseen    = max(1, round(n_total * VALID_RATIO))
        unseen_imgs = remaining[:n_unseen]
        pool_imgs   = remaining[n_unseen:]      # 85 % (sẽ thành Train)

        # ── Tập Train = toàn bộ pool (≈ 80 %) ───────────────────────────
        train_imgs = pool_imgs

        # ── Valid-Copy = 5 % copy ngẫu nhiên từ Train ────────────────────
        n_copy    = max(1, round(n_total * VALID_COPY_RATIO))
        copy_imgs = random.sample(train_imgs, min(n_copy, len(train_imgs)))

        # ── Valid = Unseen + Copy ─────────────────────────────────────────
        valid_imgs = unseen_imgs + copy_imgs

        # ── Chuẩn bị thư mục đích ────────────────────────────────────────
        for split_name in ("train", "valid", "test"):
            _makedirs(os.path.join(splits_dir, split_name, cls))

        # ── Xây danh sách (src, dst) cho multiprocessing ─────────────────
        job_list: List[Tuple[str, str]] = []

        def _build_jobs(img_list: List[str], split_name: str) -> None:
            for src in img_list:
                fname = os.path.basename(src)
                dst   = os.path.join(splits_dir, split_name, cls, fname)
                job_list.append((src, dst))

        _build_jobs(train_imgs, "train")
        _build_jobs(valid_imgs, "valid")
        _build_jobs(test_imgs,  "test")

        # ── Xử lý song song ──────────────────────────────────────────────
        n_ok = 0
        if workers > 1:
            with multiprocessing.Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_process_one_image_split, job_list),
                    total=len(job_list),
                    desc=f"  [{cls}] Stage-1",
                    leave=False,
                    ncols=90,
                ))
            n_ok = sum(1 for _, ok in results if ok)
        else:
            for item in tqdm(job_list,
                             desc=f"  [{cls}] Stage-1",
                             leave=False, ncols=90):
                _, ok = _process_one_image_split(item)
                if ok:
                    n_ok += 1

        stats[cls] = {
            "train": len(train_imgs),
            "valid": len(valid_imgs),
            "test":  len(test_imgs),
            "ok":    n_ok,
            "total_jobs": len(job_list),
        }
        log.info(
            f"  [{cls:>12}] Train={len(train_imgs):>5} | "
            f"Valid={len(valid_imgs):>4} | Test={len(test_imgs):>4} | "
            f"Saved={n_ok}/{len(job_list)}"
        )

    elapsed = time.time() - t_start
    log.info(f"Giai đoạn 1 hoàn thành trong {elapsed:.1f}s\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# GIAI ĐOẠN 2 — CLASS BALANCING
# ─────────────────────────────────────────────────────────────────────────────

def balance_data(
    splits_dir:  str,
    balanced_dir: str,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """
    Giai đoạn 2: Class Balancing.

    Thuật toán K-formula:
        N_target = max(|class|) trong splits/train/
        K_class  = ceil(N_target / N_class)

    Với mỗi lớp:
        1. Nhân bản danh sách ảnh lên K lần.
        2. Lấy mẫu ngẫu nhiên (random.sample) đúng N_target phần tử.
        3. Copy vật lý sang balanced/<class>/ với tên mới (uuid suffix)
           để tránh xung đột khi một ảnh được chọn nhiều lần.

    Kết quả: mỗi lớp có đúng N_target ảnh trong balanced/<class>/.

    Args:
        splits_dir   : Thư mục chứa splits/train/ (data/splits/).
        balanced_dir : Thư mục đích (data/balanced/).
        seed         : Random seed.

    Returns:
        stats — dict {class_name: n_final_images}
    """
    log.info("=" * 60)
    log.info("GIAI ĐOẠN 2 — CLASS BALANCING")
    log.info(f"  Nguồn : {splits_dir}/train/")
    log.info(f"  Đích   : {balanced_dir}")
    log.info("=" * 60)

    random.seed(seed)

    train_dir = os.path.join(splits_dir, "train")
    classes   = _discover_classes(train_dir)

    # ── Bước 1: đếm số lượng mỗi lớp ────────────────────────────────────
    class_imgs: Dict[str, List[str]] = {}
    for cls in classes:
        imgs = _collect_images(os.path.join(train_dir, cls))
        class_imgs[cls] = imgs
        log.info(f"  [{cls:>12}] {len(imgs):>5} ảnh")

    n_target = max(len(v) for v in class_imgs.values())
    log.info(f"\n  N_target (lớp lớn nhất) = {n_target}")

    stats: Dict[str, int] = {}
    t_start = time.time()

    for cls, imgs in class_imgs.items():
        n_current = len(imgs)
        if n_current == 0:
            log.warning(f"  [{cls}] 0 ảnh — bỏ qua.")
            continue

        # ── Tính K ───────────────────────────────────────────────────────
        K = math.ceil(n_target / n_current)

        # ── Nhân bản danh sách ──────────────────────────────────────────
        expanded = imgs * K                         # lặp lại K lần
        sampled  = random.sample(expanded, n_target)  # lấy đúng N_target

        # ── Copy sang balanced/ ───────────────────────────────────────────
        out_cls_dir = os.path.join(balanced_dir, cls)
        _makedirs(out_cls_dir)

        # Đếm số lần mỗi ảnh được chọn để sinh suffix tăng dần
        occurrence: Dict[str, int] = {}

        for src in tqdm(sampled,
                        desc=f"  [{cls}] Stage-2",
                        leave=False, ncols=90):
            stem = Path(src).stem
            ext  = Path(src).suffix.lower() or ".jpg"

            occurrence[stem] = occurrence.get(stem, 0) + 1
            count            = occurrence[stem]

            # Nếu ảnh xuất hiện lần đầu → giữ tên gốc
            # Nếu xuất hiện lần 2+ → thêm uuid ngắn để tránh trùng
            if count == 1:
                out_name = f"{stem}{ext}"
            else:
                short_id = uuid.uuid4().hex[:8]
                out_name = f"{stem}_aug{short_id}{ext}"

            dst = os.path.join(out_cls_dir, out_name)
            shutil.copy2(src, dst)

        final_count = len(_collect_images(out_cls_dir))
        stats[cls]  = final_count
        log.info(
            f"  [{cls:>12}] K={K:>3} | "
            f"{n_current:>5} → {final_count:>5} ảnh"
        )

    elapsed = time.time() - t_start
    log.info(f"Giai đoạn 2 hoàn thành trong {elapsed:.1f}s\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# GIAI ĐOẠN 3 — OFFLINE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _augment_one_image(args_tuple) -> Tuple[str, int]:
    """
    Worker function cho multiprocessing — xử lý 1 ảnh balanced thành 4 file:
        *_base.jpg, *_night.jpg, *_rain.jpg, *_sun.jpg

    Args:
        args_tuple: (src_path, out_cls_dir, stem)

    Returns:
        (stem, n_saved)  — số file đã lưu thành công (0–4)
    """
    src_path, out_cls_dir, stem = args_tuple
    n_saved = 0
    try:
        img = cv2.imread(src_path)
        if img is None:
            return stem, 0

        # Ảnh đã qua Base ở Giai đoạn 1/2, nhưng ta vẫn bảo đảm đúng size
        pipelines = {
            "_base":  img,                      # bản sao nguyên trạng
            "_night": apply_pipeline_night(img),
            "_rain":  apply_pipeline_rain(img),
            "_sun":   apply_pipeline_sun(img),
        }

        for suffix, out_img in pipelines.items():
            dst = os.path.join(out_cls_dir, f"{stem}{suffix}.jpg")
            ok  = cv2.imwrite(dst, out_img)
            if ok:
                n_saved += 1

    except Exception:
        pass
    return stem, n_saved


def augment_data(
    balanced_dir:  str,
    splits_dir:    str,
    augmented_dir: str,
    workers:       int = NUM_WORKERS,
) -> Dict[str, int]:
    """
    Giai đoạn 3: Offline Augmentation.

    Tập Train:
        - Đọc ảnh từ `balanced_dir/<class>/`.
        - Mỗi ảnh sinh 4 file vật lý lưu vào augmented_dir/train/<class>/:
            <stem>_base.jpg   — bản sao nguyên trạng
            <stem>_night.jpg  — hiệu ứng ban đêm
            <stem>_rain.jpg   — hiệu ứng mưa
            <stem>_sun.jpg    — hiệu ứng nắng
        → Nếu balanced có 71,170 ảnh → augmented/train/ có 284,680 ảnh.

    Tập Valid & Test:
        - Copy nguyên cấu trúc từ `splits_dir/valid/` → `augmented_dir/valid/`
        - Copy nguyên cấu trúc từ `splits_dir/test/`  → `augmented_dir/test/`
        - Không augment thêm (tập đánh giá phải giữ nguyên bản để đánh giá
          khách quan — không làm tăng phân phối giả tạo).

    Args:
        balanced_dir  : Thư mục chứa ảnh đã cân bằng (data/balanced/).
        splits_dir    : Thư mục splits/ để lấy valid/ và test/.
        augmented_dir : Thư mục đích (data/augmented/).
        workers       : Số process song song.

    Returns:
        stats — dict {class_name: n_augmented_train_images}
    """
    log.info("=" * 60)
    log.info("GIAI ĐOẠN 3 — OFFLINE AUGMENTATION")
    log.info(f"  Nguồn Train : {balanced_dir}")
    log.info(f"  Nguồn Valid/Test: {splits_dir}")
    log.info(f"  Đích         : {augmented_dir}")
    log.info(f"  Workers      : {workers}")
    log.info("=" * 60)

    t_start = time.time()
    stats: Dict[str, int] = {}

    # ── 3A. Tập Train — augment x4 ───────────────────────────────────────
    log.info("\n[3A] Augment tập TRAIN ...")
    classes = _discover_classes(balanced_dir)

    for cls in classes:
        src_cls_dir = os.path.join(balanced_dir, cls)
        out_cls_dir = os.path.join(augmented_dir, "train", cls)
        _makedirs(out_cls_dir)

        imgs = _collect_images(src_cls_dir)
        if not imgs:
            log.warning(f"  [{cls}] 0 ảnh trong balanced — bỏ qua.")
            continue

        # Xây job list: mỗi ảnh → 1 tuple
        jobs = [
            (src, out_cls_dir, Path(src).stem)
            for src in imgs
        ]

        n_saved_total = 0
        if workers > 1:
            with multiprocessing.Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_augment_one_image, jobs),
                    total=len(jobs),
                    desc=f"  [{cls}] Stage-3",
                    leave=False,
                    ncols=90,
                ))
            n_saved_total = sum(n for _, n in results)
        else:
            for job in tqdm(jobs,
                            desc=f"  [{cls}] Stage-3",
                            leave=False, ncols=90):
                _, n = _augment_one_image(job)
                n_saved_total += n

        n_final = len(_collect_images(out_cls_dir))
        stats[cls] = n_final
        log.info(
            f"  [{cls:>12}] {len(imgs):>5} ảnh balanced × 4 → "
            f"{n_final:>6} ảnh (saved_ops={n_saved_total})"
        )

    # ── 3B. Tập Valid — copy nguyên si ──────────────────────────────────
    log.info("\n[3B] Copy tập VALID (không augment) ...")
    valid_src = os.path.join(splits_dir, "valid")
    valid_dst = os.path.join(augmented_dir, "valid")

    if os.path.isdir(valid_src):
        if os.path.exists(valid_dst):
            shutil.rmtree(valid_dst)
        shutil.copytree(valid_src, valid_dst)
        n_valid = sum(
            len(_collect_images(os.path.join(valid_dst, c)))
            for c in os.listdir(valid_dst)
            if os.path.isdir(os.path.join(valid_dst, c))
        )
        log.info(f"  Đã copy {n_valid} ảnh Valid → {valid_dst}")
    else:
        log.warning(f"  Không tìm thấy {valid_src} — bỏ qua Valid.")

    # ── 3C. Tập Test — copy nguyên si ───────────────────────────────────
    log.info("\n[3C] Copy tập TEST (không augment) ...")
    test_src = os.path.join(splits_dir, "test")
    test_dst = os.path.join(augmented_dir, "test")

    if os.path.isdir(test_src):
        if os.path.exists(test_dst):
            shutil.rmtree(test_dst)
        shutil.copytree(test_src, test_dst)
        n_test = sum(
            len(_collect_images(os.path.join(test_dst, c)))
            for c in os.listdir(test_dst)
            if os.path.isdir(os.path.join(test_dst, c))
        )
        log.info(f"  Đã copy {n_test} ảnh Test → {test_dst}")
    else:
        log.warning(f"  Không tìm thấy {test_src} — bỏ qua Test.")

    elapsed = time.time() - t_start
    log.info(f"\nGiai đoạn 3 hoàn thành trong {elapsed:.1f}s\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ARGPARSE + ĐIỀU PHỐI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Điểm vào chính. Dùng argparse để chọn chạy từng giai đoạn hoặc tất cả.

    Ví dụ:
        python src/data_prep.py --all
        python src/data_prep.py --step 1
        python src/data_prep.py --step 2 --balanced_dir data/balanced_custom
        python src/data_prep.py --all --workers 8 --seed 0
    """

    # ── Sửa encoding stdout trên Windows ────────────────────────────────
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Vehicle Type Recognition — Data Preparation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Đường dẫn ────────────────────────────────────────────────────────
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw",
        help="Thư mục chứa dữ liệu ảnh gốc (mỗi lớp = 1 thư mục con).",
    )
    parser.add_argument(
        "--splits_dir", type=str, default="data/splits",
        help="Thư mục đầu ra của Giai đoạn 1.",
    )
    parser.add_argument(
        "--balanced_dir", type=str, default="data/balanced",
        help="Thư mục đầu ra của Giai đoạn 2.",
    )
    parser.add_argument(
        "--augmented_dir", type=str, default="data/augmented",
        help="Thư mục đầu ra của Giai đoạn 3.",
    )

    # ── Điều phối giai đoạn ──────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Chạy cả 3 giai đoạn liên tiếp (1 → 2 → 3).",
    )
    group.add_argument(
        "--step", type=int, choices=[1, 2, 3],
        help="Chạy riêng một giai đoạn (1, 2 hoặc 3).",
    )

    # ── Tham số chung ───────────────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed để đảm bảo tái tạo kết quả.",
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help="Số tiến trình song song (multiprocessing). "
             "Đặt 1 để debug trên Windows.",
    )

    args = parser.parse_args()

    # ── Kiểm tra thư mục nguồn ──────────────────────────────────────────
    def _check_dir(path: str, label: str) -> None:
        if not os.path.isdir(path):
            log.error(f"Thư mục {label} không tồn tại: {path}")
            sys.exit(1)

    # ── Điều phối ────────────────────────────────────────────────────────
    t0 = time.time()

    if args.all or args.step == 1:
        _check_dir(args.raw_dir, "--raw_dir")
        process_and_split(
            raw_dir    = args.raw_dir,
            splits_dir = args.splits_dir,
            seed       = args.seed,
            workers    = args.workers,
        )

    if args.all or args.step == 2:
        _check_dir(args.splits_dir, "--splits_dir")
        balance_data(
            splits_dir   = args.splits_dir,
            balanced_dir = args.balanced_dir,
            seed         = args.seed,
        )

    if args.all or args.step == 3:
        _check_dir(args.balanced_dir, "--balanced_dir")
        _check_dir(args.splits_dir,   "--splits_dir")
        augment_data(
            balanced_dir  = args.balanced_dir,
            splits_dir    = args.splits_dir,
            augmented_dir = args.augmented_dir,
            workers       = args.workers,
        )

    total = time.time() - t0
    log.info(f"✅ Pipeline hoàn thành. Tổng thời gian: {total:.1f}s")

    # ── In tóm tắt cấu trúc đầu ra ──────────────────────────────────────
    log.info("\nCấu trúc đầu ra kỳ vọng:")
    log.info("  data/splits/   → train/ | valid/ | test/")
    log.info("  data/balanced/ → <class>/ (cân bằng N_target mỗi lớp)")
    log.info("  data/augmented/→ train/ (x4 pipelines) | valid/ | test/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bắt buộc trên Windows khi dùng multiprocessing với spawn method
    multiprocessing.freeze_support()
    main()
