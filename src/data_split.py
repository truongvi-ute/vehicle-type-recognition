"""
Split Vehicle-10 data with the current project policy.

Outputs:
    data/splits/train/<class>/
    data/splits/valid_unseen/<class>/
    data/splits/valid_traincopy/<class>/
    data/splits/test/<class>/

Policy:
    - train: 85% independent source for training
    - valid_unseen: 5% independent validation
    - test: 10% independent final evaluation
    - valid_traincopy: auxiliary copy from train, about 5% of original class size

Base Pipeline:
    - resize while preserving aspect ratio
    - zero-pad to 224x224
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
TRAIN_RATIO = 0.85
VALID_UNSEEN_RATIO = 0.05
TEST_RATIO = 0.10
VALID_TRAINCOPY_RATIO = 0.05
IMAGE_SIZE = 224


def collect_images(class_dir: Path) -> List[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def discover_classes(raw_dir: Path) -> List[str]:
    classes = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not classes:
        raise FileNotFoundError(f"No class folders found in {raw_dir}")
    return classes


def apply_pipeline_base(img: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize with preserved aspect ratio and zero-pad to image_size x image_size."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image shape")

    scale = image_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def safe_output_name(src: Path, fallback_ext: str = ".jpg") -> str:
    ext = src.suffix.lower() if src.suffix.lower() in IMG_EXTS else fallback_ext
    return f"{src.stem}{ext}"


def write_processed(src: Path, dst: Path, image_size: int = IMAGE_SIZE) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    processed = apply_pipeline_base(img, image_size=image_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), processed))


def copy_train_file(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def allocate_counts(n_total: int) -> Dict[str, int]:
    n_test = int(round(n_total * TEST_RATIO))
    n_valid_unseen = int(round(n_total * VALID_UNSEEN_RATIO))

    if n_total >= 3:
        n_test = max(1, n_test)
        n_valid_unseen = max(1, n_valid_unseen)

    if n_test + n_valid_unseen > n_total:
        n_test = min(n_test, n_total)
        n_valid_unseen = max(0, n_total - n_test)

    n_train = n_total - n_test - n_valid_unseen
    n_valid_traincopy = min(n_train, int(round(n_total * VALID_TRAINCOPY_RATIO)))
    if n_train > 0 and n_total >= 3:
        n_valid_traincopy = max(1, n_valid_traincopy)

    return {
        "train": n_train,
        "valid_unseen": n_valid_unseen,
        "valid_traincopy": n_valid_traincopy,
        "test": n_test,
    }


def clean_output_dirs(output_dir: Path, splits: Iterable[str]) -> None:
    for split in splits:
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def split_dataset(
    raw_dir: Path,
    output_dir: Path,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    split_names = ["train", "valid_unseen", "valid_traincopy", "test"]
    clean_output_dirs(output_dir, split_names)

    stats: Dict[str, Dict[str, int]] = {}
    classes = discover_classes(raw_dir)

    for cls in classes:
        images = collect_images(raw_dir / cls)
        rng.shuffle(images)
        counts = allocate_counts(len(images))

        test_imgs = images[:counts["test"]]
        valid_unseen_start = counts["test"]
        valid_unseen_end = valid_unseen_start + counts["valid_unseen"]
        valid_unseen_imgs = images[valid_unseen_start:valid_unseen_end]
        train_imgs = images[valid_unseen_end:]

        valid_traincopy_imgs: Sequence[Path]
        if counts["valid_traincopy"] > 0:
            valid_traincopy_imgs = rng.sample(
                train_imgs,
                k=min(counts["valid_traincopy"], len(train_imgs)),
            )
        else:
            valid_traincopy_imgs = []

        class_stats = {split: 0 for split in split_names}

        for split, split_imgs in [
            ("train", train_imgs),
            ("valid_unseen", valid_unseen_imgs),
            ("test", test_imgs),
        ]:
            for src in tqdm(split_imgs, desc=f"{cls}/{split}", leave=False):
                dst = output_dir / split / cls / safe_output_name(src)
                if write_processed(src, dst, image_size=image_size):
                    class_stats[split] += 1

        for src in tqdm(valid_traincopy_imgs, desc=f"{cls}/valid_traincopy", leave=False):
            train_src = output_dir / "train" / cls / safe_output_name(src)
            dst = output_dir / "valid_traincopy" / cls / safe_output_name(src)
            if train_src.exists() and copy_train_file(train_src, dst):
                class_stats["valid_traincopy"] += 1

        stats[cls] = class_stats

    return stats


def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    header = f"{'class':<14} {'train':>8} {'valid_unseen':>14} {'valid_traincopy':>16} {'test':>8}"
    print(header)
    print("-" * len(header))
    totals = {"train": 0, "valid_unseen": 0, "valid_traincopy": 0, "test": 0}
    for cls, row in sorted(stats.items()):
        for key in totals:
            totals[key] += row.get(key, 0)
        print(
            f"{cls:<14} {row.get('train', 0):>8} "
            f"{row.get('valid_unseen', 0):>14} "
            f"{row.get('valid_traincopy', 0):>16} "
            f"{row.get('test', 0):>8}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<14} {totals['train']:>8} "
        f"{totals['valid_unseen']:>14} "
        f"{totals['valid_traincopy']:>16} "
        f"{totals['test']:>8}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Vehicle-10 dataset")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--stats_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
    )
    print_stats(stats)
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
