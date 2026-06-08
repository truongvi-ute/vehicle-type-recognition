"""
Quota-based offline augmentation for VehicleTypeRecognition.

Output:
    data/augmented/train/<class>/normal/
    data/augmented/train/<class>/rain/
    data/augmented/train/<class>/sun/
    data/augmented/train/<class>/night/

Validation and test folders are copied without augmentation:
    data/augmented/valid_unseen/<class>/
    data/augmented/valid_traincopy/<class>/
    data/augmented/test/<class>/
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .data_split import IMG_EXTS, IMAGE_SIZE, apply_pipeline_base, collect_images, discover_classes
except ImportError:
    from data_split import IMG_EXTS, IMAGE_SIZE, apply_pipeline_base, collect_images, discover_classes

TARGET_PER_CLASS = 7000
BUCKET_RATIOS = {
    "normal": 0.70,
    "rain": 0.10,
    "sun": 0.10,
    "night": 0.10,
}
BUCKETS = ["normal", "rain", "sun", "night"]


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def bucket_targets(target_per_class: int) -> Dict[str, int]:
    normal = int(round(target_per_class * BUCKET_RATIOS["normal"]))
    rain = int(round(target_per_class * BUCKET_RATIOS["rain"]))
    sun = int(round(target_per_class * BUCKET_RATIOS["sun"]))
    night = target_per_class - normal - rain - sun
    return {
        "normal": normal,
        "rain": rain,
        "sun": sun,
        "night": night,
    }


def apply_geometric_augmentation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.06, 0.06) * w
    ty = rng.uniform(-0.06, 0.06) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += (tx, ty)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if rng.random() < 0.35:
        margin = 0.06
        src = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        dst = np.float32([
            [rng.uniform(0, margin) * w, rng.uniform(0, margin) * h],
            [w - 1 - rng.uniform(0, margin) * w, rng.uniform(0, margin) * h],
            [w - 1 - rng.uniform(0, margin) * w, h - 1 - rng.uniform(0, margin) * h],
            [rng.uniform(0, margin) * w, h - 1 - rng.uniform(0, margin) * h],
        ])
        persp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, persp, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    alpha = rng.uniform(0.85, 1.2)
    beta = rng.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        drop_w = rng.randint(max(8, w // 14), max(12, w // 5))
        drop_h = rng.randint(max(8, h // 14), max(12, h // 5))
        x = rng.randint(0, max(0, w - drop_w))
        y = rng.randint(0, max(0, h - drop_h))
        out[y:y + drop_h, x:x + drop_w] = 0

    return out


def apply_rain(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    for _ in range(rng.randint(80, 180)):
        x = rng.randint(0, IMAGE_SIZE - 1)
        y = rng.randint(0, IMAGE_SIZE - 1)
        length = rng.randint(4, 10)
        cv2.line(out, (x, y), (min(IMAGE_SIZE - 1, x + 1), min(IMAGE_SIZE - 1, y + length)), (200, 200, 200), 1)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return cv2.convertScaleAbs(out, alpha=0.9, beta=-8)


def apply_sun(img: np.ndarray, rng: random.Random) -> np.ndarray:
    flare = img.copy()
    center = (rng.randint(35, 189), rng.randint(35, 189))
    radius = rng.randint(25, 60)
    cv2.circle(flare, center, radius, (255, 245, 210), -1)
    out = cv2.addWeighted(img, 0.76, flare, 0.24, 0)
    return cv2.convertScaleAbs(out, alpha=rng.uniform(1.05, 1.2), beta=rng.uniform(5, 18))


def apply_night(img: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(1.4, 2.0)
    norm = img / 255.0
    dark = np.power(norm, gamma)
    out = np.clip(dark * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.65, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    noise = rng.normalvariate(0, 1)
    if noise:
        gaussian = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 6, out.shape)
        out = np.clip(out + gaussian, 0, 255).astype(np.uint8)
    return out


def apply_environment(img: np.ndarray, bucket: str, rng: random.Random) -> np.ndarray:
    if bucket == "normal":
        return img
    if bucket == "rain":
        return apply_rain(img, rng)
    if bucket == "sun":
        return apply_sun(img, rng)
    if bucket == "night":
        return apply_night(img, rng)
    raise ValueError(f"Unknown bucket: {bucket}")


def choose_sources(images: List[Path], count: int, rng: random.Random) -> List[Tuple[Path, bool]]:
    if count <= 0:
        return []
    if len(images) >= count:
        return [(p, False) for p in rng.sample(images, count)]

    selected: List[Tuple[Path, bool]] = [(p, False) for p in images]
    while len(selected) < count:
        selected.append((rng.choice(images), True))
    rng.shuffle(selected)
    return selected


def make_output_name(src: Path, bucket: str, index: int, geometric: bool) -> str:
    marker = "geo" if geometric else "orig"
    return f"{src.stem}_{bucket}_{marker}_{index:05d}.jpg"


def augment_class(
    cls: str,
    images: List[Path],
    output_root: Path,
    targets: Dict[str, int],
    rng: random.Random,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    if not images:
        return {bucket: 0 for bucket in BUCKETS}

    for bucket in BUCKETS:
        out_dir = output_root / "train" / cls / bucket
        clean_dir(out_dir)
        selected = choose_sources(images, targets[bucket], rng)

        saved = 0
        for index, (src, use_geo) in enumerate(tqdm(selected, desc=f"{cls}/{bucket}", leave=False)):
            img = cv2.imread(str(src))
            if img is None:
                continue
            base = apply_pipeline_base(img)
            if use_geo:
                base = apply_geometric_augmentation(base, rng)
                base = apply_pipeline_base(base)
            out = apply_environment(base, bucket, rng)
            dst = out_dir / make_output_name(src, bucket, index, use_geo)
            if cv2.imwrite(str(dst), out):
                saved += 1
        stats[bucket] = saved

    return stats


def copy_split(splits_dir: Path, augmented_dir: Path, split: str) -> int:
    src = splits_dir / split
    dst = augmented_dir / split
    if not src.exists():
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    splits_dir: Path | None = None,
    target_per_class: int = TARGET_PER_CLASS,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    train_output = output_dir / "train"
    if train_output.exists():
        shutil.rmtree(train_output)
    train_output.mkdir(parents=True, exist_ok=True)

    targets = bucket_targets(target_per_class)
    stats: Dict[str, Dict[str, int]] = {}

    for cls in discover_classes(input_dir):
        images = collect_images(input_dir / cls)
        stats[cls] = augment_class(cls, images, output_dir, targets, rng)

    if splits_dir:
        for split in ["valid_unseen", "valid_traincopy", "test"]:
            copied = copy_split(splits_dir, output_dir, split)
            stats[f"__copied_{split}"] = {"total": copied}

    return stats


def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    header = f"{'class':<20} {'normal':>8} {'rain':>8} {'sun':>8} {'night':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    totals = defaultdict(int)
    for cls, row in sorted(stats.items()):
        if cls.startswith("__copied_"):
            print(f"{cls:<20} {'':>8} {'':>8} {'':>8} {'':>8} {row.get('total', 0):>8}")
            continue
        total = sum(row.get(bucket, 0) for bucket in BUCKETS)
        for bucket in BUCKETS:
            totals[bucket] += row.get(bucket, 0)
        totals["total"] += total
        print(
            f"{cls:<20} {row.get('normal', 0):>8} {row.get('rain', 0):>8} "
            f"{row.get('sun', 0):>8} {row.get('night', 0):>8} {total:>8}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<20} {totals['normal']:>8} {totals['rain']:>8} "
        f"{totals['sun']:>8} {totals['night']:>8} {totals['total']:>8}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quota-based offline augmentation")
    parser.add_argument("--input_dir", type=Path, default=Path("data/splits/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/augmented"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--target_per_class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )
    print_stats(stats)
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
