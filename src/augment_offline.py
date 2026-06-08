"""
Quota-based offline augmentation for VehicleTypeRecognition.

Output:
    data/augmented/train/<class>/<source>_<bucket>_<orig|geo>_<index>.jpg

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
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .data_split import IMG_EXTS, collect_images, discover_classes
    from .image_pipelines import apply_base_pipeline, apply_environment_pipeline
except ImportError:
    from data_split import IMG_EXTS, collect_images, discover_classes
    from image_pipelines import apply_base_pipeline, apply_environment_pipeline

TARGET_PER_CLASS: int | None = None
BUCKET_RATIOS = {
    "normal": 0.70,
    "rain": 0.10,
    "sun": 0.10,
    "night": 0.10,
}
BUCKETS = ["normal", "rain", "sun", "night"]
WEATHER_BUCKETS = ["rain", "sun", "night"]


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def cv2_bgr_to_pil_rgb(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_rgb_to_cv2_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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


def resolve_target_per_class(class_images: Dict[str, List[Path]], target_per_class: int | None) -> int:
    if target_per_class is not None:
        if target_per_class <= 0:
            raise ValueError("target_per_class must be positive when provided.")
        return target_per_class

    max_count = max((len(images) for images in class_images.values()), default=0)
    if max_count <= 0:
        raise ValueError("Cannot infer target_per_class because no input images were found.")
    return max_count


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


def repeated_geo_sources(images: List[Path], count: int, rng: random.Random) -> List[Tuple[Path, bool]]:
    return [(rng.choice(images), True) for _ in range(max(0, count))]


def build_weather_plan(
    images: List[Path],
    available_originals: List[Path],
    targets: Dict[str, int],
    rng: random.Random,
) -> Dict[str, List[Tuple[Path, bool]]]:
    weather_plan: Dict[str, List[Tuple[Path, bool]]] = {bucket: [] for bucket in WEATHER_BUCKETS}
    slots: List[str] = []
    for bucket in WEATHER_BUCKETS:
        slots.extend([bucket] * targets[bucket])
    rng.shuffle(slots)

    originals = available_originals[:]
    rng.shuffle(originals)
    for bucket, src in zip(slots, originals):
        weather_plan[bucket].append((src, False))

    for bucket in slots[len(originals):]:
        weather_plan[bucket].append((rng.choice(images), True))

    for bucket in WEATHER_BUCKETS:
        rng.shuffle(weather_plan[bucket])
    return weather_plan


def build_class_plan(
    images: List[Path],
    targets: Dict[str, int],
    rng: random.Random,
) -> Dict[str, List[Tuple[Path, bool]]]:
    """
    Allocate a class by policy:
    - cap classes at or above target;
    - fill normal to 70% first;
    - use originals above the normal quota for weather before generating geo samples.
    """
    plan: Dict[str, List[Tuple[Path, bool]]] = {bucket: [] for bucket in BUCKETS}
    if not images:
        return plan

    target_total = sum(targets[bucket] for bucket in BUCKETS)
    normal_target = targets["normal"]
    shuffled = images[:]
    rng.shuffle(shuffled)

    if len(shuffled) >= target_total:
        capped = shuffled[:target_total]
        plan["normal"] = [(src, False) for src in capped[:normal_target]]
        weather_originals = capped[normal_target:]
    elif len(shuffled) >= normal_target:
        plan["normal"] = [(src, False) for src in shuffled[:normal_target]]
        weather_originals = shuffled[normal_target:]
    else:
        plan["normal"] = [(src, False) for src in shuffled]
        plan["normal"].extend(repeated_geo_sources(images, normal_target - len(shuffled), rng))
        weather_originals = []

    weather_plan = build_weather_plan(images, weather_originals, targets, rng)
    for bucket in WEATHER_BUCKETS:
        plan[bucket] = weather_plan[bucket]
    return plan


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

    out_dir = output_root / "train" / cls
    clean_dir(out_dir)
    plan = build_class_plan(images, targets, rng)

    for bucket in BUCKETS:
        selected = plan[bucket]

        saved = 0
        for index, (src, use_geo) in enumerate(tqdm(selected, desc=f"{cls}/{bucket}", leave=False)):
            img_bgr = cv2.imread(str(src))
            if img_bgr is None:
                continue

            base = apply_base_pipeline(cv2_bgr_to_pil_rgb(img_bgr))
            if use_geo:
                base_bgr = pil_rgb_to_cv2_bgr(base)
                geo_bgr = apply_geometric_augmentation(base_bgr, rng)
                base = apply_base_pipeline(cv2_bgr_to_pil_rgb(geo_bgr))

            out, _ = apply_environment_pipeline(
                base,
                bucket,
                seed=rng.randint(0, 2**32 - 1),
            )
            out_bgr = pil_rgb_to_cv2_bgr(out)
            dst = out_dir / make_output_name(src, bucket, index, use_geo)
            if cv2.imwrite(str(dst), out_bgr):
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
    target_per_class: int | None = TARGET_PER_CLASS,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    train_output = output_dir / "train"
    if train_output.exists():
        shutil.rmtree(train_output)
    train_output.mkdir(parents=True, exist_ok=True)

    class_images = {
        cls: collect_images(input_dir / cls)
        for cls in discover_classes(input_dir)
    }
    resolved_target = resolve_target_per_class(class_images, target_per_class)
    targets = bucket_targets(resolved_target)
    stats: Dict[str, Dict[str, int]] = {}

    for cls, images in class_images.items():
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
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=TARGET_PER_CLASS,
        help="Images per class after augmentation. Defaults to the largest input class size.",
    )
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
