"""Quota-based offline augmentation for Geo Version 2.

This script reads the existing split dataset and creates a separate V2 dataset:

    data/augmented_v2/train/<class>/<source>_<bucket>_<orig|geo>_<index>.jpg

The split, source-selection policy, geometric augmentation, quota calculation,
and random seed behavior are inherited from Version 1. Only the final 30%
pipelines are mapped as follows:

    rain  -> gaussian_blur
    sun   -> motion_blur
    night -> unsharp_mask

Validation and test folders are copied without augmentation.
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
from tqdm import tqdm

try:
    from .augment_offline import (
        apply_geometric_augmentation,
        build_class_plan as build_v1_class_plan,
        bucket_targets as v1_bucket_targets,
        cv2_bgr_to_pil_rgb,
        pil_rgb_to_cv2_bgr,
        resolve_target_per_class,
    )
    from .data_split import IMG_EXTS, collect_images, discover_classes
    from .image_pipelines import apply_base_pipeline, apply_v2_pipeline
except ImportError:
    from augment_offline import (
        apply_geometric_augmentation,
        build_class_plan as build_v1_class_plan,
        bucket_targets as v1_bucket_targets,
        cv2_bgr_to_pil_rgb,
        pil_rgb_to_cv2_bgr,
        resolve_target_per_class,
    )
    from data_split import IMG_EXTS, collect_images, discover_classes
    from image_pipelines import apply_base_pipeline, apply_v2_pipeline


TARGET_PER_CLASS: int | None = None
BUCKET_RATIOS = {
    "normal": 0.70,
    "gaussian_blur": 0.10,
    "motion_blur": 0.10,
    "unsharp_mask": 0.10,
}
BUCKETS = ["normal", "gaussian_blur", "motion_blur", "unsharp_mask"]
V1_TO_V2_BUCKET = {
    "normal": "normal",
    "rain": "gaussian_blur",
    "sun": "motion_blur",
    "night": "unsharp_mask",
}
EVAL_SPLITS = ["valid_unseen", "valid_traincopy", "test"]


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def bucket_targets(target_per_class: int) -> Dict[str, int]:
    """Return V2 quotas with exactly the same rounding policy as Version 1."""
    v1_targets = v1_bucket_targets(target_per_class)
    return {
        V1_TO_V2_BUCKET[v1_bucket]: count
        for v1_bucket, count in v1_targets.items()
    }


def build_class_plan(
    images: List[Path],
    targets: Dict[str, int],
    rng: random.Random,
) -> Dict[str, List[Tuple[Path, bool]]]:
    """Build a V2 plan by remapping the exact Version 1 allocation plan."""
    v1_targets = {
        v1_bucket: targets[v2_bucket]
        for v1_bucket, v2_bucket in V1_TO_V2_BUCKET.items()
    }
    v1_plan = build_v1_class_plan(images, v1_targets, rng)
    return {
        V1_TO_V2_BUCKET[v1_bucket]: selected
        for v1_bucket, selected in v1_plan.items()
    }


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
    if not images:
        return {bucket: 0 for bucket in BUCKETS}

    out_dir = output_root / "train" / cls
    clean_dir(out_dir)
    plan = build_class_plan(images, targets, rng)
    stats: Dict[str, int] = {}

    for bucket in BUCKETS:
        saved = 0
        selected = plan[bucket]

        for index, (src, use_geo) in enumerate(
            tqdm(selected, desc=f"{cls}/{bucket}", leave=False)
        ):
            img_bgr = cv2.imread(str(src))
            if img_bgr is None:
                continue

            base = apply_base_pipeline(cv2_bgr_to_pil_rgb(img_bgr))
            if use_geo:
                base_bgr = pil_rgb_to_cv2_bgr(base)
                geo_bgr = apply_geometric_augmentation(base_bgr, rng)
                base = apply_base_pipeline(cv2_bgr_to_pil_rgb(geo_bgr))

            output, _ = apply_v2_pipeline(
                base,
                bucket,
                seed=rng.randint(0, 2**32 - 1),
            )
            output_bgr = pil_rgb_to_cv2_bgr(output)
            destination = out_dir / make_output_name(src, bucket, index, use_geo)
            if cv2.imwrite(str(destination), output_bgr):
                saved += 1

        stats[bucket] = saved

    return stats


def copy_split(splits_dir: Path, output_dir: Path, split: str) -> int:
    source = splits_dir / split
    destination = output_dir / split
    if not source.exists():
        return 0
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return sum(
        1
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_paths(input_dir: Path, output_dir: Path, splits_dir: Path | None) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Training source directory not found: {input_dir}")

    input_path = _resolved(input_dir)
    output_path = _resolved(output_dir)
    if (
        output_path == input_path
        or output_path in input_path.parents
        or input_path in output_path.parents
    ):
        raise ValueError("output_dir must not overlap input_dir.")

    if splits_dir is not None:
        if not splits_dir.is_dir():
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
        splits_path = _resolved(splits_dir)
        if (
            output_path == splits_path
            or output_path in splits_path.parents
            or splits_path in output_path.parents
        ):
            raise ValueError("output_dir must not overlap splits_dir.")


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    splits_dir: Path | None = None,
    target_per_class: int | None = TARGET_PER_CLASS,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    validate_paths(input_dir, output_dir, splits_dir)
    rng = random.Random(seed)

    train_output = output_dir / "train"
    clean_dir(train_output)

    class_images = {
        cls: collect_images(input_dir / cls)
        for cls in discover_classes(input_dir)
    }
    resolved_target = resolve_target_per_class(class_images, target_per_class)
    targets = bucket_targets(resolved_target)
    stats: Dict[str, Dict[str, int]] = {}

    for cls, images in class_images.items():
        stats[cls] = augment_class(cls, images, output_dir, targets, rng)

    if splits_dir is not None:
        for split in EVAL_SPLITS:
            copied = copy_split(splits_dir, output_dir, split)
            stats[f"__copied_{split}"] = {"total": copied}

    return stats


def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    header = (
        f"{'class':<20} {'normal':>8} {'gaussian':>10} "
        f"{'motion':>8} {'unsharp':>8} {'total':>8}"
    )
    print(header)
    print("-" * len(header))
    totals = defaultdict(int)

    for cls, row in sorted(stats.items()):
        if cls.startswith("__copied_"):
            print(f"{cls:<20} {'':>8} {'':>10} {'':>8} {'':>8} {row.get('total', 0):>8}")
            continue

        total = sum(row.get(bucket, 0) for bucket in BUCKETS)
        for bucket in BUCKETS:
            totals[bucket] += row.get(bucket, 0)
        totals["total"] += total
        print(
            f"{cls:<20} {row.get('normal', 0):>8} "
            f"{row.get('gaussian_blur', 0):>10} "
            f"{row.get('motion_blur', 0):>8} "
            f"{row.get('unsharp_mask', 0):>8} {total:>8}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<20} {totals['normal']:>8} "
        f"{totals['gaussian_blur']:>10} {totals['motion_blur']:>8} "
        f"{totals['unsharp_mask']:>8} {totals['total']:>8}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Geo Version 2 from the existing dataset splits."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("data/splits/train"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/augmented_v2"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=TARGET_PER_CLASS,
        help="Images per class after augmentation. Defaults to the largest train class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats_json",
        type=Path,
        default=Path("outputs/geo_v2/data_prep_counts.json"),
    )
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
        report = {
            "version": "geo_v2",
            "config": {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "splits_dir": str(args.splits_dir),
                "target_per_class": args.target_per_class,
                "seed": args.seed,
                "bucket_ratios": BUCKET_RATIOS,
            },
            "stats": stats,
        }
        args.stats_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved Geo V2 report: {args.stats_json}")


if __name__ == "__main__":
    main()
