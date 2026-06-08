"""
Orchestrate the current VehicleTypeRecognition data pipeline.

This file is kept as a convenience wrapper. The canonical implementations are:

    src/data_split.py
    src/augment_offline.py

Pipeline:
    step 1: split raw data into train / valid_unseen / valid_traincopy / test
    step 2: prepare train-only balanced workspace
    step 3: quota-based offline augmentation

Training code is intentionally not modified by this wrapper.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

try:
    from .augment_offline import augment_dataset, print_stats as print_augment_stats
    from .data_split import IMG_EXTS, split_dataset, print_stats as print_split_stats
except ImportError:
    from augment_offline import augment_dataset, print_stats as print_augment_stats
    from data_split import IMG_EXTS, split_dataset, print_stats as print_split_stats


def copy_train_to_balanced(splits_dir: Path, balanced_dir: Path) -> Dict[str, int]:
    """Create a train-only balanced workspace without changing validation/test data."""
    src = splits_dir / "train"
    if not src.exists():
        raise FileNotFoundError(f"Missing train split: {src}")

    if balanced_dir.exists():
        shutil.rmtree(balanced_dir)
    shutil.copytree(src, balanced_dir)

    stats: Dict[str, int] = {}
    for class_dir in sorted(p for p in balanced_dir.iterdir() if p.is_dir()):
        stats[class_dir.name] = sum(
            1 for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    return stats


def print_balance_stats(stats: Dict[str, int]) -> None:
    header = f"{'class':<14} {'balanced_workspace':>20}"
    print(header)
    print("-" * len(header))
    total = 0
    for cls, count in sorted(stats.items()):
        total += count
        print(f"{cls:<14} {count:>20}")
    print("-" * len(header))
    print(f"{'TOTAL':<14} {total:>20}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VehicleTypeRecognition data pipeline")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--balanced_dir", type=Path, default=Path("data/balanced"))
    parser.add_argument("--augmented_dir", type=Path, default=Path("data/augmented"))
    parser.add_argument("--target_per_class", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats_json", type=Path, default=None)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--step", type=int, choices=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_stats: Dict[str, object] = {}

    if args.all or args.step == 1:
        split_stats = split_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.splits_dir,
            seed=args.seed,
        )
        print("\nSplit stats")
        print_split_stats(split_stats)
        all_stats["split"] = split_stats

    if args.all or args.step == 2:
        balance_stats = copy_train_to_balanced(args.splits_dir, args.balanced_dir)
        print("\nBalanced workspace stats")
        print_balance_stats(balance_stats)
        all_stats["balanced"] = balance_stats

    if args.all or args.step == 3:
        input_dir = args.balanced_dir if args.balanced_dir.exists() else args.splits_dir / "train"
        augment_stats = augment_dataset(
            input_dir=input_dir,
            output_dir=args.augmented_dir,
            splits_dir=args.splits_dir,
            target_per_class=args.target_per_class,
            seed=args.seed,
        )
        print("\nAugmentation stats")
        print_augment_stats(augment_stats)
        all_stats["augmented"] = augment_stats

    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
