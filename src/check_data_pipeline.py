"""
Check data pipeline outputs.

Reports:
    - image counts per class for split folders
    - image counts per class and per bucket for augmented train
    - whether test images appear to have been augmented
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
BUCKETS = ["normal", "rain", "sun", "night"]
GENERATED_SUFFIX_RE = re.compile(r"_(normal|rain|sun|night)_(orig|geo)_\d{5}$")


def image_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def class_counts(root: Path) -> Dict[str, int]:
    if not root.exists():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[class_dir.name] = len(image_files(class_dir))
    return counts


def bucket_counts(train_root: Path) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    if not train_root.exists():
        return counts
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        counts[class_dir.name] = {}
        for bucket in BUCKETS:
            counts[class_dir.name][bucket] = len(image_files(class_dir / bucket))
    return counts


def print_class_counts(title: str, counts: Dict[str, int]) -> None:
    print(f"\n{title}")
    print(f"{'class':<14} {'count':>8}")
    print("-" * 24)
    total = 0
    for cls, count in sorted(counts.items()):
        total += count
        print(f"{cls:<14} {count:>8}")
    print("-" * 24)
    print(f"{'TOTAL':<14} {total:>8}")


def print_bucket_counts(counts: Dict[str, Dict[str, int]]) -> None:
    print("\nAugmented train bucket counts")
    header = f"{'class':<14} {'normal':>8} {'rain':>8} {'sun':>8} {'night':>8} {'total':>8}"
    print(header)
    print("-" * len(header))
    totals = {bucket: 0 for bucket in BUCKETS}
    grand_total = 0
    for cls, row in sorted(counts.items()):
        total = sum(row.get(bucket, 0) for bucket in BUCKETS)
        grand_total += total
        for bucket in BUCKETS:
            totals[bucket] += row.get(bucket, 0)
        print(
            f"{cls:<14} {row.get('normal', 0):>8} {row.get('rain', 0):>8} "
            f"{row.get('sun', 0):>8} {row.get('night', 0):>8} {total:>8}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<14} {totals['normal']:>8} {totals['rain']:>8} "
        f"{totals['sun']:>8} {totals['night']:>8} {grand_total:>8}"
    )


def canonical_stem(path: Path) -> str:
    return GENERATED_SUFFIX_RE.sub("", path.stem)


def stems_by_class(root: Path) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    if not root.exists():
        return result
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        result[class_dir.name] = {canonical_stem(p) for p in image_files(class_dir)}
    return result


def augmented_train_stems_by_class(train_root: Path) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    if not train_root.exists():
        return result
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        result[class_dir.name] = {canonical_stem(p) for p in image_files(class_dir)}
    return result


def check_test_not_augmented(splits_dir: Path, augmented_dir: Path) -> bool:
    ok = True

    augmented_test = augmented_dir / "test"
    for bucket in BUCKETS:
        if list(augmented_test.rglob(bucket)):
            print(f"[FAIL] Found bucket folder named '{bucket}' under augmented test.")
            ok = False

    for p in image_files(augmented_test):
        if GENERATED_SUFFIX_RE.search(p.stem):
            print(f"[FAIL] Test file looks augmented: {p}")
            ok = False

    test_stems = stems_by_class(splits_dir / "test")
    train_aug_stems = augmented_train_stems_by_class(augmented_dir / "train")
    for cls, stems in test_stems.items():
        overlap = stems.intersection(train_aug_stems.get(cls, set()))
        if overlap:
            sample = ", ".join(sorted(list(overlap))[:5])
            print(f"[FAIL] Possible test/train augmented overlap in class '{cls}': {sample}")
            ok = False

    if ok:
        print("\n[OK] No augmented test files or test/train augmented stem overlaps detected.")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check data pipeline outputs")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--augmented_dir", type=Path, default=Path("data/augmented"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for split in ["train", "valid_unseen", "valid_traincopy", "test"]:
        print_class_counts(f"Split counts: {split}", class_counts(args.splits_dir / split))

    print_bucket_counts(bucket_counts(args.augmented_dir / "train"))

    ok = check_test_not_augmented(args.splits_dir, args.augmented_dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
