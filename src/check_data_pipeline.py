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
from typing import Dict, List, Set

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
BUCKETS = ["normal", "rain", "sun", "night"]
WEATHER_BUCKETS = ["rain", "sun", "night"]
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
        counts[class_dir.name] = {bucket: 0 for bucket in BUCKETS}
        for image_path in image_files(class_dir):
            match = GENERATED_SUFFIX_RE.search(image_path.stem)
            if match:
                counts[class_dir.name][match.group(1)] += 1
    return counts


def marker_counts(train_root: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not train_root.exists():
        return counts
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        counts[class_dir.name] = {
            bucket: {"orig": 0, "geo": 0}
            for bucket in BUCKETS
        }
        for image_path in image_files(class_dir):
            match = GENERATED_SUFFIX_RE.search(image_path.stem)
            if match:
                counts[class_dir.name][match.group(1)][match.group(2)] += 1
    return counts


def bucket_targets(target_per_class: int) -> Dict[str, int]:
    normal = int(round(target_per_class * 0.70))
    rain = int(round(target_per_class * 0.10))
    sun = int(round(target_per_class * 0.10))
    night = target_per_class - normal - rain - sun
    return {
        "normal": normal,
        "rain": rain,
        "sun": sun,
        "night": night,
    }


def infer_target_per_class(counts: Dict[str, Dict[str, int]]) -> int:
    return max((sum(row.get(bucket, 0) for bucket in BUCKETS) for row in counts.values()), default=0)


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


def print_marker_counts(counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    print("\nAugmented train orig/geo counts")
    header = (
        f"{'class':<14} {'normal_o':>8} {'normal_g':>8} "
        f"{'weather_o':>9} {'weather_g':>9}"
    )
    print(header)
    print("-" * len(header))
    totals = {"normal_o": 0, "normal_g": 0, "weather_o": 0, "weather_g": 0}
    for cls, row in sorted(counts.items()):
        normal_o = row.get("normal", {}).get("orig", 0)
        normal_g = row.get("normal", {}).get("geo", 0)
        weather_o = sum(row.get(bucket, {}).get("orig", 0) for bucket in WEATHER_BUCKETS)
        weather_g = sum(row.get(bucket, {}).get("geo", 0) for bucket in WEATHER_BUCKETS)
        totals["normal_o"] += normal_o
        totals["normal_g"] += normal_g
        totals["weather_o"] += weather_o
        totals["weather_g"] += weather_g
        print(f"{cls:<14} {normal_o:>8} {normal_g:>8} {weather_o:>9} {weather_g:>9}")
    print("-" * len(header))
    print(
        f"{'TOTAL':<14} {totals['normal_o']:>8} {totals['normal_g']:>8} "
        f"{totals['weather_o']:>9} {totals['weather_g']:>9}"
    )


def check_augmented_policy(
    source_counts: Dict[str, int],
    buckets: Dict[str, Dict[str, int]],
    markers: Dict[str, Dict[str, Dict[str, int]]],
    target_per_class: int,
) -> bool:
    if target_per_class <= 0:
        print("\n[FAIL] Cannot check augmentation policy because target_per_class is 0.")
        return False

    ok = True
    targets = bucket_targets(target_per_class)
    normal_target = targets["normal"]
    weather_target = target_per_class - normal_target

    print(f"\nAugmentation policy check target={target_per_class}")
    for cls, row in sorted(buckets.items()):
        total = sum(row.get(bucket, 0) for bucket in BUCKETS)
        if total != target_per_class:
            print(f"[FAIL] {cls}: total={total}, expected={target_per_class}")
            ok = False
        for bucket in BUCKETS:
            if row.get(bucket, 0) != targets[bucket]:
                print(f"[FAIL] {cls}: {bucket}={row.get(bucket, 0)}, expected={targets[bucket]}")
                ok = False

        original_count = source_counts.get(cls, 0)
        marker_row = markers.get(cls, {})
        normal_orig = marker_row.get("normal", {}).get("orig", 0)
        normal_geo = marker_row.get("normal", {}).get("geo", 0)
        weather_orig = sum(marker_row.get(bucket, {}).get("orig", 0) for bucket in WEATHER_BUCKETS)
        weather_geo = sum(marker_row.get(bucket, {}).get("geo", 0) for bucket in WEATHER_BUCKETS)

        if original_count >= target_per_class:
            expected = (normal_target, 0, weather_target, 0)
        elif original_count >= normal_target:
            excess = original_count - normal_target
            expected = (normal_target, 0, excess, weather_target - excess)
        else:
            expected = (original_count, normal_target - original_count, 0, weather_target)

        actual = (normal_orig, normal_geo, weather_orig, weather_geo)
        if actual != expected:
            print(
                f"[FAIL] {cls}: orig/geo normal/weather={actual}, expected={expected} "
                f"from source_count={original_count}"
            )
            ok = False

    if ok:
        print("[OK] Augmented train follows the 70% normal then 30% weather fill policy.")
    return ok


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
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=None,
        help="Expected augmented images per class. Defaults to the largest augmented train class total.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for split in ["train", "valid_unseen", "valid_traincopy", "test"]:
        print_class_counts(f"Split counts: {split}", class_counts(args.splits_dir / split))

    train_bucket_counts = bucket_counts(args.augmented_dir / "train")
    train_marker_counts = marker_counts(args.augmented_dir / "train")
    print_bucket_counts(train_bucket_counts)
    print_marker_counts(train_marker_counts)

    ok = check_test_not_augmented(args.splits_dir, args.augmented_dir)
    target_per_class = args.target_per_class or infer_target_per_class(train_bucket_counts)
    ok = check_augmented_policy(
        class_counts(args.splits_dir / "train"),
        train_bucket_counts,
        train_marker_counts,
        target_per_class,
    ) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
