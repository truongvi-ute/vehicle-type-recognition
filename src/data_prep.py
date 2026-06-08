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
import re
import shutil
from pathlib import Path
from typing import Dict, List

try:
    from .augment_offline import augment_dataset, bucket_targets, print_stats as print_augment_stats
    from .data_split import IMG_EXTS, split_dataset, print_stats as print_split_stats
except ImportError:
    from augment_offline import augment_dataset, bucket_targets, print_stats as print_augment_stats
    from data_split import IMG_EXTS, split_dataset, print_stats as print_split_stats

BUCKETS = ["normal", "rain", "sun", "night"]
WEATHER_BUCKETS = ["rain", "sun", "night"]
GENERATED_SUFFIX_RE = re.compile(r"_(normal|rain|sun|night)_(orig|geo)_\d{5}$")


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


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def count_class_images(root: Path) -> Dict[str, int]:
    if not root.exists():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[class_dir.name] = len(image_files(class_dir))
    return counts


def count_split_images(root: Path, splits: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        split: count_class_images(root / split)
        for split in splits
        if (root / split).exists()
    }


def count_augmented_train(root: Path) -> Dict[str, Dict[str, int]]:
    train_root = root / "train"
    if not train_root.exists():
        return {}

    counts: Dict[str, Dict[str, int]] = {}
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        class_counts: Dict[str, int] = {bucket: 0 for bucket in BUCKETS}
        for image_path in image_files(class_dir):
            match = GENERATED_SUFFIX_RE.search(image_path.stem)
            if match:
                class_counts[match.group(1)] += 1
        counts[class_dir.name] = class_counts
    return counts


def count_augmented_train_markers(root: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    train_root = root / "train"
    if not train_root.exists():
        return {}

    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        class_counts: Dict[str, Dict[str, int]] = {
            bucket: {"orig": 0, "geo": 0}
            for bucket in BUCKETS
        }
        for image_path in image_files(class_dir):
            match = GENERATED_SUFFIX_RE.search(image_path.stem)
            if match:
                class_counts[match.group(1)][match.group(2)] += 1
        counts[class_dir.name] = class_counts
    return counts


def total_counts(counts: Dict[str, int]) -> int:
    return sum(counts.values())


def nested_totals(counts: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {
        name: total_counts(row)
        for name, row in counts.items()
    }


def resolve_report_target(source_counts: Dict[str, int], target_per_class: int | None) -> int | None:
    if target_per_class is not None:
        return target_per_class
    if not source_counts:
        return None
    return max(source_counts.values())


def policy_case(source_count: int, target_per_class: int, normal_target: int) -> str:
    if source_count >= target_per_class:
        return "at_or_above_target"
    if source_count >= normal_target:
        return "between_70_and_100_percent"
    return "below_70_percent"


def expected_marker_counts(source_count: int, targets: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    target_total = sum(targets.get(bucket, 0) for bucket in BUCKETS)
    normal_target = targets["normal"]
    weather_target = target_total - normal_target

    if source_count >= target_total:
        normal_orig = normal_target
        normal_geo = 0
        weather_orig = weather_target
        weather_geo = 0
    elif source_count >= normal_target:
        excess = source_count - normal_target
        normal_orig = normal_target
        normal_geo = 0
        weather_orig = excess
        weather_geo = weather_target - excess
    else:
        normal_orig = source_count
        normal_geo = normal_target - source_count
        weather_orig = 0
        weather_geo = weather_target

    return {
        "normal": {
            "orig": normal_orig,
            "geo": normal_geo,
        },
        "weather_total": {
            "orig": weather_orig,
            "geo": weather_geo,
        },
    }


def compact_marker_counts(row: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    normal = row.get("normal", {"orig": 0, "geo": 0})
    weather = {
        "orig": sum(row.get(bucket, {}).get("orig", 0) for bucket in WEATHER_BUCKETS),
        "geo": sum(row.get(bucket, {}).get("geo", 0) for bucket in WEATHER_BUCKETS),
    }
    return {
        "normal": normal,
        "weather_total": weather,
        "by_bucket": row,
    }


def build_class_detail_report(
    snapshot: Dict[str, object],
    target_per_class: int | None,
) -> Dict[str, object]:
    splits = snapshot.get("splits", {}).get("per_split", {})  # type: ignore[union-attr]
    split_train_counts = splits.get("train", {}) if isinstance(splits, dict) else {}
    valid_unseen_counts = splits.get("valid_unseen", {}) if isinstance(splits, dict) else {}
    valid_traincopy_counts = splits.get("valid_traincopy", {}) if isinstance(splits, dict) else {}
    test_counts = splits.get("test", {}) if isinstance(splits, dict) else {}
    raw_counts = snapshot.get("raw", {}).get("per_class", {})  # type: ignore[union-attr]
    balanced_counts = snapshot.get("balanced", {}).get("per_class", {})  # type: ignore[union-attr]
    source_counts = balanced_counts or split_train_counts
    resolved_target = resolve_report_target(source_counts, target_per_class)

    augmented = snapshot.get("augmented", {})  # type: ignore[assignment]
    augmented_buckets = augmented.get("train_per_class_bucket", {}) if isinstance(augmented, dict) else {}
    augmented_markers = (
        augmented.get("train_per_class_bucket_marker", {})
        if isinstance(augmented, dict)
        else {}
    )

    class_names = sorted(
        set(raw_counts)
        | set(split_train_counts)
        | set(balanced_counts)
        | set(augmented_buckets)
    )

    if resolved_target is None:
        return {
            "target_per_class": None,
            "bucket_targets": {},
            "classes": {},
            "summary": {
                "classes": len(class_names),
                "policy_checked": False,
                "policy_ok_classes": 0,
                "policy_failed_classes": 0,
            },
        }

    targets = bucket_targets(resolved_target)
    classes: Dict[str, object] = {}
    policy_ok_count = 0
    policy_fail_count = 0

    for cls in class_names:
        source_count = int(source_counts.get(cls, 0))
        actual_buckets = augmented_buckets.get(cls, {})
        actual_markers_raw = augmented_markers.get(cls, {})
        actual_markers = compact_marker_counts(actual_markers_raw)
        expected_markers = expected_marker_counts(source_count, targets)
        actual_total = sum(actual_buckets.get(bucket, 0) for bucket in BUCKETS)

        expected_flat = (
            expected_markers["normal"]["orig"],
            expected_markers["normal"]["geo"],
            expected_markers["weather_total"]["orig"],
            expected_markers["weather_total"]["geo"],
        )
        actual_flat = (
            actual_markers["normal"]["orig"],  # type: ignore[index]
            actual_markers["normal"]["geo"],  # type: ignore[index]
            actual_markers["weather_total"]["orig"],  # type: ignore[index]
            actual_markers["weather_total"]["geo"],  # type: ignore[index]
        )

        bucket_ok = all(actual_buckets.get(bucket, 0) == targets[bucket] for bucket in BUCKETS)
        total_ok = actual_total == resolved_target
        marker_ok = actual_flat == expected_flat
        has_augmented_output = bool(actual_buckets)
        class_policy_ok = has_augmented_output and bucket_ok and total_ok and marker_ok

        if has_augmented_output:
            if class_policy_ok:
                policy_ok_count += 1
            else:
                policy_fail_count += 1

        normal_target = targets["normal"]
        weather_target = resolved_target - normal_target
        source_for_policy = min(source_count, resolved_target)
        excess_over_normal = max(0, source_for_policy - normal_target)
        weather_shortfall = max(0, weather_target - excess_over_normal)
        classes[cls] = {
            "source": {
                "raw": raw_counts.get(cls, 0),
                "split_train": split_train_counts.get(cls, 0),
                "valid_unseen": valid_unseen_counts.get(cls, 0),
                "valid_traincopy": valid_traincopy_counts.get(cls, 0),
                "test": test_counts.get(cls, 0),
                "balanced_or_train_source": source_count,
            },
            "target": {
                "class_total": resolved_target,
                "normal_70_percent": normal_target,
                "weather_30_percent": resolved_target - normal_target,
                "bucket_targets": targets,
                "policy_case": policy_case(source_count, resolved_target, normal_target),
            },
            "expected": {
                "augmentation_needed": max(0, resolved_target - source_count),
                "capped_originals": max(0, source_count - resolved_target),
                "excess_over_normal_70": excess_over_normal,
                "weather_shortfall": weather_shortfall,
                "marker_counts": expected_markers,
            },
            "actual": {
                "bucket_counts": actual_buckets,
                "total": actual_total,
                "marker_counts": actual_markers,
            },
            "policy_check": {
                "has_augmented_output": has_augmented_output,
                "bucket_counts_ok": bucket_ok,
                "total_ok": total_ok,
                "marker_counts_ok": marker_ok,
                "ok": class_policy_ok,
            },
        }

    return {
        "target_per_class": resolved_target,
        "bucket_targets": targets,
        "classes": classes,
        "summary": {
            "classes": len(class_names),
            "policy_checked": bool(augmented_buckets),
            "policy_ok_classes": policy_ok_count,
            "policy_failed_classes": policy_fail_count,
        },
    }


def build_matrix_tables(class_details: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    classes = class_details.get("classes", {})
    if not isinstance(classes, dict):
        classes = {}

    split_rows = []
    augmentation_rows = []
    split_totals = {
        "raw": 0,
        "train": 0,
        "valid_unseen": 0,
        "valid_traincopy": 0,
        "test": 0,
    }
    augmentation_totals = {
        "train_source": 0,
        "physical_added": 0,
        "geo_generated": 0,
        "normal_orig": 0,
        "normal_geo": 0,
        "rain_orig": 0,
        "rain_geo": 0,
        "sun_orig": 0,
        "sun_geo": 0,
        "night_orig": 0,
        "night_geo": 0,
        "final_total": 0,
    }

    for cls, detail in sorted(classes.items()):
        if not isinstance(detail, dict):
            continue
        source = detail.get("source", {})
        expected = detail.get("expected", {})
        actual = detail.get("actual", {})

        source = source if isinstance(source, dict) else {}
        expected = expected if isinstance(expected, dict) else {}
        actual = actual if isinstance(actual, dict) else {}

        marker_counts = actual.get("marker_counts", {})
        marker_counts = marker_counts if isinstance(marker_counts, dict) else {}
        normal_markers = marker_counts.get("normal", {})
        by_bucket = marker_counts.get("by_bucket", {})
        normal_markers = normal_markers if isinstance(normal_markers, dict) else {}
        by_bucket = by_bucket if isinstance(by_bucket, dict) else {}

        valid_unseen = int(source.get("valid_unseen", 0))
        valid_traincopy = int(source.get("valid_traincopy", 0))
        test = int(source.get("test", 0))
        raw = int(source.get("raw", 0))
        train = int(source.get("split_train", 0))
        split_row = [cls, raw, train, valid_unseen, valid_traincopy, test]
        split_rows.append(split_row)
        for key, value in zip(split_totals, split_row[1:]):
            split_totals[key] += int(value)

        rain = by_bucket.get("rain", {}) if isinstance(by_bucket.get("rain", {}), dict) else {}
        sun = by_bucket.get("sun", {}) if isinstance(by_bucket.get("sun", {}), dict) else {}
        night = by_bucket.get("night", {}) if isinstance(by_bucket.get("night", {}), dict) else {}
        normal_orig = int(normal_markers.get("orig", 0))
        normal_geo = int(normal_markers.get("geo", 0))
        rain_orig = int(rain.get("orig", 0))
        rain_geo = int(rain.get("geo", 0))
        sun_orig = int(sun.get("orig", 0))
        sun_geo = int(sun.get("geo", 0))
        night_orig = int(night.get("orig", 0))
        night_geo = int(night.get("geo", 0))
        geo_generated = normal_geo + rain_geo + sun_geo + night_geo
        augmentation_row = [
            cls,
            int(source.get("balanced_or_train_source", 0)),
            int(expected.get("augmentation_needed", 0)),
            geo_generated,
            normal_orig,
            normal_geo,
            rain_orig,
            rain_geo,
            sun_orig,
            sun_geo,
            night_orig,
            night_geo,
            int(actual.get("total", 0)),
        ]
        augmentation_rows.append(augmentation_row)
        for key, value in zip(augmentation_totals, augmentation_row[1:]):
            augmentation_totals[key] += int(value)

    split_rows.append([
        "TOTAL",
        split_totals["raw"],
        split_totals["train"],
        split_totals["valid_unseen"],
        split_totals["valid_traincopy"],
        split_totals["test"],
    ])
    augmentation_rows.append([
        "TOTAL",
        augmentation_totals["train_source"],
        augmentation_totals["physical_added"],
        augmentation_totals["geo_generated"],
        augmentation_totals["normal_orig"],
        augmentation_totals["normal_geo"],
        augmentation_totals["rain_orig"],
        augmentation_totals["rain_geo"],
        augmentation_totals["sun_orig"],
        augmentation_totals["sun_geo"],
        augmentation_totals["night_orig"],
        augmentation_totals["night_geo"],
        augmentation_totals["final_total"],
    ])

    return {
        "split_dataset_matrix": {
            "columns": ["class", "raw", "train", "valid_unseen", "valid_traincopy", "test"],
            "rows": split_rows,
        },
        "augmentation_summary_matrix": {
            "columns": [
                "class",
                "train_source",
                "physical_added",
                "geo_generated",
                "normal_orig",
                "normal_geo",
                "rain_orig",
                "rain_geo",
                "sun_orig",
                "sun_geo",
                "night_orig",
                "night_geo",
                "final_total",
            ],
            "rows": augmentation_rows,
        },
    }


def build_folder_snapshot(
    raw_dir: Path,
    splits_dir: Path,
    balanced_dir: Path,
    augmented_dir: Path,
) -> Dict[str, object]:
    split_names = ["train", "valid_unseen", "valid_traincopy", "test"]
    raw_counts = count_class_images(raw_dir)
    split_counts = count_split_images(splits_dir, split_names)
    balanced_counts = count_class_images(balanced_dir)
    augmented_train_counts = count_augmented_train(augmented_dir)
    augmented_train_marker_counts = count_augmented_train_markers(augmented_dir)
    augmented_eval_counts = count_split_images(
        augmented_dir,
        ["valid_unseen", "valid_traincopy", "test"],
    )

    return {
        "raw": {
            "per_class": raw_counts,
            "total": total_counts(raw_counts),
        },
        "splits": {
            "per_split": split_counts,
            "totals": nested_totals(split_counts),
        },
        "balanced": {
            "per_class": balanced_counts,
            "total": total_counts(balanced_counts),
        },
        "augmented": {
            "train_per_class_bucket": augmented_train_counts,
            "train_per_class_bucket_marker": augmented_train_marker_counts,
            "train_totals_per_class": nested_totals(augmented_train_counts),
            "eval_splits": augmented_eval_counts,
            "eval_totals": nested_totals(augmented_eval_counts),
        },
    }


def build_count_report(
    raw_dir: Path,
    splits_dir: Path,
    balanced_dir: Path,
    augmented_dir: Path,
    target_per_class: int | None,
    seed: int,
    operation_stats: Dict[str, object],
) -> Dict[str, object]:
    snapshot = build_folder_snapshot(
        raw_dir=raw_dir,
        splits_dir=splits_dir,
        balanced_dir=balanced_dir,
        augmented_dir=augmented_dir,
    )
    class_details = build_class_detail_report(snapshot, target_per_class)

    return {
        "config": {
            "raw_dir": str(raw_dir),
            "splits_dir": str(splits_dir),
            "balanced_dir": str(balanced_dir),
            "augmented_dir": str(augmented_dir),
            "target_per_class": target_per_class,
            "target_policy": (
                "largest train class size"
                if target_per_class is None
                else "manual override"
            ),
            "seed": seed,
        },
        "snapshot": snapshot,
        "class_details": class_details,
        "matrices": build_matrix_tables(class_details),
        "operation_stats": operation_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VehicleTypeRecognition data pipeline")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--balanced_dir", type=Path, default=Path("data/balanced"))
    parser.add_argument("--augmented_dir", type=Path, default=Path("data/augmented"))
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=None,
        help="Images per class after augmentation. Defaults to the largest train class size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats_json",
        type=Path,
        default=Path("outputs/data_prep_counts.json"),
        help="JSON report with current class counts, policy checks, and matrix tables.",
    )

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

    count_report = build_count_report(
        raw_dir=args.raw_dir,
        splits_dir=args.splits_dir,
        balanced_dir=args.balanced_dir,
        augmented_dir=args.augmented_dir,
        target_per_class=args.target_per_class,
        seed=args.seed,
        operation_stats=all_stats,
    )
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(count_report, indent=2), encoding="utf-8")
    print(f"\nSaved data count report: {args.stats_json}")


if __name__ == "__main__":
    main()
