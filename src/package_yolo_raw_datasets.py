"""Validate and package Raw Original V1/V2 datasets for Kaggle.

The local folders use explicit ``augmented_raw_v1`` and ``augmented_raw_v2``
names. Inside each ZIP, they are normalized to ``augmented`` and
``augmented_v2`` so the Kaggle notebooks have the same layout as the cleaning
experiments.
"""

from __future__ import annotations

from pathlib import Path

from package_yolo_cleaning_datasets import (
    CLASS_NAMES,
    create_package,
    validate_variant,
    verify_shared_evaluation_splits,
)

import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package Raw Original V1/V2 datasets for YOLO on Kaggle."
    )
    parser.add_argument(
        "--v1_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "augmented_raw_v1",
    )
    parser.add_argument(
        "--v2_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "augmented_raw_v2",
    )
    parser.add_argument("--src_dir", type=Path, default=PROJECT_ROOT / "src")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle_datasets",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run all dataset checks without creating the large ZIP packages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_dir = args.v1_dir.expanduser().resolve()
    v2_dir = args.v2_dir.expanduser().resolve()
    src_dir = args.src_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    print("Validating Raw Original V1...")
    v1_report = validate_variant(v1_dir)
    print("Validating Raw Original V2...")
    v2_report = validate_variant(v2_dir)
    shared_report = verify_shared_evaluation_splits(v1_dir, v2_dir)

    expected_train_total = 75_620
    for variant, report in (("v1", v1_report), ("v2", v2_report)):
        actual = report["splits"]["train"]["total"]
        if actual != expected_train_total:
            raise ValueError(
                f"Unexpected Raw Original {variant.upper()} train total: "
                f"expected={expected_train_total}, actual={actual}"
            )

    common = {
        "dataset_id": "raw_original",
        "source_dataset": "data/raw",
        "expected_classes": CLASS_NAMES,
        "split_seed": 42,
        "shared_evaluation_splits": shared_report,
        "note": (
            "Raw Original V1 and V2 reuse the same split generated with seed 42. "
            "Only the offline train augmentation pipeline differs."
        ),
    }
    v1_manifest = {
        "dataset_name": "Vehicle-YOLO-AugmentedV1-RawOriginal",
        "variant": "v1",
        "kaggle_data_root": "augmented",
        "offline_buckets": ["normal", "rain", "sun", "night"],
        "dataset": v1_report,
        **common,
    }
    v2_manifest = {
        "dataset_name": "Vehicle-YOLO-AugmentedV2-RawOriginal",
        "variant": "v2",
        "kaggle_data_root": "augmented_v2",
        "offline_buckets": [
            "normal",
            "gaussian_blur",
            "motion_blur",
            "unsharp_mask",
        ],
        "dataset": v2_report,
        **common,
    }

    if args.validate_only:
        print("Validation completed. ZIP creation was skipped.")
        return

    v1_zip = output_dir / "Vehicle-YOLO-AugmentedV1-RawOriginal.zip"
    v2_zip = output_dir / "Vehicle-YOLO-AugmentedV2-RawOriginal.zip"
    print(f"Creating {v1_zip}...")
    create_package(v1_dir, "augmented", src_dir, v1_zip, v1_manifest)
    print(f"Creating {v2_zip}...")
    create_package(v2_dir, "augmented_v2", src_dir, v2_zip, v2_manifest)

    print("Packages ready:")
    print(f"  {v1_zip}")
    print(f"  {v2_zip}")


if __name__ == "__main__":
    main()
