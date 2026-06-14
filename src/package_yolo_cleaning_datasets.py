"""Validate and package the local V1/V2 cleaning datasets for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = [
    "bicycle",
    "boat",
    "bus",
    "car",
    "helicopter",
    "minibus",
    "motorcycle",
    "taxi",
    "train",
    "truck",
]
PRIMARY_SPLITS = ["train", "valid_unseen", "valid_traincopy", "test"]
EVALUATION_SPLITS = ["valid_unseen", "valid_traincopy", "test"]


def image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_variant(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"root": str(root.resolve()), "splits": {}}
    for split in PRIMARY_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")
        actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if actual_classes != CLASS_NAMES:
            raise ValueError(
                f"Invalid classes in {split_dir}. Expected={CLASS_NAMES}, actual={actual_classes}"
            )
        per_class = {
            class_name: len(image_files(split_dir / class_name))
            for class_name in CLASS_NAMES
        }
        empty = [name for name, count in per_class.items() if count == 0]
        if empty:
            raise ValueError(f"Empty classes in {split}: {empty}")
        report["splits"][split] = {
            "total": sum(per_class.values()),
            "per_class": per_class,
        }
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(root: Path, split: str) -> dict[str, str]:
    split_dir = root / split
    return {
        path.relative_to(split_dir).as_posix(): sha256(path)
        for path in image_files(split_dir)
    }


def verify_shared_evaluation_splits(v1_dir: Path, v2_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in EVALUATION_SPLITS:
        print(f"Comparing V1 and V2 {split} files...")
        v1_fingerprint = split_fingerprint(v1_dir, split)
        v2_fingerprint = split_fingerprint(v2_dir, split)
        if v1_fingerprint != v2_fingerprint:
            v1_names = set(v1_fingerprint)
            v2_names = set(v2_fingerprint)
            missing_in_v2 = sorted(v1_names - v2_names)[:20]
            missing_in_v1 = sorted(v2_names - v1_names)[:20]
            changed = sorted(
                name
                for name in v1_names & v2_names
                if v1_fingerprint[name] != v2_fingerprint[name]
            )[:20]
            raise ValueError(
                f"V1/V2 mismatch in {split}. Missing in V2={missing_in_v2}; "
                f"missing in V1={missing_in_v1}; changed={changed}"
            )
        report[split] = {
            "identical": True,
            "files": len(v1_fingerprint),
        }
    return report


def add_tree(archive: zipfile.ZipFile, source: Path, archive_root: str) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(source).as_posix()
        archive_name = f"{archive_root}/{relative}"
        compression = (
            zipfile.ZIP_STORED
            if path.suffix.lower() in IMAGE_EXTENSIONS
            else zipfile.ZIP_DEFLATED
        )
        archive.write(path, archive_name, compress_type=compression)


def create_package(
    dataset_dir: Path,
    dataset_root_name: str,
    src_dir: Path,
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        add_tree(archive, dataset_dir, dataset_root_name)
        add_tree(archive, src_dir, "src")
        archive.writestr(
            "dataset_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            compress_type=zipfile.ZIP_DEFLATED,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and package V1/V2 cleaning datasets for YOLO on Kaggle."
    )
    parser.add_argument("--v1_dir", type=Path, default=PROJECT_ROOT / "data" / "augmented")
    parser.add_argument("--v2_dir", type=Path, default=PROJECT_ROOT / "data" / "augmented_v2")
    parser.add_argument("--src_dir", type=Path, default=PROJECT_ROOT / "src")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "kaggle_datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_dir = args.v1_dir.expanduser().resolve()
    v2_dir = args.v2_dir.expanduser().resolve()
    src_dir = args.src_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    print("Validating Augmented V1 Dataset Cleaning...")
    v1_report = validate_variant(v1_dir)
    print("Validating Augmented V2 Dataset Cleaning...")
    v2_report = validate_variant(v2_dir)
    shared_report = verify_shared_evaluation_splits(v1_dir, v2_dir)

    common = {
        "expected_classes": CLASS_NAMES,
        "shared_evaluation_splits": shared_report,
        "note": (
            "V1 and V2 use the same local split generated with seed 42. "
            "Only the offline train augmentation pipelines differ."
        ),
    }
    v1_manifest = {
        "dataset_name": "Vehicle-YOLO-AugmentedV1-DatasetCleaning",
        "variant": "v1",
        "offline_buckets": ["normal", "rain", "sun", "night"],
        "dataset": v1_report,
        **common,
    }
    v2_manifest = {
        "dataset_name": "Vehicle-YOLO-AugmentedV2-DatasetCleaning",
        "variant": "v2",
        "offline_buckets": ["normal", "gaussian_blur", "motion_blur", "unsharp_mask"],
        "dataset": v2_report,
        **common,
    }

    v1_zip = output_dir / "Vehicle-YOLO-AugmentedV1-DatasetCleaning.zip"
    v2_zip = output_dir / "Vehicle-YOLO-AugmentedV2-DatasetCleaning.zip"
    print(f"Creating {v1_zip}...")
    create_package(v1_dir, "augmented", src_dir, v1_zip, v1_manifest)
    print(f"Creating {v2_zip}...")
    create_package(v2_dir, "augmented_v2", src_dir, v2_zip, v2_manifest)

    print("Packages ready:")
    print(f"  {v1_zip}")
    print(f"  {v2_zip}")


if __name__ == "__main__":
    main()
