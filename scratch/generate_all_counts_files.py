import os
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
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

def count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

def get_class_counts(parent_dir: Path) -> dict[str, int]:
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = parent_dir / cls
        counts[cls] = count_images_in_dir(cls_dir)
    return counts

def generate_counts():
    print("Generating dataset counts files...")
    
    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy data_raw_original_v1_counts.json from outputs/vit/data_prep_counts.json
    vit_counts_path = OUTPUTS_DIR / "vit" / "data_prep_counts.json"
    raw_v1_counts_path = OUTPUTS_DIR / "data_raw_original_v1_counts.json"
    if vit_counts_path.is_file():
        shutil.copy2(vit_counts_path, raw_v1_counts_path)
        print(f"Copied {vit_counts_path} -> {raw_v1_counts_path}")
    else:
        # Fallback if vit/data_prep_counts.json was not found
        print(f"WARNING: {vit_counts_path} not found!")
        
    # 2. Copy data_raw_original_v2_counts.json from outputs/geo_v2/data_prep_counts.json
    geo_v2_counts_path = OUTPUTS_DIR / "geo_v2" / "data_prep_counts.json"
    raw_v2_counts_path = OUTPUTS_DIR / "data_raw_original_v2_counts.json"
    if geo_v2_counts_path.is_file():
        shutil.copy2(geo_v2_counts_path, raw_v2_counts_path)
        print(f"Copied {geo_v2_counts_path} -> {raw_v2_counts_path}")
    else:
        print(f"WARNING: {geo_v2_counts_path} not found!")

    # 3. Reconstruct data_cleaning_v1_counts.json
    print("\nProcessing data_cleaning_v1_counts.json...")
    clean_raw_dir = DATA_DIR / "raw-cleaning" / "raw"
    clean_splits_dir = DATA_DIR / "cleaning" / "splits"
    clean_balanced_dir = DATA_DIR / "cleaning" / "balanced"
    clean_augmented_dir = DATA_DIR / "augmented"  # Weather augmented clean train set
    
    # Scan raw class counts
    raw_counts = get_class_counts(clean_raw_dir)
    raw_total = sum(raw_counts.values())
    
    # Scan splits class counts
    splits_per_class = {}
    split_totals = {}
    for split in ["train", "valid_unseen", "valid_traincopy", "test"]:
        counts = get_class_counts(clean_splits_dir / split)
        splits_per_class[split] = counts
        split_totals[split] = sum(counts.values())
        
    # Scan balanced class counts
    balanced_counts = get_class_counts(clean_balanced_dir)
    balanced_total = sum(balanced_counts.values())
    
    # Scan augmented train buckets counts
    augmented_buckets = {}
    for cls in CLASS_NAMES:
        cls_augmented_dir = clean_augmented_dir / "train" / cls
        buckets = {"normal": 0, "rain": 0, "sun": 0, "night": 0}
        if cls_augmented_dir.exists():
            for p in cls_augmented_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    stem = p.stem.lower()
                    if "_rain_" in stem:
                        buckets["rain"] += 1
                    elif "_sun_" in stem:
                        buckets["sun"] += 1
                    elif "_night_" in stem:
                        buckets["night"] += 1
                    else:
                        buckets["normal"] += 1
        augmented_buckets[cls] = buckets

    cleaning_v1_data = {
        "config": {
            "raw_dir": "data\\raw-cleaning\\raw",
            "splits_dir": "data\\cleaning\\splits",
            "balanced_dir": "data\\cleaning\\balanced",
            "augmented_dir": "data\\augmented",
            "target_per_class": None,
            "seed": 42
        },
        "snapshot": {
            "raw": {
                "per_class": raw_counts,
                "total": raw_total
            },
            "splits": {
                "per_split": splits_per_class,
                "totals": split_totals
            },
            "balanced": {
                "per_class": balanced_counts,
                "total": balanced_total
            },
            "augmented": {
                "train_per_class_bucket": augmented_buckets
            }
        }
    }
    
    cleaning_v1_counts_path = OUTPUTS_DIR / "data_cleaning_v1_counts.json"
    with open(cleaning_v1_counts_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_v1_data, f, indent=2)
    print(f"Created {cleaning_v1_counts_path}")

    # 4. Reconstruct data_cleaning_v2_counts.json (V2 statistics format)
    print("\nProcessing data_cleaning_v2_counts.json...")
    clean_augmented_v2_dir = DATA_DIR / "augmented_v2" # Blur augmented clean train set
    
    augmented_v2_buckets = {}
    for cls in CLASS_NAMES:
        cls_augmented_v2_dir = clean_augmented_v2_dir / "train" / cls
        buckets = {"normal": 0, "gaussian_blur": 0, "motion_blur": 0, "unsharp_mask": 0}
        if cls_augmented_v2_dir.exists():
            for p in cls_augmented_v2_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    stem = p.stem.lower()
                    if "_gaussian_blur_" in stem:
                        buckets["gaussian_blur"] += 1
                    elif "_motion_blur_" in stem:
                        buckets["motion_blur"] += 1
                    elif "_unsharp_mask_" in stem:
                        buckets["unsharp_mask"] += 1
                    else:
                        buckets["normal"] += 1
        augmented_v2_buckets[cls] = buckets

    cleaning_v2_data = {
        "version": "geo_v2",
        "config": {
            "input_dir": "data\\cleaning\\splits\\train",
            "output_dir": "data\\augmented_v2",
            "splits_dir": "data\\cleaning\\splits",
            "target_per_class": None,
            "seed": 42,
            "bucket_ratios": {
                "normal": 0.7,
                "gaussian_blur": 0.1,
                "motion_blur": 0.1,
                "unsharp_mask": 0.1
            }
        },
        "stats": augmented_v2_buckets
    }
    # Add validation/test split totals under stats matching geo_v2 structure
    cleaning_v2_data["stats"]["__copied_valid_unseen"] = {"total": split_totals["valid_unseen"]}
    cleaning_v2_data["stats"]["__copied_valid_traincopy"] = {"total": split_totals["valid_traincopy"]}
    cleaning_v2_data["stats"]["__copied_test"] = {"total": split_totals["test"]}
    
    cleaning_v2_counts_path = OUTPUTS_DIR / "data_cleaning_v2_counts.json"
    with open(cleaning_v2_counts_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_v2_data, f, indent=2)
    print(f"Created {cleaning_v2_counts_path}")
    print("\nAll count files generated successfully!")

if __name__ == "__main__":
    generate_counts()
