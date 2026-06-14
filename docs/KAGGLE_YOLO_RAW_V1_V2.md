# YOLOV8N-CLS - RAW ORIGINAL V1 VA V2 TREN KAGGLE

Quy trinh nay danh cho bo `data/raw` chua loc nhieu thu cong. Tien xu ly va
augmentation da duoc thuc hien local. Kaggle chi huan luyen, danh gia va xuat
artifact; khong chia lai du lieu va khong augmentation online.

## 1. Hai thi nghiem doc lap

| Thi nghiem | Kaggle Dataset | Train augmentation |
| --- | --- | --- |
| Raw Original V1 | `Vehicle-YOLO-AugmentedV1-RawOriginal` | 70% Normal + 10% Rain + 10% Sun + 10% Night |
| Raw Original V2 | `Vehicle-YOLO-AugmentedV2-RawOriginal` | 70% Normal + 10% Gaussian + 10% Motion + 10% Unsharp |

Moi phien ban co:

- train: 75.620 anh, 7.562 anh cho moi lop;
- valid unseen: 1.800 anh;
- valid train-copy: 1.800 anh, chi dung tham khao;
- test: 3.601 anh;
- 10 lop theo thu tu alphabet;
- cung split va seed 42 de so sanh V1/V2 truc tiep.

## 2. Dong goi tai may local

Chay mot dong tai thu muc goc project:

```bash
python src/package_yolo_raw_datasets.py --v1_dir data/augmented_raw_v1 --v2_dir data/augmented_raw_v2 --src_dir src --output_dir kaggle_datasets
```

Script kiem tra cau truc, so anh va SHA-256 cua ba tap danh gia truoc khi tao:

```text
kaggle_datasets/Vehicle-YOLO-AugmentedV1-RawOriginal.zip
kaggle_datasets/Vehicle-YOLO-AugmentedV2-RawOriginal.zip
```

Neu chi muon audit truoc khi tao hai ZIP lon:

```bash
python src/package_yolo_raw_datasets.py --validate_only
```

Upload moi ZIP thanh mot Kaggle Dataset rieng. Moi notebook chi gan dung mot
dataset de khong tim nham V1/V2.

## 3. Cau hinh notebook

Tao hai notebook tu cung bo cell ben duoi. Tai Cell 1 chi thay `VARIANT`:

```python
# Notebook V1
VARIANT = "v1"
```

hoac:

```python
# Notebook V2
VARIANT = "v2"
```

Chon `GPU T4 x1`. Bat Internet neu Kaggle chua cache `yolov8n-cls.pt`.

## Cell 1 - Cau hinh va tim dung tai nguyen

```python
from pathlib import Path
import json
import shutil
import zipfile

VARIANT = "v1"  # Doi thanh "v2" trong notebook V2
assert VARIANT in {"v1", "v2"}

DATASET_ID = "raw_original"
DATA_ROOT_NAME = "augmented" if VARIANT == "v1" else "augmented_v2"
RUN_NAME = f"yolo_raw_original_{VARIANT}"
INPUT_ROOT = Path("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
EXTRACT_DIR = Path(f"/kaggle/working/raw_original_{VARIANT}_dataset")

EXPECTED_CLASSES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def find_resources(root):
    src_list = [
        path for path in root.rglob("src")
        if (path / "train_yolo.py").is_file()
        and (path / "evaluate_yolo.py").is_file()
    ]
    data_list = [
        path for path in root.rglob(DATA_ROOT_NAME)
        if all((path / split).is_dir() for split in ("train", "valid_unseen", "test"))
    ]
    return src_list, data_list

src_candidates, data_candidates = find_resources(INPUT_ROOT)
if not src_candidates or not data_candidates:
    zip_candidates = list(INPUT_ROOT.rglob("*.zip"))
    assert zip_candidates, "Khong tim thay source/dataset hoac ZIP trong Kaggle Input"
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    EXTRACT_DIR.mkdir(parents=True)
    for zip_path in zip_candidates:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(EXTRACT_DIR)
    src_candidates, data_candidates = find_resources(EXTRACT_DIR)

assert len(src_candidates) == 1, f"Can dung 1 src, tim thay: {src_candidates}"
assert len(data_candidates) == 1, f"Can dung 1 {DATA_ROOT_NAME}, tim thay: {data_candidates}"

SRC_DIR = src_candidates[0]
DATA_DIR = data_candidates[0]
print("Variant:", VARIANT)
print("Source :", SRC_DIR)
print("Dataset:", DATA_DIR)
```

## Cell 2 - Audit du lieu that va manifest

```python
dataset_counts = {}
for split in ("train", "valid_unseen", "valid_traincopy", "test"):
    split_dir = DATA_DIR / split
    assert split_dir.is_dir(), f"Thieu split: {split_dir}"
    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    assert actual_classes == EXPECTED_CLASSES, f"Sai class mapping tai {split}: {actual_classes}"

    per_class = {}
    for class_name in EXPECTED_CLASSES:
        per_class[class_name] = sum(
            1 for path in (split_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        assert per_class[class_name] > 0, f"Lop rong: {split}/{class_name}"
    dataset_counts[split] = per_class
    print(f"{split:16s} total={sum(per_class.values()):,}", per_class)

assert sum(dataset_counts["train"].values()) == 75620
assert set(dataset_counts["train"].values()) == {7562}
assert sum(dataset_counts["valid_unseen"].values()) == 1800
assert sum(dataset_counts["valid_traincopy"].values()) == 1800
assert sum(dataset_counts["test"].values()) == 3601

manifest_candidates = list(DATA_DIR.parent.rglob("dataset_manifest.json"))
assert manifest_candidates, "Thieu dataset_manifest.json"
manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
assert manifest["dataset_id"] == DATASET_ID
assert manifest["variant"] == VARIANT
assert manifest["kaggle_data_root"] == DATA_ROOT_NAME
print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])
```

## Cell 3 - Tao project working

```python
shutil.rmtree(PROJECT_DIR, ignore_errors=True)
PROJECT_DIR.mkdir(parents=True)
shutil.copytree(SRC_DIR, PROJECT_DIR / "src")

train_code = (PROJECT_DIR / "src" / "train_yolo.py").read_text(encoding="utf-8")
assert "--dataset_variant" in train_code
assert "history_yolo.json" in train_code
print("Project ready:", PROJECT_DIR)
```

## Cell 4 - Cai va kiem tra moi truong

```python
import importlib.metadata
import subprocess
import sys

PINNED_ULTRALYTICS = "8.4.66"
try:
    current_version = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    current_version = None

if current_version != PINNED_ULTRALYTICS:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         f"ultralytics=={PINNED_ULTRALYTICS}"],
        check=True,
    )

import torch
import ultralytics
assert torch.cuda.is_available(), "Hay bat GPU Accelerator trong Kaggle"
print("Ultralytics:", ultralytics.__version__)
print("Torch       :", torch.__version__)
print("GPU         :", torch.cuda.get_device_name(0))
```

## Cell 5 - Kiem tra adapter truoc khi train

```python
prepare_command = [
    sys.executable, str(PROJECT_DIR / "src" / "train_yolo.py"),
    "--data_dir", str(DATA_DIR),
    "--dataset_id", DATASET_ID,
    "--dataset_variant", VARIANT,
    "--prepare_only",
    "--reset",
]
subprocess.run(prepare_command, cwd=PROJECT_DIR, check=True)

audit_path = PROJECT_DIR / "outputs" / "yolo" / "dataset_audit.json"
assert audit_path.is_file()
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert audit["splits"]["train"]["total"] == 75620
assert audit["splits"]["valid_unseen"]["total"] == 1800
assert audit["splits"]["test"]["total"] == 3601
print(json.dumps(audit, ensure_ascii=False, indent=2))
```

## Cell 6 - Train YOLOv8n-cls

```python
train_command = [
    sys.executable, str(PROJECT_DIR / "src" / "train_yolo.py"),
    "--data_dir", str(DATA_DIR),
    "--dataset_id", DATASET_ID,
    "--dataset_variant", VARIANT,
    "--model", "yolov8n-cls.pt",
    "--epochs", "30",
    "--patience", "10",
    "--batch", "128",
    "--imgsz", "224",
    "--workers", "2",
    "--seed", "42",
    "--device", "0",
    "--name", RUN_NAME,
    "--reset",
]
subprocess.run(train_command, cwd=PROJECT_DIR, check=True)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo"
MODEL_PATH = PROJECT_DIR / "models" / "yolo" / "yolo_cls_best.pt"
for path in (
    MODEL_PATH,
    OUTPUT_DIR / "history_yolo.json",
    OUTPUT_DIR / "metrics_yolo.json",
    OUTPUT_DIR / "experiment_manifest.json",
    OUTPUT_DIR / "dataset_audit.json",
    OUTPUT_DIR / "results.csv",
):
    assert path.is_file(), f"Training xong nhung thieu: {path}"
    print(path.relative_to(PROJECT_DIR), path.stat().st_size, "bytes")
```

Khong them `--project`. Script hien tai tu luu run, history, metrics va best
checkpoint vao dung thu muc chuan.

## Cell 7 - Doc history va ve learning curves

```python
import matplotlib.pyplot as plt

history = json.loads((OUTPUT_DIR / "history_yolo.json").read_text(encoding="utf-8"))
assert history, "history_yolo.json rong"

epochs = [int(row["epoch"]) for row in history]
train_loss = [float(row["train_loss"]) for row in history]
valid_loss = [float(row["valid_unseen_loss"]) for row in history]
valid_acc = [float(row["valid_unseen_acc"]) for row in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, valid_loss, marker="o", label="Validation unseen loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title(f"YOLOv8n-cls Loss Curves - Raw Original {VARIANT.upper()}")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_loss.png", dpi=200)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, valid_acc, marker="o", color="#2ca02c")
plt.xlabel("Epoch"); plt.ylabel("Top-1 Accuracy"); plt.ylim(0, 1)
plt.title(f"YOLOv8n-cls Validation Accuracy - Raw Original {VARIANT.upper()}")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_valid_accuracy.png", dpi=200)
plt.show()
```

## Cell 8 - Danh gia best checkpoint

```python
evaluate_command = [
    sys.executable, str(PROJECT_DIR / "src" / "evaluate_yolo.py"),
    "--model_path", str(MODEL_PATH),
    "--data_dir", str(DATA_DIR),
    "--dataset_id", DATASET_ID,
    "--dataset_variant", VARIANT,
    "--output", str(OUTPUT_DIR / "evaluation_yolo_cls_best.json"),
    "--metrics_output", str(OUTPUT_DIR / "metrics_yolo.json"),
    "--imgsz", "224",
    "--batch", "128",
    "--device", "0",
]
subprocess.run(evaluate_command, cwd=PROJECT_DIR, check=True)

evaluation_path = OUTPUT_DIR / "evaluation_yolo_cls_best.json"
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
assert evaluation["dataset_id"] == DATASET_ID
assert evaluation["dataset_variant"] == VARIANT
assert evaluation["valid_unseen"]["samples"] == 1800
assert evaluation["test"]["samples"] == 3601
```

## Cell 9 - Kiem chung metric va xuat bang tung lop

```python
import pandas as pd

test = evaluation["test"]
report = test["classification_report"]
matrix = test["confusion_matrix"]
support_total = sum(int(report[name]["support"]) for name in EXPECTED_CLASSES)
matrix_total = sum(sum(int(value) for value in row) for row in matrix)
correct = sum(matrix[index][index] for index in range(len(matrix)))

assert support_total == test["samples"] == 3601
assert matrix_total == test["samples"]
assert abs(correct / matrix_total - float(test["accuracy"])) < 1e-6

rows = [{
    "class": name,
    "precision": float(report[name]["precision"]),
    "recall": float(report[name]["recall"]),
    "f1": float(report[name]["f1-score"]),
    "support": int(report[name]["support"]),
} for name in EXPECTED_CLASSES]
metrics_df = pd.DataFrame(rows)
display(metrics_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}))

print("Test accuracy:", f"{test['accuracy']:.6f}")
print("Macro F1     :", f"{report['macro avg']['f1-score']:.6f}")
print("Weighted F1  :", f"{report['weighted avg']['f1-score']:.6f}")
```

## Cell 10 - Confusion matrix va F1 theo lop

```python
import seaborn as sns

plt.figure(figsize=(11, 9))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"YOLOv8n-cls Confusion Matrix - Test - Raw Original {VARIANT.upper()}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_confusion_matrix.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_df, x="class", y="f1", color="#2ca02c")
plt.ylim(0, 1); plt.xticks(rotation=35, ha="right")
plt.xlabel("Class"); plt.ylabel("F1-score")
plt.title(f"YOLOv8n-cls F1 per Class - Raw Original {VARIANT.upper()}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_f1_per_class.png", dpi=200)
plt.show()
```

## Cell 11 - Kiem tra va nen output

```python
required_outputs = [
    MODEL_PATH,
    OUTPUT_DIR / "dataset_audit.json",
    OUTPUT_DIR / "experiment_manifest.json",
    OUTPUT_DIR / "history_yolo.json",
    OUTPUT_DIR / "metrics_yolo.json",
    OUTPUT_DIR / "evaluation_yolo_cls_best.json",
    OUTPUT_DIR / "results.csv",
    OUTPUT_DIR / "yolo_class_metrics.csv",
    OUTPUT_DIR / "yolo_loss.png",
    OUTPUT_DIR / "yolo_valid_accuracy.png",
    OUTPUT_DIR / "yolo_confusion_matrix.png",
    OUTPUT_DIR / "yolo_f1_per_class.png",
]
missing = [str(path) for path in required_outputs if not path.is_file()]
assert not missing, "Thieu output:\n" + "\n".join(missing)

ZIP_PATH = Path(f"/kaggle/working/yolo_raw_original_{VARIANT}_outputs.zip")
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for source in (PROJECT_DIR / "models" / "yolo", OUTPUT_DIR):
        for file_path in sorted(source.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(PROJECT_DIR))

print("Created:", ZIP_PATH)
print("Size   :", f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB")
```

## 4. Nguyen tac so sanh V1 va V2

- Dung cung model `yolov8n-cls.pt`, epoch, patience, batch, image size va seed.
- Chon checkpoint chi bang `valid_unseen` Top-1 Accuracy.
- Chi danh gia test sau khi da chon xong checkpoint.
- So sanh Accuracy, Macro F1, Weighted F1 va F1 tung lop.
- Tap trung cac nham lan `car/taxi/minibus/truck` thay vi chi nhin accuracy tong.
- Khong suy dien test loss: evaluation bang predict nen test loss phai la `null`.
- `valid_traincopy` trung nguon voi train, khong duoc dung nhu validation doc lap.
