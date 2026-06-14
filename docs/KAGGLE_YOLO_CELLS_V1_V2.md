# YOLOV8N-CLS: AUGMENTED V1 VA V2 DATASET CLEANING

Tai lieu nay thay the notebook ResNet-50 bang hai thi nghiem YOLO co cung cau hinh. Tien xu ly va tang cuong anh duoc chay local; Kaggle chi huan luyen, danh gia va xuat ket qua.

## 1. Thiet ke thi nghiem

Hai thi nghiem:

| Thi nghiem | Kaggle Dataset | Thu muc anh |
| --- | --- | --- |
| YOLO V1 Cleaning | `Vehicle-YOLO-AugmentedV1-DatasetCleaning` | `augmented/` |
| YOLO V2 Cleaning | `Vehicle-YOLO-AugmentedV2-DatasetCleaning` | `augmented_v2/` |

Quy tac bat buoc:

- V1 va V2 phai duoc tao tu cung `data/splits`, seed 42.
- `train` duoc dung de cap nhat trong so.
- `valid_unseen` duoc dung cho early stopping va chon `best.pt`.
- `test` chi duoc danh gia sau khi da chon checkpoint.
- `valid_traincopy` trung nguon voi train, chi la tap phu va khong tham gia chon model.
- Hai notebook su dung cung kien truc, seed va sieu tham so.
- Khong so sanh V1 va V2 neu validation/test cua hai goi khong giong nhau.

Luu y: `augmented_v2/train/train` la cau truc dung. `train` dau tien la ten split, `train` thu hai la lop tau hoa.

## 2. Tien xu ly local

Chay tai thu muc goc project `vehicle-type-recognition`.

### 2.1. Tao split va Augmented V1

Phai dung dung cu phap cua terminal dang mo. Khong dung dau backtick cua PowerShell trong Git Bash hoac CMD.

**Git Bash:**

```bash
python src/data_prep.py --all \
  --raw_dir data/raw-cleaning/raw \
  --splits_dir data/cleaning/splits \
  --balanced_dir data/cleaning/balanced \
  --augmented_dir data/augmented \
  --seed 42 \
  --stats_json outputs/data_cleaning_v1_counts.json
```

**PowerShell:**

```powershell
python src/data_prep.py --all `
  --raw_dir data/raw-cleaning/raw `
  --splits_dir data/cleaning/splits `
  --balanced_dir data/cleaning/balanced `
  --augmented_dir data/augmented `
  --seed 42 `
  --stats_json outputs/data_cleaning_v1_counts.json
```

**CMD:**

```bat
python src\data_prep.py --all ^
  --raw_dir data\raw-cleaning\raw ^
  --splits_dir data\cleaning\splits ^
  --balanced_dir data\cleaning\balanced ^
  --augmented_dir data\augmented ^
  --seed 42 ^
  --stats_json outputs\data_cleaning_v1_counts.json
```

Lenh mot dong, dung duoc trong ca ba terminal:

```text
python src/data_prep.py --all --raw_dir data/raw-cleaning/raw --splits_dir data/cleaning/splits --balanced_dir data/cleaning/balanced --augmented_dir data/augmented --seed 42 --stats_json outputs/data_cleaning_v1_counts.json
```

Lenh tren thuc hien:

1. Chia tung lop thanh train 85%, `valid_unseen` 5% va test 10%.
2. Tao `valid_traincopy` khoang 5% tu train de kiem tra phu.
3. Resize giu nguyen ti le va zero-pad den 224x224.
4. Tao V1 chi tren train, theo ty le 70% normal, 10% rain, 10% sun, 10% night.
5. Sao chep nguyen trang `valid_unseen`, `valid_traincopy`, test vao `data/augmented`.

Bao cao local da tao xac nhan `data/raw-cleaning/raw` co 35.347 anh va phan chia nhu sau:

| Tap | So anh du kien |
| --- | ---: |
| Train nguon | 30.045 |
| Valid unseen | 1.767 |
| Valid traincopy | 1.767 |
| Test | 3.535 |

Lop train lon nhat la `boat` voi 7.390 anh, do do quota tu dong cua ca V1 va V2 la 7.390 anh moi lop, tong train sau augmentation la 73.900 anh. Moi lop gom 5.173 normal va 739 anh cho moi nhom trong ba nhom con lai. Cac gia tri nay da duoc ghi trong `outputs/data_cleaning_v1_counts.json` va `outputs/data_cleaning_v2_counts.json`.

### 2.2. Tao Augmented V2 tu chinh split V1

Khong chay lai buoc split. V2 phai doc `data/cleaning/balanced` va `data/cleaning/splits` vua tao.

**Git Bash:**

```bash
python src/augment_offline_v2.py \
  --input_dir data/cleaning/balanced \
  --splits_dir data/cleaning/splits \
  --output_dir data/augmented_v2 \
  --seed 42 \
  --stats_json outputs/data_cleaning_v2_counts.json
```

**PowerShell:**

```powershell
python src/augment_offline_v2.py `
  --input_dir data/cleaning/balanced `
  --splits_dir data/cleaning/splits `
  --output_dir data/augmented_v2 `
  --seed 42 `
  --stats_json outputs/data_cleaning_v2_counts.json
```

Lenh mot dong:

```text
python src/augment_offline_v2.py --input_dir data/cleaning/balanced --splits_dir data/cleaning/splits --output_dir data/augmented_v2 --seed 42 --stats_json outputs/data_cleaning_v2_counts.json
```

V2 giu 70% normal va thay ba nhom 10% bang:

- Gaussian blur: kernel 3 hoac 5, sigma 0.4-1.4.
- Motion blur: kernel 3, 5 hoac 7, goc 0-180 do.
- Unsharp mask: kernel 3 hoac 5, sigma 0.5-1.2, amount 0.3-0.8, threshold 0-5.

### 2.3. Kiem tra va dong goi hai Kaggle Dataset

**Git Bash:**

```bash
python src/package_yolo_cleaning_datasets.py \
  --v1_dir data/augmented \
  --v2_dir data/augmented_v2 \
  --src_dir src \
  --output_dir kaggle_datasets
```

**PowerShell:**

```powershell
python src/package_yolo_cleaning_datasets.py `
  --v1_dir data/augmented `
  --v2_dir data/augmented_v2 `
  --src_dir src `
  --output_dir kaggle_datasets
```

Lenh mot dong:

```text
python src/package_yolo_cleaning_datasets.py --v1_dir data/augmented --v2_dir data/augmented_v2 --src_dir src --output_dir kaggle_datasets
```

Script se:

- xac nhan moi split co dung 10 lop;
- dem anh theo lop;
- bam SHA-256 de dam bao V1/V2 co cung `valid_unseen`, `valid_traincopy`, test;
- tao hai file:
  - `kaggle_datasets/Vehicle-YOLO-AugmentedV1-DatasetCleaning.zip`
  - `kaggle_datasets/Vehicle-YOLO-AugmentedV2-DatasetCleaning.zip`

Upload moi ZIP thanh mot Kaggle Dataset rieng. Khong giai nen roi zip them mot cap thu muc, vi notebook can tim thay truc tiep `src` va `augmented` hoac `augmented_v2`.

## 3. Cau hinh YOLO dung chung

| Thiet lap | Gia tri | Ly do |
| --- | --- | --- |
| Kien truc | `yolov8n-cls.pt` | Nhe, nhanh, phu hop T4 va demo |
| Chien luoc | Transfer learning | Tan dung dac trung ImageNet, khong khoi tao ngau nhien |
| Max epoch | 30 | Dong bo voi cac thi nghiem nhom |
| Patience | 10 | Dung som khi validation khong tang |
| Batch | 128 | Phu hop YOLOv8n-cls 224x224 tren T4 16 GB |
| Input | 224x224 | Dong bo pipeline local va ResNet/ViT |
| Optimizer | `auto` | Ultralytics chon optimizer theo mo hinh |
| AMP | Bat | Giam VRAM va tang toc tren GPU |
| Seed | 42 | Lap lai va so sanh cong bang |
| Deterministic | Bat | Han che sai khac ngau nhien |
| Workers | 2 | On dinh trong Kaggle |
| RandomResizedCrop | Tat | Khong cat mat noi dung da zero-pad local |
| RandAugment | Tat | Khong tron voi V1/V2 offline |
| Random erasing | Tat | Khong tao bien so ngoai thiet ke |
| Online flip | Tat | Geometric augmentation da duoc tao local |

## 4. Notebook YOLO V1 Cleaning

Gan Kaggle Dataset `Vehicle-YOLO-AugmentedV1-DatasetCleaning` vao notebook va chon GPU T4.

### Cell V1.1 - Tim source va dataset

```python
from pathlib import Path

EXPECTED_CLASSES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]

src_candidates = [
    path for path in Path("/kaggle/input").rglob("src")
    if (path / "train_yolo.py").is_file()
]
data_candidates = [
    path for path in Path("/kaggle/input").rglob("augmented")
    if all((path / split).is_dir() for split in ("train", "valid_unseen", "test"))
]

assert src_candidates, "Cannot find src/train_yolo.py"
assert data_candidates, "Cannot find augmented dataset"

SRC_DIR = src_candidates[0]
DATA_DIR = data_candidates[0]
PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
VARIANT = "v1"
RUN_NAME = "yolo_aug_v1_cleaning"

print("SRC :", SRC_DIR)
print("DATA:", DATA_DIR)
```

### Cell V1.2 - Kiem tra cau truc va so luong

```python
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

dataset_counts = {}
for split in ("train", "valid_unseen", "valid_traincopy", "test"):
    split_dir = DATA_DIR / split
    assert split_dir.is_dir(), f"Missing split: {split_dir}"
    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    assert actual_classes == EXPECTED_CLASSES, (split, actual_classes)

    per_class = {}
    for class_name in EXPECTED_CLASSES:
        per_class[class_name] = sum(
            1 for path in (split_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        assert per_class[class_name] > 0, f"Empty class: {split}/{class_name}"
    dataset_counts[split] = per_class
    print(split, "total =", sum(per_class.values()), per_class)
```

### Cell V1.3 - Copy source code, khong copy anh

```python
import shutil

shutil.rmtree(PROJECT_DIR, ignore_errors=True)
(PROJECT_DIR / "src").parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC_DIR, PROJECT_DIR / "src")
print("Project ready:", PROJECT_DIR)
```

Anh duoc doc truc tiep tu Kaggle Input thong qua dataset adapter. Cach nay tiet kiem thoi gian va dung luong working.

### Cell V1.4 - Cai Ultralytics

```python
import importlib.metadata
import subprocess
import sys

PINNED_ULTRALYTICS = "8.4.66"
try:
    current = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    current = None

if current != PINNED_ULTRALYTICS:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={PINNED_ULTRALYTICS}"],
        check=True,
    )

print("Ultralytics:", importlib.metadata.version("ultralytics"))
```

Khong cai lai Torch, NumPy, Pandas hoac RAPIDS.

### Cell V1.5 - Train moi

```python
import subprocess
import sys

train_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "train_yolo.py"),
    "--data_dir", str(DATA_DIR),
    "--dataset_id", "raw_cleaning",
    "--model", "yolov8n-cls.pt",
    "--epochs", "30",
    "--patience", "10",
    "--batch", "128",
    "--imgsz", "224",
    "--workers", "2",
    "--seed", "42",
    "--device", "0",
    "--name", RUN_NAME,
    "--dataset_variant", VARIANT,
    "--reset",
]

subprocess.run(train_command, cwd=PROJECT_DIR, check=True)
```

### Cell V1.6 - Ve loss va validation accuracy

```python
import json
import matplotlib.pyplot as plt

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo"
history = json.loads((OUTPUT_DIR / "history_yolo.json").read_text(encoding="utf-8"))

epochs = [row["epoch"] for row in history]
train_loss = [row["train_loss"] for row in history]
valid_loss = [row["valid_unseen_loss"] for row in history]
valid_acc = [row["valid_unseen_acc"] for row in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, valid_loss, marker="o", label="Valid unseen loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("YOLOv8n-cls Loss Curves - Augmented V1 Cleaning")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_loss.png", dpi=200)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, valid_acc, marker="o", color="#2ca02c")
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy")
plt.title("YOLOv8n-cls Validation Accuracy - Augmented V1 Cleaning")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_valid_accuracy.png", dpi=200)
plt.show()
```

### Cell V1.7 - Danh gia best.pt tren validation va test

```python
evaluate_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "evaluate_yolo.py"),
    "--model_path", "models/yolo/yolo_cls_best.pt",
    "--data_dir", str(DATA_DIR),
    "--dataset_id", "raw_cleaning",
    "--dataset_variant", VARIANT,
    "--output", "outputs/yolo/evaluation_yolo_cls_best.json",
    "--metrics_output", "outputs/yolo/metrics_yolo.json",
    "--imgsz", "224",
    "--batch", "128",
    "--device", "0",
]

subprocess.run(evaluate_command, cwd=PROJECT_DIR, check=True)
```

`valid_traincopy` mac dinh khong duoc danh gia. Neu can phu luc, them `--include_valid_traincopy`, nhung khong dung chi so nay de chon model.

### Cell V1.8 - In bang chi so

```python
evaluation = json.loads(
    (OUTPUT_DIR / "evaluation_yolo_cls_best.json").read_text(encoding="utf-8")
)
test = evaluation["test"]
report = test["classification_report"]

print("Test accuracy:", test["accuracy"])
print("Macro F1:", report["macro avg"]["f1-score"])
print("Weighted F1:", report["weighted avg"]["f1-score"])
print()

for class_name in evaluation["class_names"]:
    row = report[class_name]
    print(
        f"{class_name:12s} "
        f"P={row['precision']:.4f} "
        f"R={row['recall']:.4f} "
        f"F1={row['f1-score']:.4f} "
        f"support={int(row['support'])}"
    )
```

### Cell V1.9 - Confusion matrix va chi so theo lop

```python
import pandas as pd
import seaborn as sns

class_names = evaluation["class_names"]
cm = evaluation["test"]["confusion_matrix"]
metrics_df = pd.read_csv(OUTPUT_DIR / "yolo_class_metrics.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("YOLOv8n-cls Confusion Matrix - Test - Augmented V1 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_confusion_matrix.png", dpi=200)
plt.show()

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="f1", color="#4C78A8")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.title("YOLOv8n-cls F1-score per Class - Augmented V1 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_f1_per_class.png", dpi=200)
plt.show()

long_df = metrics_df.melt(
    id_vars="class",
    value_vars=["precision", "recall", "f1"],
    var_name="metric",
    value_name="score",
)
plt.figure(figsize=(12, 5))
sns.barplot(data=long_df, x="class", y="score", hue="metric")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.title("YOLOv8n-cls Precision / Recall / F1 - Augmented V1 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_precision_recall_f1_per_class.png", dpi=200)
plt.show()
```

### Cell V1.10 - Phan bo test va nen output

```python
import zipfile

metrics_df["ratio"] = metrics_df["support"] / metrics_df["support"].sum()

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="support", color="#72B7B2")
plt.xticks(rotation=35, ha="right")
plt.title("Test Samples per Class - Augmented V1 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_test_support_per_class.png", dpi=200)
plt.show()

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="ratio", color="#F58518")
plt.xticks(rotation=35, ha="right")
plt.title("Test Class Ratio - Augmented V1 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_test_class_ratio.png", dpi=200)
plt.show()

ZIP_PATH = Path("/kaggle/working/yolo_aug_v1_cleaning_outputs.zip")
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for source in [PROJECT_DIR / "models" / "yolo", OUTPUT_DIR]:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(PROJECT_DIR))

print("Created:", ZIP_PATH)
```

## 5. Notebook YOLO V2 Cleaning

Toan bo quy trinh V2, gom 14 cell doc lap tu tim dataset den nen ZIP, duoc trinh bay tai:

`docs/KAGGLE_YOLO_V2_CLEANING_FULL.md`

Tao notebook moi, chi gan Kaggle Dataset `Vehicle-YOLO-AugmentedV2-DatasetCleaning`, sau do chay lan luot Cell 1 den Cell 14 trong tai lieu tren. Khong chay V2 tiep tren working directory cua V1.

## 6. Dieu kien chap nhan ket qua

Chi dung ket qua de so sanh khi:

1. Hai notebook in dung 10 lop theo cung thu tu.
2. `valid_unseen` va test co cung so mau, support tung lop giong nhau.
3. Hai notebook dung cung Ultralytics, model, epoch, patience, batch, seed va transforms.
4. So mau test trong `evaluation_yolo_cls_best.json` bang dung tong test duoc in khi kiem tra dataset; khong hard-code theo dataset cu.
5. ZIP co `models/yolo/yolo_cls_best.pt`, history, manifest, evaluation, CSV va cac bieu do.

Khi viet bao cao, so sanh toi thieu:

- best validation Top-1 Accuracy;
- test Accuracy, Macro F1 va Weighted F1;
- F1 tung lop, dac biet taxi, minibus, car va truck;
- cac cap nham lan lon nhat trong confusion matrix;
- so epoch thuc te va dau hieu overfitting qua train/validation loss;
- tac dong cua nhom bien doi moi truong V1 so voi blur/sharpen V2.
