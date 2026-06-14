# YOLOV8N-CLS - AUGMENTED V2 DATASET CLEANING

Tai lieu nay trinh bay day du quy trinh cua rieng thi nghiem:

`Vehicle-YOLO-AugmentedV2-DatasetCleaning`

Tien xu ly duoc thuc hien local. Kaggle chi nhan `src` va `augmented_v2` de huan luyen, danh gia va xuat ket qua.

---

## 1. Muc tieu cua V2 Cleaning

V2 duoc tao de danh gia tac dong cua ba phep bien doi tap trung vao chat luong va do net anh:

| Nhom anh | Ti le | Thiet lap |
| --- | ---: | --- |
| Normal | 70% | Base pipeline, khong blur/sharpen |
| Gaussian blur | 10% | Kernel 3 hoac 5; sigma 0.4-1.4 |
| Motion blur | 10% | Kernel 3, 5 hoac 7; goc 0-180 do |
| Unsharp mask | 10% | Kernel 3 hoac 5; sigma 0.5-1.2; amount 0.3-0.8; threshold 0-5 |

V2 khong tao split moi. No bat buoc su dung lai `splits` da tao khi chay V1 de bao dam:

- train V1 va V2 co cung nguon anh;
- `valid_unseen` giong nhau;
- `valid_traincopy` giong nhau;
- test giong nhau;
- ket qua V1/V2 co the so sanh truc tiep.

## 2. Cau truc du lieu mong doi

```text
augmented_v2/
|-- train/
|   |-- bicycle/
|   |-- boat/
|   |-- bus/
|   |-- car/
|   |-- helicopter/
|   |-- minibus/
|   |-- motorcycle/
|   |-- taxi/
|   |-- train/
|   `-- truck/
|-- valid_unseen/
|-- valid_traincopy/
`-- test/
```

`augmented_v2/train/train` la cau truc hop le: `train` thu nhat la split huan luyen, `train` thu hai la lop tau hoa.

## 3. Tao V2 Cleaning tai may local

Mo terminal tai:

```text
D:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition
```

### 3.1. Tao split goc va V1 truoc

Buoc nay bat buoc vi V2 can dung lai `data/cleaning/splits` va `data/cleaning/balanced`.

**Git Bash, mot dong:**

```bash
python src/data_prep.py --all --raw_dir data/raw-cleaning/raw --splits_dir data/cleaning/splits --balanced_dir data/cleaning/balanced --augmented_dir data/augmented --seed 42 --stats_json outputs/data_cleaning_v1_counts.json
```

### 3.2. Tao Augmented V2

**Git Bash, mot dong:**

```bash
python src/augment_offline_v2.py --input_dir data/cleaning/balanced --splits_dir data/cleaning/splits --output_dir data/augmented_v2 --seed 42 --stats_json outputs/data_cleaning_v2_counts.json
```

Khong doi `seed`, khong thay `splits_dir`, va khong chay lai `data_prep` chi cho V2.

### 3.3. Kiem tra va dong goi

```bash
python src/package_yolo_cleaning_datasets.py --v1_dir data/augmented --v2_dir data/augmented_v2 --src_dir src --output_dir kaggle_datasets
```

Script se bam SHA-256 cho ba tap danh gia va dung ngay neu V1/V2 khong giong nhau.

File can upload len Kaggle:

```text
kaggle_datasets/Vehicle-YOLO-AugmentedV2-DatasetCleaning.zip
```

Upload file thanh Kaggle Dataset co ten:

```text
Vehicle-YOLO-AugmentedV2-DatasetCleaning
```

## 4. Thiet lap Kaggle Notebook

1. Tao notebook moi cho rieng V2.
2. Gan dataset `Vehicle-YOLO-AugmentedV2-DatasetCleaning`.
3. Chon accelerator `GPU T4 x1`.
4. Bat Internet de tai `yolov8n-cls.pt` neu checkpoint chua duoc cache.
5. Khong gan dong thoi dataset V1 de tranh cell tim nham `src`.

## 5. Cau hinh huan luyen

| Tham so | Gia tri | Giai thich |
| --- | --- | --- |
| Model | `yolov8n-cls.pt` | YOLO classification Nano pretrained |
| Strategy | Transfer learning | Bat dau tu trong so ImageNet pretrained |
| Epoch toi da | 30 | Dong bo thi nghiem V1 va cac model khac |
| Patience | 10 | Early stopping theo validation |
| Batch | 128 | Phu hop T4 voi anh 224x224 |
| Image size | 224 | Trung voi base pipeline local |
| Optimizer | Auto | Ultralytics tu chon |
| AMP | Bat | Giam VRAM va tang toc |
| Seed | 42 | Lap lai va so sanh cong bang |
| Deterministic | Bat | Giam sai khac ngau nhien |
| Workers | 2 | On dinh tren Kaggle |
| RandomResizedCrop | Tat | Khong cat mat noi dung anh da zero-pad |
| RandAugment | Tat | Khong tron augmentation online vao V2 |
| Random erasing | Tat | Giu dung thiet ke preprocessing local |
| Online flip | Tat | Hinh hoc da duoc xu ly offline |

Checkpoint chi duoc chon bang `valid_unseen`. Test chi duoc mo sau khi huan luyen ket thuc.

---

# CAC CELL KAGGLE V2 DAY DU

Chay lan luot tu Cell 1 den Cell 14.

## Cell 1 - Tim source code va Augmented V2

Cell nay ho tro ca hai truong hop: Kaggle da giai nen noi dung dataset, hoac file ZIP van nam trong `/kaggle/input`.

```python
from pathlib import Path
import shutil
import zipfile

EXPECTED_CLASSES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]

INPUT_ROOT = Path("/kaggle/input")
PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
EXTRACT_DIR = Path("/kaggle/working/v2_cleaning_dataset")
VARIANT = "v2"
RUN_NAME = "yolo_aug_v2_cleaning"

def find_resources(root: Path):
    src_list = [
        path for path in root.rglob("src")
        if path.is_dir() and (path / "train_yolo.py").is_file()
    ]
    data_list = [
        path for path in root.rglob("augmented_v2")
        if path.is_dir()
        and all((path / split).is_dir() for split in ("train", "valid_unseen", "test"))
    ]
    return src_list, data_list

src_candidates, data_candidates = find_resources(INPUT_ROOT)

if not src_candidates or not data_candidates:
    zip_candidates = list(INPUT_ROOT.rglob("Vehicle-YOLO-AugmentedV2-DatasetCleaning.zip"))
    if not zip_candidates:
        zip_candidates = [
            path for path in INPUT_ROOT.rglob("*.zip")
            if "augmentedv2" in path.name.lower().replace("-", "")
        ]
    assert zip_candidates, "Cannot find V2 Cleaning files or ZIP in /kaggle/input"

    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_candidates[0]) as archive:
        archive.extractall(EXTRACT_DIR)
    src_candidates, data_candidates = find_resources(EXTRACT_DIR)

assert src_candidates, "Cannot find src/train_yolo.py"
assert data_candidates, "Cannot find augmented_v2/train, valid_unseen and test"

SRC_DIR = src_candidates[0]
DATA_DIR = data_candidates[0]

print("Source code :", SRC_DIR)
print("Dataset V2 :", DATA_DIR)
print("Project     :", PROJECT_DIR)
```

Ket qua dung phai co `DATA_DIR` ket thuc bang `augmented_v2`.

## Cell 2 - Kiem tra cau truc, lop va so luong anh

```python
import json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def count_images(path: Path) -> int:
    return sum(
        1 for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )

dataset_counts = {}
for split in ("train", "valid_unseen", "valid_traincopy", "test"):
    split_dir = DATA_DIR / split
    assert split_dir.is_dir(), f"Missing split: {split_dir}"

    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    assert actual_classes == EXPECTED_CLASSES, (
        f"Class mismatch in {split}", actual_classes
    )

    per_class = {
        class_name: count_images(split_dir / class_name)
        for class_name in EXPECTED_CLASSES
    }
    empty_classes = [name for name, count in per_class.items() if count == 0]
    assert not empty_classes, f"Empty classes in {split}: {empty_classes}"

    dataset_counts[split] = {
        "total": sum(per_class.values()),
        "per_class": per_class,
    }
    print(f"\n{split}: {dataset_counts[split]['total']:,} images")
    print(per_class)

train_counts = dataset_counts["train"]["per_class"]
assert len(set(train_counts.values())) == 1, (
    "Augmented V2 train is expected to be balanced", train_counts
)

print("\nClasses       :", len(EXPECTED_CLASSES))
print("Train/class   :", next(iter(train_counts.values())))
print("Validation    :", dataset_counts["valid_unseen"]["total"])
print("Official test :", dataset_counts["test"]["total"])
```

Bao cao local `outputs/data_cleaning_v2_counts.json` hien tai da xac nhan:

- 10 lop, moi lop train co 7.390 anh;
- moi lop gom 5.173 normal, 739 Gaussian blur, 739 motion blur va 739 unsharp mask;
- tong train V2 la 73.900 anh;
- `valid_unseen` co 1.767 anh;
- `valid_traincopy` co 1.767 anh;
- test co 3.535 anh.

Cell nay van dem lai truc tiep tu Kaggle Dataset de xac nhan qua trinh upload khong lam thieu file.

## Cell 3 - Doc manifest cua dataset

```python
manifest_candidates = list(DATA_DIR.parent.rglob("dataset_manifest.json"))

if manifest_candidates:
    dataset_manifest = json.loads(
        manifest_candidates[0].read_text(encoding="utf-8")
    )
    print(json.dumps(dataset_manifest, indent=2, ensure_ascii=False))
    assert dataset_manifest.get("variant") == "v2"
else:
    dataset_manifest = None
    print("dataset_manifest.json not found; structural checks from Cell 2 passed.")
```

## Cell 4 - Tao project working va copy source code

Khong copy 73.900 anh vao working. Training doc truc tiep tu Kaggle Input thong qua symbolic link.

```python
shutil.rmtree(PROJECT_DIR, ignore_errors=True)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC_DIR, PROJECT_DIR / "src")

required_scripts = [
    PROJECT_DIR / "src" / "train_yolo.py",
    PROJECT_DIR / "src" / "evaluate_yolo.py",
]
missing_scripts = [str(path) for path in required_scripts if not path.is_file()]
assert not missing_scripts, f"Missing scripts: {missing_scripts}"

print("Project ready:", PROJECT_DIR)
```

## Cell 5 - Cai dat va kiem tra moi truong

Chi cai Ultralytics. Khong cai lai Torch, CUDA, NumPy, Pandas hoac RAPIDS.

```python
import importlib.metadata
import subprocess
import sys

PINNED_ULTRALYTICS = "8.4.66"

try:
    installed = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    installed = None

if installed != PINNED_ULTRALYTICS:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"ultralytics=={PINNED_ULTRALYTICS}",
        ],
        check=True,
    )

import torch
import ultralytics

assert torch.cuda.is_available(), "GPU is not enabled. Select GPU T4 in Notebook settings."

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__)
print("Ultralytics :", ultralytics.__version__)
print("CUDA        :", torch.version.cuda)
print("GPU         :", torch.cuda.get_device_name(0))
```

## Cell 6 - Huan luyen YOLOv8n-cls tren V2

Cell nay xoa rieng ket qua YOLO trong project working va train mot thi nghiem moi.

```python
train_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "train_yolo.py"),
    "--data_dir", str(DATA_DIR),
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

print("Running:", " ".join(train_command))
subprocess.run(train_command, cwd=PROJECT_DIR, check=True)
```

Sau cell nay phai co:

```text
models/yolo/yolo_cls_best.pt
outputs/yolo/history_yolo.json
outputs/yolo/metrics_yolo.json
outputs/yolo/experiment_manifest.json
outputs/yolo/dataset_audit.json
outputs/yolo/results.csv
```

## Cell 7 - Doc tom tat training va best checkpoint

```python
OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo"
MODEL_PATH = PROJECT_DIR / "models" / "yolo" / "yolo_cls_best.pt"

history_path = OUTPUT_DIR / "history_yolo.json"
metrics_path = OUTPUT_DIR / "metrics_yolo.json"
experiment_path = OUTPUT_DIR / "experiment_manifest.json"

for path in (history_path, metrics_path, experiment_path, MODEL_PATH):
    assert path.is_file(), f"Missing training artifact: {path}"

history = json.loads(history_path.read_text(encoding="utf-8"))
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
summary = metrics["training_summary"]

print("Completed epochs            :", summary["completed_epochs"])
print("Best checkpoint epoch       :", summary["best_checkpoint_epoch"])
print("Best validation Top-1       :", summary["best_valid_unseen_accuracy"])
print("Validation loss at best.pt  :", summary["valid_loss_at_best_checkpoint"])
print("Minimum validation loss     :", summary["minimum_valid_unseen_loss"])
print("Minimum loss epoch          :", summary["minimum_valid_unseen_loss_epoch"])
```

Best checkpoint epoch va minimum validation loss epoch co the khac nhau. Bao cao can ghi ro hai gia tri, khong gom chung thanh mot “best epoch”.

## Cell 8 - Ve loss va validation accuracy

```python
import matplotlib.pyplot as plt

epochs = [row["epoch"] for row in history]
train_loss = [row["train_loss"] for row in history]
valid_loss = [row["valid_unseen_loss"] for row in history]
valid_acc = [row["valid_unseen_acc"] for row in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, valid_loss, marker="o", label="Valid unseen loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("YOLOv8n-cls Loss Curves - Augmented V2 Cleaning")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_loss.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, valid_acc, marker="o", color="#2ca02c")
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy")
plt.title("YOLOv8n-cls Validation Accuracy - Augmented V2 Cleaning")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_valid_accuracy.png", dpi=200, bbox_inches="tight")
plt.show()
```

## Cell 9 - Danh gia best.pt tren valid_unseen va test

Test chi duoc dung tai buoc nay, sau khi `best.pt` da duoc chon.

```python
evaluate_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "evaluate_yolo.py"),
    "--model_path", "models/yolo/yolo_cls_best.pt",
    "--data_dir", str(DATA_DIR),
    "--output", "outputs/yolo/evaluation_yolo_cls_best.json",
    "--metrics_output", "outputs/yolo/metrics_yolo.json",
    "--imgsz", "224",
    "--batch", "128",
    "--device", "0",
]

subprocess.run(evaluate_command, cwd=PROJECT_DIR, check=True)
```

Khong them `--include_valid_traincopy` vao danh gia chinh. Tap nay lay tu train va chi phu hop lam phu luc.

## Cell 10 - In ket qua tong quan va bang tung lop

```python
evaluation_path = OUTPUT_DIR / "evaluation_yolo_cls_best.json"
assert evaluation_path.is_file(), f"Missing: {evaluation_path}"

evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
valid_result = evaluation["valid_unseen"]
test_result = evaluation["test"]
report = test_result["classification_report"]

assert test_result["samples"] == dataset_counts["test"]["total"]

print("Validation accuracy:", f"{valid_result['accuracy']:.4f}")
print("Test accuracy      :", f"{test_result['accuracy']:.4f}")
print("Test macro F1      :", f"{report['macro avg']['f1-score']:.4f}")
print("Test weighted F1   :", f"{report['weighted avg']['f1-score']:.4f}")
print("Test samples       :", test_result["samples"])
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

## Cell 11 - Confusion matrix va cac cap nham lan lon

```python
import numpy as np
import pandas as pd
import seaborn as sns

class_names = evaluation["class_names"]
cm = np.asarray(test_result["confusion_matrix"], dtype=int)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("YOLOv8n-cls Confusion Matrix - Test - Augmented V2 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

errors = []
for true_index, true_name in enumerate(class_names):
    class_total = int(cm[true_index].sum())
    for pred_index, pred_name in enumerate(class_names):
        if true_index == pred_index or cm[true_index, pred_index] == 0:
            continue
        count = int(cm[true_index, pred_index])
        errors.append({
            "true_class": true_name,
            "predicted_class": pred_name,
            "count": count,
            "rate_in_true_class": count / class_total if class_total else 0.0,
        })

error_df = pd.DataFrame(errors).sort_values("count", ascending=False)
error_df.to_csv(OUTPUT_DIR / "yolo_top_confusions.csv", index=False)
display(error_df.head(15))
```

## Cell 12 - Ve Precision, Recall va F1 tung lop

```python
metrics_df = pd.read_csv(OUTPUT_DIR / "yolo_class_metrics.csv")

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="f1", color="#4C78A8")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.title("YOLOv8n-cls F1-score per Class - Augmented V2 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_f1_per_class.png", dpi=200, bbox_inches="tight")
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
plt.title("YOLOv8n-cls Precision / Recall / F1 - Augmented V2 Cleaning")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "yolo_precision_recall_f1_per_class.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
```

## Cell 13 - Ve support va ti le lop test

```python
metrics_df["ratio"] = metrics_df["support"] / metrics_df["support"].sum()

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="support", color="#72B7B2")
plt.xticks(rotation=35, ha="right")
plt.title("Test Samples per Class - Augmented V2 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_test_support_per_class.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(11, 5))
sns.barplot(data=metrics_df, x="class", y="ratio", color="#F58518")
plt.xticks(rotation=35, ha="right")
plt.title("Test Class Ratio - Augmented V2 Cleaning")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_test_class_ratio.png", dpi=200, bbox_inches="tight")
plt.show()
```

## Cell 14 - Kiem tra output va nen ZIP

```python
required_outputs = [
    PROJECT_DIR / "models" / "yolo" / "yolo_cls_best.pt",
    OUTPUT_DIR / "dataset_audit.json",
    OUTPUT_DIR / "experiment_manifest.json",
    OUTPUT_DIR / "history_yolo.json",
    OUTPUT_DIR / "metrics_yolo.json",
    OUTPUT_DIR / "evaluation_yolo_cls_best.json",
    OUTPUT_DIR / "results.csv",
    OUTPUT_DIR / "yolo_class_metrics.csv",
    OUTPUT_DIR / "yolo_top_confusions.csv",
    OUTPUT_DIR / "yolo_loss.png",
    OUTPUT_DIR / "yolo_valid_accuracy.png",
    OUTPUT_DIR / "yolo_confusion_matrix.png",
    OUTPUT_DIR / "yolo_f1_per_class.png",
    OUTPUT_DIR / "yolo_precision_recall_f1_per_class.png",
    OUTPUT_DIR / "yolo_test_support_per_class.png",
    OUTPUT_DIR / "yolo_test_class_ratio.png",
]

missing = [str(path) for path in required_outputs if not path.is_file()]
assert not missing, "Missing outputs:\n" + "\n".join(missing)

ZIP_PATH = Path("/kaggle/working/yolo_aug_v2_cleaning_outputs.zip")
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for source in [PROJECT_DIR / "models" / "yolo", OUTPUT_DIR]:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(PROJECT_DIR))

print("All required artifacts are ready.")
print("Created:", ZIP_PATH)
print("ZIP size:", f"{ZIP_PATH.stat().st_size / (1024 ** 2):.2f} MB")
```

Tai file `yolo_aug_v2_cleaning_outputs.zip` tu khu vuc Output cua Kaggle Notebook.

## 6. Noi dung can dua vao bao cao V2

### 6.1. Mo ta du lieu

- Tong anh raw-cleaning va so anh tung lop.
- Chinh sach split 85% train, 5% validation, 10% test.
- `valid_traincopy` la tap phu co nguon tu train.
- Quota train V2 bang lop train lon nhat.
- Validation va test khong bi augmentation.

### 6.2. Ly do chon V2

- Gaussian blur mo phong mat chi tiet do out-of-focus hoac nen anh.
- Motion blur mo phong xe hoac camera chuyen dong.
- Unsharp mask tang cuong bien va chi tiet hinh hoc.
- Ba bien doi bo sung cac truong hop chat luong anh ma V1 rain/sun/night chua tap trung truc tiep.

### 6.3. Chi so phai bao cao

- Best validation Top-1 Accuracy va epoch checkpoint.
- Minimum validation loss va epoch tuong ung.
- Test Accuracy, Macro F1, Weighted F1.
- Precision, Recall, F1, support tung lop.
- Cac cap nham lan lon nhat.
- So sanh taxi, minibus, car va truck voi V1.

### 6.4. Cach ket luan

Khong ket luan V2 tot hon chi bang test accuracy. Can xem dong thoi:

- Macro F1 co tang hay khong;
- cac lop it mau co cai thien hay khong;
- blur co lam giam phan biet car, taxi va truck hay khong;
- unsharp mask co giup cac lop co bien ro nhu bicycle, motorcycle, train hay khong;
- khoang cach train loss va validation loss co tang hay khong.

## 7. Dieu kien chap nhan thi nghiem V2

Thi nghiem chi hop le khi:

1. Cell 2 xac nhan du 10 lop va khong co lop rong.
2. Train V2 can bang, cung quota cho 10 lop.
3. `experiment_manifest.json` ghi dataset variant la `v2`.
4. Test samples trong evaluation bang dung tong test cua Cell 2.
5. Test khong tham gia early stopping hoac chon checkpoint.
6. Test loss khong duoc tu suy dien; neu khong do thi phai la `null`.
7. ZIP co model, JSON, CSV va day du bieu do.
