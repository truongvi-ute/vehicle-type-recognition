# YOLO V1 Kaggle - ban sua loi

Notebook cu da train thanh cong nhung dung giao dien cu cua `train_yolo.py`:

```text
--project outputs/yolo
```

Run cua Ultralytics vi vay nam trong `runs/classify/...`, con
`outputs/yolo/history_yolo.json` khong duoc tao. Cell ve bieu do sau do loi
`FileNotFoundError`.

Ban duoi day dung voi `src/train_yolo.py` hien tai. Cac cell phai duoc chay theo
dung thu tu. Notebook nay danh cho **Raw Original + Augmented V1** trong log da
gui: 75.620 anh train, 1.800 validation unseen va 3.601 test. Neu dung Raw
Cleaning, doi `DATASET_ID` thanh `raw_cleaning`; khong sua so lieu bang tay.

## Cach A - Cuu run cu, khong train lai

Chi dung cach nay neu Kaggle session hien tai van con `best.pt` va `results.csv`
trong `/kaggle/working/VehicleTypeRecognition/runs/classify`.

### Cell R1 - Tim artifact cu va source code moi

```python
from pathlib import Path
import shutil

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")

src_candidates = []
for path in Path("/kaggle/input").rglob("src"):
    train_file = path / "train_yolo.py"
    evaluate_file = path / "evaluate_yolo.py"
    if not train_file.is_file() or not evaluate_file.is_file():
        continue
    code = train_file.read_text(encoding="utf-8", errors="ignore")
    if "--dataset_variant" in code and "history_yolo.json" in code:
        src_candidates.append(path)

assert len(src_candidates) == 1, (
    "Can tim dung mot bo src moi. Tim thay: "
    + str([str(path) for path in src_candidates])
)
SRC_DIR = src_candidates[0]

results_candidates = list((PROJECT_DIR / "runs" / "classify").rglob("results.csv"))
best_candidates = list((PROJECT_DIR / "runs" / "classify").rglob("weights/best.pt"))
assert results_candidates, "Khong con results.csv cua run cu"
assert best_candidates, "Khong con best.pt cua run cu"

RESULTS_CSV = max(results_candidates, key=lambda path: path.stat().st_mtime)
RUN_DIR = RESULTS_CSV.parent
BEST_PT = RUN_DIR / "weights" / "best.pt"
assert BEST_PT.is_file(), f"Run moi nhat khong co best.pt: {RUN_DIR}"

shutil.copytree(SRC_DIR, PROJECT_DIR / "src", dirs_exist_ok=True)
print("Source :", SRC_DIR)
print("Run cu :", RUN_DIR)
print("Best PT:", BEST_PT)
```

### Cell R2 - Tao lai history, metrics va checkpoint chuan

```python
import json
import sys

sys.path.insert(0, str(PROJECT_DIR))
from src.train_yolo import convert_results, copy_training_artifacts

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo"
MODEL_DIR = PROJECT_DIR / "models" / "yolo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

history, training_summary = convert_results(RESULTS_CSV)
(OUTPUT_DIR / "history_yolo.json").write_text(
    json.dumps(history, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
shutil.copy2(BEST_PT, MODEL_DIR / "yolo_cls_best.pt")
copy_training_artifacts(RUN_DIR, OUTPUT_DIR)

metrics = {
    "model": "yolo",
    "architecture": "yolov8n-cls",
    "dataset_id": "raw_original",
    "dataset_variant": "v1",
    "checkpoint": "models/yolo/yolo_cls_best.pt",
    "primary_metric_split": "valid_unseen",
    "official_evaluation_split": "test",
    "training_summary": training_summary,
    "valid_unseen": {
        "loss": training_summary["valid_loss_at_best_checkpoint"],
        "accuracy": training_summary["best_valid_unseen_accuracy"],
    },
    "test": None,
    "valid_traincopy": None,
}
(OUTPUT_DIR / "metrics_yolo.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

required = [
    OUTPUT_DIR / "history_yolo.json",
    OUTPUT_DIR / "metrics_yolo.json",
    MODEL_DIR / "yolo_cls_best.pt",
]
for path in required:
    assert path.is_file(), f"Chua tao duoc: {path}"
    print(path, path.stat().st_size, "bytes")
```

Sau do chay tu **Cell 7** cua quy trinh moi ben duoi.

## Cach B - Chay moi tu dau

### Cell 1 - Tim dung dataset va dung phien ban src

```python
from pathlib import Path

EXPECTED_CLASSES = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
DATASET_ID = "raw_original"
VARIANT = "v1"
RUN_NAME = "yolo_raw_original_v1"

src_candidates = []
for path in Path("/kaggle/input").rglob("src"):
    train_file = path / "train_yolo.py"
    evaluate_file = path / "evaluate_yolo.py"
    if not train_file.is_file() or not evaluate_file.is_file():
        continue
    code = train_file.read_text(encoding="utf-8", errors="ignore")
    if "--dataset_variant" in code and "history_yolo.json" in code:
        src_candidates.append(path)

data_candidates = [
    path for path in Path("/kaggle/input").rglob("augmented")
    if all((path / split).is_dir() for split in ("train", "valid_unseen", "test"))
]

assert len(src_candidates) == 1, (
    "Can dung mot bo src moi co --dataset_variant. Tim thay: "
    + str([str(path) for path in src_candidates])
)
assert len(data_candidates) == 1, (
    "Can dung mot thu muc augmented. Tim thay: "
    + str([str(path) for path in data_candidates])
)

SRC_DIR = src_candidates[0]
DATA_DIR = data_candidates[0]
print("SRC :", SRC_DIR)
print("DATA:", DATA_DIR)
```

Neu assertion source loi, Kaggle Dataset `vehicle-src` dang chua code cu. Upload
lai `src.zip` hien tai cua project roi gan dataset source moi vao notebook. Khong
nen bat dau train khi cell nay chua qua.

### Cell 2 - Audit dataset that, khong hard-code so mau

```python
dataset_counts = {}
for split in ("train", "valid_unseen", "valid_traincopy", "test"):
    split_dir = DATA_DIR / split
    assert split_dir.is_dir(), f"Missing split: {split_dir}"

    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    assert actual_classes == EXPECTED_CLASSES, (
        f"Sai class mapping tai {split}: {actual_classes}"
    )

    per_class = {}
    for class_name in EXPECTED_CLASSES:
        count = sum(
            1 for path in (split_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        assert count > 0, f"Empty class: {split}/{class_name}"
        per_class[class_name] = count

    dataset_counts[split] = per_class
    print(f"{split:16s} total={sum(per_class.values()):,}", per_class)

assert sum(dataset_counts["valid_unseen"].values()) > 0
assert sum(dataset_counts["test"].values()) > 0
```

### Cell 3 - Copy code, doc anh truc tiep tu Kaggle Input

```python
import shutil

shutil.rmtree(PROJECT_DIR, ignore_errors=True)
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC_DIR, PROJECT_DIR / "src")

train_code = (PROJECT_DIR / "src" / "train_yolo.py").read_text(encoding="utf-8")
assert "--dataset_variant" in train_code
assert "history_yolo.json" in train_code
print("Project ready:", PROJECT_DIR)
```

### Cell 4 - Cai dung Ultralytics, khong thay doi RAPIDS/Torch cua Kaggle

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
        [
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            f"ultralytics=={PINNED_ULTRALYTICS}",
        ],
        check=True,
    )

import pandas
import sklearn
import torch
import ultralytics

assert torch.cuda.is_available(), "Hay bat GPU Accelerator trong Kaggle"
print("Ultralytics:", ultralytics.__version__)
print("Torch       :", torch.__version__)
print("GPU         :", torch.cuda.get_device_name(0))
```

Canh bao xung dot `dask-cuda`, `numba-cuda` trong log cu den tu viec cho `pip`
tu nang cap dependency. `--no-deps` tranh thay doi moi truong CUDA/RAPIDS co san.

### Cell 5 - Kiem tra adapter truoc khi train

```python
prepare_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "train_yolo.py"),
    "--data_dir", str(DATA_DIR),
    "--dataset_id", DATASET_ID,
    "--dataset_variant", VARIANT,
    "--prepare_only",
    "--reset",
]
subprocess.run(prepare_command, cwd=PROJECT_DIR, check=True)

audit_path = PROJECT_DIR / "outputs" / "yolo" / "dataset_audit.json"
assert audit_path.is_file(), f"Missing audit: {audit_path}"
print(audit_path.read_text(encoding="utf-8")[:3000])
```

### Cell 6 - Train YOLO V1

```python
train_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "train_yolo.py"),
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

# Khong them --project. Script tu ghi vao outputs/yolo_runs va outputs/yolo.
subprocess.run(train_command, cwd=PROJECT_DIR, check=True)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "yolo"
MODEL_PATH = PROJECT_DIR / "models" / "yolo" / "yolo_cls_best.pt"
required_after_train = [
    MODEL_PATH,
    OUTPUT_DIR / "history_yolo.json",
    OUTPUT_DIR / "metrics_yolo.json",
    OUTPUT_DIR / "experiment_manifest.json",
    OUTPUT_DIR / "dataset_audit.json",
    OUTPUT_DIR / "results.csv",
]
for path in required_after_train:
    assert path.is_file(), f"Training xong nhung thieu artifact: {path}"
    print(path.relative_to(PROJECT_DIR), path.stat().st_size, "bytes")
```

### Cell 7 - Danh gia best checkpoint tren validation unseen va test

```python
evaluate_command = [
    sys.executable,
    str(PROJECT_DIR / "src" / "evaluate_yolo.py"),
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
assert evaluation_path.is_file(), f"Missing evaluation: {evaluation_path}"
```

### Cell 8 - Kiem chung so lieu truoc khi ve bieu do

```python
import json

history = json.loads((OUTPUT_DIR / "history_yolo.json").read_text(encoding="utf-8"))
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

assert history, "History rong"
assert evaluation["dataset_id"] == DATASET_ID
assert evaluation["dataset_variant"] == VARIANT

test = evaluation["test"]
report = test["classification_report"]
matrix = test["confusion_matrix"]
support_total = sum(int(report[name]["support"]) for name in evaluation["class_names"])
matrix_total = sum(sum(int(value) for value in row) for row in matrix)
correct = sum(matrix[index][index] for index in range(len(matrix)))
computed_accuracy = correct / matrix_total

assert support_total == test["samples"]
assert matrix_total == test["samples"]
assert abs(computed_accuracy - test["accuracy"]) < 1e-6
assert test["samples"] == sum(dataset_counts["test"].values())

print("Completed epochs:", len(history))
print("Test samples    :", test["samples"])
print("Test accuracy   :", f"{test['accuracy']:.6f}")
print("Macro F1        :", f"{report['macro avg']['f1-score']:.6f}")
print("Weighted F1     :", f"{report['weighted avg']['f1-score']:.6f}")
```

### Cell 9 - Ve loss va validation accuracy

```python
import matplotlib.pyplot as plt

epochs = [int(row["epoch"]) for row in history]
train_loss = [float(row["train_loss"]) for row in history]
valid_loss = [float(row["valid_unseen_loss"]) for row in history]
valid_acc = [float(row["valid_unseen_acc"]) for row in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, valid_loss, marker="o", label="Validation unseen loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("YOLOv8n-cls Loss Curves - Raw Original V1")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_loss.png", dpi=200)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, valid_acc, marker="o", color="#2ca02c")
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy")
plt.title("YOLOv8n-cls Validation Accuracy - Raw Original V1")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_valid_accuracy.png", dpi=200)
plt.show()
```

### Cell 10 - Bang chi so va confusion matrix

```python
import pandas as pd
import seaborn as sns

rows = []
for class_name in evaluation["class_names"]:
    row = report[class_name]
    rows.append({
        "class": class_name,
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1-score"]),
        "support": int(row["support"]),
    })

metrics_df = pd.DataFrame(rows)
display(metrics_df.style.format({
    "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"
}))

plt.figure(figsize=(11, 9))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=evaluation["class_names"],
    yticklabels=evaluation["class_names"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("YOLOv8n-cls Confusion Matrix - Test - Raw Original V1")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yolo_confusion_matrix.png", dpi=200)
plt.show()
```

### Cell 11 - Nen output de tai ve

```python
import zipfile

ZIP_PATH = Path("/kaggle/working/yolo_raw_original_v1_outputs.zip")
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for source in [PROJECT_DIR / "models" / "yolo", OUTPUT_DIR]:
        for file_path in sorted(source.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(PROJECT_DIR))

with zipfile.ZipFile(ZIP_PATH) as archive:
    archived = set(archive.namelist())

required_in_zip = {
    "models/yolo/yolo_cls_best.pt",
    "outputs/yolo/history_yolo.json",
    "outputs/yolo/metrics_yolo.json",
    "outputs/yolo/evaluation_yolo_cls_best.json",
    "outputs/yolo/experiment_manifest.json",
    "outputs/yolo/dataset_audit.json",
}
missing = sorted(required_in_zip - archived)
assert not missing, f"ZIP thieu file: {missing}"

print("Created:", ZIP_PATH)
print("Size   :", f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB")
```

## Loi cu va sua doi tuong ung

| Loi cu | Ban sua |
| --- | --- |
| `find ... | head -1` co the lay nham dataset/src | Kiem tra cau truc va marker cua source, dung lai neu co nhieu candidate |
| Cai lai ca dependency lam xung dot RAPIDS | Pin `ultralytics==8.4.66` voi `--no-deps` |
| Dung `--project` cua script cu | Bo `--project`, dung `--dataset_variant v1` |
| Khong phan biet raw va cleaning | Ghi `--dataset_id raw_original` vao manifest/evaluation |
| Ve bieu do truoc khi xac nhan history | Assert du artifact ngay sau train |
| So lieu co the lech ma khong biet | Kiem tra support, confusion matrix, samples va accuracy truoc khi ve |
