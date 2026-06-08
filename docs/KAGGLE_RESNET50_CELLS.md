# Kaggle ResNet-50 Complete Cell Guide

This guide defines the recommended Kaggle notebook cell order for ResNet-50.

Goals:

1. Train ResNet-50 and save epoch history.
2. Save the best checkpoint.
3. Evaluate after training and save full metrics.
4. Draw charts directly inside Kaggle.
5. Package `models` and `outputs` into one zip.

Important note about resume:

- Current `src/train.py` saves checkpoints, including `models/resnet50_best.pth`.
- However, current `src/train.py` does **not yet expose** a `--resume_checkpoint` CLI argument.
- Therefore, true resume-after-shutdown requires a small future code update.
- The cells below include the intended resume cell, but mark it clearly.

## Cell 1. Find Project Source And Augmented Data

Purpose:

- Locate the project source folder inside Kaggle input.
- Locate the preprocessed `augmented` dataset.

```python
SRC_DIR = !find /kaggle/input -type d -name src | head -1
AUG_DIR = !find /kaggle/input -type d -name augmented | head -1

src_dir = SRC_DIR[0] if len(SRC_DIR) else ""
aug_dir = AUG_DIR[0] if len(AUG_DIR) else ""

print("SRC:", src_dir)
print("AUG:", aug_dir)

assert src_dir, "Cannot find src directory in /kaggle/input"
assert aug_dir, "Cannot find augmented directory in /kaggle/input"
```

Expected:

```text
SRC: /kaggle/input/.../src
AUG: /kaggle/input/.../augmented
```

## Cell 2. Copy Project And Data Into Working Directory

Purpose:

- Put code and data under `/kaggle/working/VehicleTypeRecognition`.
- Kaggle working directory is writable, unlike Kaggle input.

```bash
!mkdir -p /kaggle/working/VehicleTypeRecognition/data
!cp -r "{src_dir}" /kaggle/working/VehicleTypeRecognition/
!cp -r "{aug_dir}" /kaggle/working/VehicleTypeRecognition/data/
```

Expected:

```text
/kaggle/working/VehicleTypeRecognition/src
/kaggle/working/VehicleTypeRecognition/data/augmented
```

## Cell 3. Verify Dataset Structure

Purpose:

- Confirm that `data/augmented` has the expected split folders.

```bash
!cd /kaggle/working/VehicleTypeRecognition && find data/augmented -maxdepth 2 -type d | head -80
```

Expected folders:

```text
data/augmented/train
data/augmented/valid_unseen
data/augmented/valid_traincopy
data/augmented/test
```

## Cell 4. Optional: Restore Previous Models And Outputs

Purpose:

- Use this when continuing from a previous Kaggle run.
- Add the previous `resnet50_outputs.zip` or previous `models/outputs` as a Kaggle input dataset first.

Example if previous run contains `models` and `outputs` folders:

```bash
PREV_ROOT=$(find /kaggle/input -type d -name models | head -1 | xargs dirname)
echo "PREV_ROOT=$PREV_ROOT"

if [ -n "$PREV_ROOT" ]; then
  cp -r "$PREV_ROOT/models" /kaggle/working/VehicleTypeRecognition/ || true
  cp -r "$PREV_ROOT/outputs" /kaggle/working/VehicleTypeRecognition/ || true
fi
```

This cell copies previous files only. It does not automatically resume training
unless `src/train.py` supports a resume argument.

## Cell 5A. Train ResNet-50 From Start

Purpose:

- Train ResNet-50.
- Save train history by epoch.
- Save best checkpoint.
- Save simple final metrics.

```bash
!cd /kaggle/working/VehicleTypeRecognition && python src/train.py \
    --data_dir data/augmented \
    --model resnet50 \
    --batch_size 32 \
    --epochs 30 \
    --patience 7 \
    --lr_head 1e-3 \
    --lr_backbone 1e-5 \
    --num_workers 2
```

Expected files:

```text
models/resnet50_best.pth
models/resnet50_epoch*.pth
outputs/history_resnet50.json
outputs/metrics_resnet50.json
```

`outputs/history_resnet50.json` should contain per-epoch:

- `epoch`
- `train_loss`
- `valid_unseen_loss`
- `valid_unseen_acc`
- `elapsed_s`

`outputs/metrics_resnet50.json` should contain simple final split metrics:

- validation loss/accuracy
- test loss/accuracy
- checkpoint path

## Cell 5B. Intended Resume Training Cell

Purpose:

- Resume training after Kaggle shuts down.

Current status:

- This is the desired command shape.
- It requires `src/train.py` to support `--resume_checkpoint`.
- Current repo code does not fully support this yet.

```bash
!cd /kaggle/working/VehicleTypeRecognition && python src/train.py \
    --data_dir data/augmented \
    --model resnet50 \
    --batch_size 32 \
    --epochs 30 \
    --patience 7 \
    --lr_head 1e-3 \
    --lr_backbone 1e-5 \
    --num_workers 2 \
    --resume_checkpoint models/resnet50_best.pth
```

When resume support is implemented, expected behavior:

- load model weights
- load optimizer state
- continue from `checkpoint_epoch + 1`
- append to or preserve `outputs/history_resnet50.json`
- continue saving best checkpoints

Until then, use Cell 5A for a normal run.

## Cell 6. Plot Loss Directly On Kaggle

Purpose:

- Draw train/validation loss directly inside Kaggle.
- Save the chart into `outputs`.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

history_path = OUTPUT_DIR / "history_resnet50.json"

with open(history_path, "r") as f:
    history = json.load(f)

epochs = [int(row["epoch"]) for row in history]
train_loss = [row["train_loss"] for row in history]
valid_loss = [row["valid_unseen_loss"] for row in history]
valid_acc = [row["valid_unseen_acc"] for row in history]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, valid_loss, marker="o", label="Valid unseen loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-50 Loss Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_loss.png", dpi=200)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(epochs, valid_acc, marker="o", color="#2ca02c", label="Valid unseen accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-50 Validation Accuracy")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_valid_accuracy.png", dpi=200)
plt.show()
```

Expected files:

```text
outputs/resnet50_loss.png
outputs/resnet50_valid_accuracy.png
```

## Cell 7. Evaluate Best Checkpoint

Purpose:

- Evaluate the best checkpoint after training.
- Generate full metrics:
  - Accuracy
  - Precision per class
  - Recall per class
  - F1-score per class
  - Macro F1
  - Weighted F1
  - Confusion matrix
  - Classification report

```bash
!cd /kaggle/working/VehicleTypeRecognition && python src/evaluate.py \
    --checkpoint models/resnet50_best.pth \
    --data_dir data/augmented \
    --batch_size 64 \
    --num_workers 2 \
    --skip_valid_traincopy \
    --output outputs/evaluation_resnet50_best.json
```

Expected file:

```text
outputs/evaluation_resnet50_best.json
```

Important JSON locations:

```text
test.accuracy
test.classification_report
test.classification_report.<class>.precision
test.classification_report.<class>.recall
test.classification_report.<class>.f1-score
test.classification_report.macro avg.f1-score
test.classification_report.weighted avg.f1-score
test.confusion_matrix
```

## Cell 8. Print Main Evaluation Numbers

Purpose:

- Show important metrics directly in Kaggle text output.

```python
import json
from pathlib import Path

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
OUTPUT_DIR = PROJECT_DIR / "outputs"
eval_path = OUTPUT_DIR / "evaluation_resnet50_best.json"

with open(eval_path, "r") as f:
    result = json.load(f)

test = result["test"]
report = test["classification_report"]

print("Test accuracy:", test["accuracy"])
print("Macro F1:", report["macro avg"]["f1-score"])
print("Weighted F1:", report["weighted avg"]["f1-score"])
print()

for cls in result["class_names"]:
    row = report[cls]
    print(
        f"{cls:12s} "
        f"P={row['precision']:.4f} "
        f"R={row['recall']:.4f} "
        f"F1={row['f1-score']:.4f} "
        f"support={int(row['support'])}"
    )
```

## Cell 9. Plot Confusion Matrix Directly On Kaggle

Purpose:

- Draw confusion matrix directly in Kaggle.
- Save chart into `outputs`.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
OUTPUT_DIR = PROJECT_DIR / "outputs"
eval_path = OUTPUT_DIR / "evaluation_resnet50_best.json"

with open(eval_path, "r") as f:
    result = json.load(f)

class_names = result["class_names"]
cm = result["test"]["confusion_matrix"]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet-50 Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_confusion_matrix.png", dpi=200)
plt.show()
```

Expected file:

```text
outputs/resnet50_confusion_matrix.png
```

## Cell 10. Plot F1 / Precision / Recall Per Class

Purpose:

- Draw per-class metrics directly in Kaggle.
- Save chart into `outputs`.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
OUTPUT_DIR = PROJECT_DIR / "outputs"
eval_path = OUTPUT_DIR / "evaluation_resnet50_best.json"

with open(eval_path, "r") as f:
    result = json.load(f)

report = result["test"]["classification_report"]
class_names = result["class_names"]

rows = []
for cls in class_names:
    row = report[cls]
    rows.append({
        "class": cls,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1-score"],
        "support": row["support"],
    })

df = pd.DataFrame(rows)

plt.figure(figsize=(11, 5))
sns.barplot(data=df, x="class", y="f1", color="#4C78A8")
plt.ylim(0, 1)
plt.xticks(rotation=35)
plt.title("ResNet-50 F1-score Per Class - Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_f1_per_class.png", dpi=200)
plt.show()

metric_df = df.melt(
    id_vars="class",
    value_vars=["precision", "recall", "f1"],
    var_name="metric",
    value_name="score",
)

plt.figure(figsize=(12, 5))
sns.barplot(data=metric_df, x="class", y="score", hue="metric")
plt.ylim(0, 1)
plt.xticks(rotation=35)
plt.title("ResNet-50 Precision / Recall / F1 Per Class - Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_precision_recall_f1_per_class.png", dpi=200)
plt.show()
```

Expected files:

```text
outputs/resnet50_f1_per_class.png
outputs/resnet50_precision_recall_f1_per_class.png
```

## Cell 11. Plot Class Support / Class Ratio

Purpose:

- Show the number and ratio of test samples per class.
- This helps explain F1 behavior for each class.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path("/kaggle/working/VehicleTypeRecognition")
OUTPUT_DIR = PROJECT_DIR / "outputs"
eval_path = OUTPUT_DIR / "evaluation_resnet50_best.json"

with open(eval_path, "r") as f:
    result = json.load(f)

report = result["test"]["classification_report"]
class_names = result["class_names"]

rows = []
total_support = 0
for cls in class_names:
    support = int(report[cls]["support"])
    total_support += support
    rows.append({"class": cls, "support": support})

df = pd.DataFrame(rows)
df["ratio"] = df["support"] / max(total_support, 1)

plt.figure(figsize=(11, 5))
sns.barplot(data=df, x="class", y="support", color="#72B7B2")
plt.xticks(rotation=35)
plt.title("Test Samples Per Class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_test_support_per_class.png", dpi=200)
plt.show()

plt.figure(figsize=(11, 5))
sns.barplot(data=df, x="class", y="ratio", color="#F58518")
plt.xticks(rotation=35)
plt.title("Test Class Ratio")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "resnet50_test_class_ratio.png", dpi=200)
plt.show()
```

Expected files:

```text
outputs/resnet50_test_support_per_class.png
outputs/resnet50_test_class_ratio.png
```

## Cell 12. Optional: Plot Data Prep Matrices

Purpose:

- Draw class distribution and augmentation-policy matrix charts.
- This requires `outputs/data_prep_counts.json`.

If `data_prep_counts.json` already exists in your Kaggle input, copy it into
the working project first:

```bash
DATA_PREP_JSON=$(find /kaggle/input -type f -name data_prep_counts.json | head -1)
echo "DATA_PREP_JSON=$DATA_PREP_JSON"

if [ -n "$DATA_PREP_JSON" ]; then
  mkdir -p /kaggle/working/VehicleTypeRecognition/outputs
  cp "$DATA_PREP_JSON" /kaggle/working/VehicleTypeRecognition/outputs/data_prep_counts.json
fi
```

Then plot:

```bash
!cd /kaggle/working/VehicleTypeRecognition && python src/plot_data_prep_matrices.py \
    --report outputs/data_prep_counts.json \
    --out_dir outputs/figures
```

Display directly:

```python
from pathlib import Path
from IPython.display import Image, display

FIG_DIR = Path("/kaggle/working/VehicleTypeRecognition/outputs/figures")

for path in sorted(FIG_DIR.glob("data_prep_*.png")):
    print(path.name)
    display(Image(filename=str(path)))
```

Expected files:

```text
outputs/figures/data_prep_source_target_matrix.png
outputs/figures/data_prep_bucket_count_matrix.png
outputs/figures/data_prep_orig_geo_matrix.png
outputs/figures/data_prep_policy_check_matrix.png
```

## Cell 13. Optional: Generate SVG Figures From Repo Script

Purpose:

- Use the repo plotting script to generate report-ready SVGs.

```bash
!cd /kaggle/working/VehicleTypeRecognition && python src/plot_results.py \
    --history outputs/history_resnet50.json \
    --evaluation outputs/evaluation_resnet50_best.json \
    --split test \
    --out_dir outputs/figures
```

Expected files:

```text
outputs/figures/resnet50_loss.svg
outputs/figures/resnet50_confusion_matrix_test.svg
```

This cell is optional because earlier cells already draw PNGs directly.

## Cell 14. Inspect Output Files Before Zipping

Purpose:

- Confirm all required files exist before packaging.

```bash
!cd /kaggle/working/VehicleTypeRecognition && find models outputs -maxdepth 3 -type f | sort
```

Recommended minimum files:

```text
models/resnet50_best.pth
outputs/history_resnet50.json
outputs/metrics_resnet50.json
outputs/evaluation_resnet50_best.json
outputs/resnet50_loss.png
outputs/resnet50_valid_accuracy.png
outputs/resnet50_confusion_matrix.png
outputs/resnet50_f1_per_class.png
outputs/resnet50_precision_recall_f1_per_class.png
outputs/resnet50_test_support_per_class.png
outputs/resnet50_test_class_ratio.png
```

If using data prep matrices, also expect:

```text
outputs/data_prep_counts.json
outputs/figures/data_prep_*.png
```

## Cell 15. Zip Models And Outputs

Purpose:

- Package everything needed for download or later resume/evaluation.

```bash
!cd /kaggle/working/VehicleTypeRecognition && zip -r /kaggle/working/resnet50_outputs.zip models outputs
```

Expected:

```text
/kaggle/working/resnet50_outputs.zip
```

This zip should contain:

- best checkpoint
- epoch checkpoints
- history JSON
- final train metrics JSON
- evaluation JSON with F1/accuracy/confusion matrix/classification report
- PNG/SVG charts
- optional data prep matrix plots

## Final Required Order

Use this order for a complete run:

```text
1. Find project source and augmented data
2. Copy project and data into working directory
3. Verify dataset structure
4. Optional restore previous models/outputs
5. Train ResNet-50 from start, or resume when resume support exists
6. Plot loss and validation accuracy
7. Evaluate best checkpoint
8. Print main evaluation numbers
9. Plot confusion matrix
10. Plot F1 / precision / recall per class
11. Plot class support / class ratio
12. Optional plot data prep matrices
13. Optional generate SVG figures
14. Inspect output files
15. Zip models and outputs
```

## Requirement Checklist

| Requirement | Covered by cells |
|-------------|------------------|
| Save train loss, valid loss, and accuracy by epoch | Cell 5A, Cell 6 |
| Save best checkpoint | Cell 5A |
| Evaluate on test and valid_unseen | Cell 7 |
| Accuracy | Cell 7, Cell 8 |
| Precision / Recall / F1 per class | Cell 7, Cell 8, Cell 10 |
| Macro F1 and Weighted F1 | Cell 7, Cell 8 |
| Confusion matrix | Cell 7, Cell 9 |
| Classification report | Cell 7 |
| Loss chart | Cell 6 |
| F1 / accuracy chart by class | Cell 10 |
| Class ratio chart | Cell 11 |
| Data prep matrix chart | Cell 12 |
| Zip all outputs | Cell 15 |
| Resume after shutdown | Cell 4 + Cell 5B, but requires future `src/train.py --resume_checkpoint` support |

