# VehicleTypeRecognition Architecture Final

This document is the Single Source of Truth for the VehicleTypeRecognition project. All other Markdown documents must stay consistent with this file.

## Dataset

Vehicle-10 contains **36,006 RGB images** across 10 vehicle classes:

| Class | Vietnamese name | Data group |
|-------|-----------------|------------|
| `bicycle` | Xe dap | Minority |
| `boat` | Thuyen/Tau | Large |
| `bus` | Xe buyt | Medium |
| `car` | O to | Large |
| `helicopter` | Truc thang | Minority |
| `minibus` | Xe khach nho | Minority |
| `motorcycle` | Xe mo to | Medium |
| `taxi` | Taxi | Minority |
| `train` | Tau hoa | Minority |
| `truck` | Xe tai | Medium |

## Split Strategy

The independent split of the original dataset is:

| Split | Ratio | Approx. count | Role |
|-------|-------|---------------|------|
| `train` | 85% | ~30,605 | Training source; the only split used for balancing and augmentation |
| `valid_unseen` | 5% | ~1,800 | Independent validation used during training |
| `test` | 10% | ~3,601 | Independent final evaluation set |

An additional auxiliary validation view is created:

| Auxiliary set | Size policy | Source | Role |
|---------------|-------------|--------|------|
| `valid_traincopy` | Same working scale as validation, about ~1,800 images unless configured otherwise | Copied from `train` | Instructor-required reference only |

Important rules:

- `valid_traincopy` is **not** an independent split and does **not** add another 5% of original data.
- The independent original split is always `train` + `valid_unseen` + `test` = 100%.
- `test` is used only after the model/checkpoint is finalized.
- `valid_traincopy` must never be used as official evidence of generalization.

## Class Balancing

Class balancing is applied only to `train`.

Default target:

- Target images/class defaults to the largest train class size.
- The target is configurable; final dataset size depends on the chosen quota for each class.

Large classes: `boat`, `car`.

Handling:

- Do not generate excessive extra images.
- Cap the final output to the configured class target.
- Keep 70% of the target as normal-condition outputs and distribute 30% across weather buckets.
- Generate extra images only when the class is still below the configured target.

Minority classes: `helicopter`, `taxi`, `train`, `bicycle`, `minibus`.

Handling:

- If the class has fewer than 70% of the target, generate geometric/content variants until the normal-condition quota reaches 70%.
- After the 70% normal quota is filled, generate or reuse images for the 30% weather quota.
- If the class already has more than 70% but less than 100% of the target, use the available excess over 70% for weather outputs first, then generate only the remaining shortfall.
- Allowed transforms include Horizontal Flip, Rotation, Shift/Scale, Perspective Transform, Brightness/Contrast, and Coarse Dropout.

The `data/balanced/` folder is a train-only intermediate workspace for quota preparation. It must not contain `valid_unseen`, `valid_traincopy`, or `test`.

## Base Pipeline

The Base Pipeline is the only preprocessing used for validation, test, and deployment input normalization:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

No additional cleanup filter or image transformation is part of the Base Pipeline.

## Offline Augmentation

Offline augmentation is quota-based and applies only to `train`.

Default environment distribution per class:

| Environment bucket | Ratio | Example for target class size |
|--------------------|-------|--------------------------------|
| `normal` | 70% | `round(target * 0.70)` |
| `rain` | 10% | `round(target * 0.10)` |
| `sun` | 10% | `round(target * 0.10)` |
| `night` | 10% | remaining images |

Purpose:

- Keep the environment distribution close to realistic usage.
- Keep normal-condition images as the majority.
- Fill the 70% normal quota before using geometric/content transforms for weather shortfalls.
- Avoid duplicating every image into every environment.

Class fill behavior:

- Classes at or above target are capped to the target distribution and are not over-generated.
- Classes below 70% of target are first expanded with geometric/content transforms until the normal bucket reaches 70% of target.
- Classes between 70% and 100% of target reserve 70% for `normal`, use the available excess over 70% for weather outputs, and generate only the missing weather count.
- Weather outputs are split across `rain`, `sun`, and `night`, each targeting about 10% of the class target.

Bucket behavior:

- `normal`: Base Pipeline output.
- `rain`: simulated rain or wet/blurred conditions.
- `sun`: simulated glare or strong sunlight.
- `night`: simulated low-light conditions.

Validation and test restrictions:

- `valid_unseen` is not augmented.
- `valid_traincopy` is not augmented.
- `test` is not augmented.
- No augmented variant of validation/test images may appear in train.

## Online Augmentation

Online augmentation is optional and applies only during training batches.

| Model family | Online augmentation |
|--------------|---------------------|
| ResNet-50 | MixUp and/or CutMix |
| Vision Transformer | MixUp and/or CutMix |
| YOLO-cls | Mosaic if supported by the training configuration |

Online augmentation must not be applied to validation, test, or deployment inference.

## Training Strategy

Supported model families:

| Model | Role |
|-------|------|
| ResNet-50 | Stable CNN baseline |
| Vision Transformer | Global-context comparison model |
| YOLO-cls | Fast inference comparison model |

Training defaults:

- Train on `data/augmented/train`.
- Use a maximum of 100 epochs unless configured otherwise.
- Use early stopping with patience around 10-15 epochs.
- Prefer transfer learning from suitable pretrained checkpoints.
- Track `valid_unseen` as the primary validation signal.
- Track `valid_traincopy` separately only when required.

## Evaluation Strategy

Official reporting priority:

1. **Test set**: primary final metric source.
2. **`valid_unseen`**: secondary reference metric.
3. **`valid_traincopy`**: auxiliary reference only, never official.

Recommended outputs:

- Accuracy.
- Macro F1.
- Per-class Precision, Recall, and F1-score.
- Confusion matrix.
- Classification report.

Evaluation preprocessing:

- Use the Base Pipeline only.
- Do not apply offline or online augmentation.

## Deployment Strategy

The official deployment direction is **Flask + React**.

Streamlit is no longer the primary deployment path. If a legacy Streamlit `app.py` still exists, it is treated only as a prototype, not the production demo.

Deployment flow:

```text
React UI -> Flask API -> Base Pipeline preprocessing -> Model inference -> JSON response -> React visualization
```

Backend responsibilities:

- Use Flask as the REST API server.
- Load trained model checkpoints from `models/`.
- Receive uploaded images from the frontend.
- Apply the Base Pipeline exactly as used in evaluation.
- Run inference with the selected model.
- Return JSON with top-3 predicted classes, confidence scores, model name, and processing time when available.

Frontend responsibilities:

- Use React for the user interface.
- Let users choose the model.
- Upload and preview the original image.
- Display the preprocessed image if the backend returns it.
- Display top-3 predictions.
- Display a confidence chart.
- Provide a more flexible UI than the legacy Streamlit prototype.

Official model quality must still be reported from `test`, with `valid_unseen` as secondary reference.

## Project Structure

Canonical project structure:

```text
VehicleTypeRecognition/
├── backend/
│   ├── app.py
│   ├── routes/
│   ├── services/
│   ├── utils/
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── src/
│   └── public/
├── models/
├── outputs/
├── data/
│   ├── raw/
│   │   └── <class>/
│   ├── splits/
│   │   ├── train/<class>/
│   │   ├── valid_unseen/<class>/
│   │   ├── valid_traincopy/<class>/
│   │   └── test/<class>/
│   ├── balanced/
│   │   └── <class>/
│   └── augmented/
│       ├── train/<class>/<source>_<bucket>_<orig|geo>_<index>.jpg
│       ├── valid_unseen/<class>/
│       ├── valid_traincopy/<class>/
│       └── test/<class>/
├── src/
│   ├── data_split.py
│   ├── balance_classes.py
│   ├── augment_offline.py
│   ├── dataset.py
│   ├── train.py
│   ├── train_vit.py
│   ├── train_yolo.py
│   └── evaluate.py
├── docs/
│   ├── ARCHITECTURE_FINAL.md
│   ├── DATA_STRATEGY.md
│   ├── AUGMENTATION.md
│   ├── DATASET_INFO.md
│   ├── TRAINING_GUIDE.md
│   ├── DEPLOYMENT.md
│   ├── MODEL_COMPARISON.md
│   └── AUDIT_REPORT.md
├── requirements.txt
└── README.md
```

Notes:

- `data/augmented/` is the training-ready dataset root.
- Only `data/augmented/train/` contains offline-augmented images.
- `data/augmented/valid_unseen/`, `data/augmented/valid_traincopy/`, and `data/augmented/test/` are pass-through Base Pipeline outputs without augmentation.
- `backend/app.py` is the Flask API entry point in the official deployment structure.
- A root-level Streamlit `app.py`, if present, is legacy/prototype only.
