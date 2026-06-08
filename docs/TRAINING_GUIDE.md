# Training Guide

This document describes model training and evaluation. The authoritative design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

## Training Data

Train on:

- `data/augmented/train`

This folder contains quota-based offline-augmented training images.

Validation sources:

- `data/augmented/valid_unseen`: independent validation, no augmentation.
- `data/augmented/valid_traincopy`: copied from train, no augmentation, auxiliary only.

## Training Strategy

Defaults:

- Maximum 100 epochs unless configured otherwise.
- Early stopping patience around 10-15 epochs.
- Transfer learning from suitable pretrained checkpoints.
- Primary validation signal: `valid_unseen`.
- Optional auxiliary tracking: `valid_traincopy`.

Do not use `test` during training, tuning, or checkpoint selection.

## Commands

Quick commands:

```bash
python src/train.py --model resnet50 --data_dir data/augmented --max_epochs 100 --patience 10
python src/train_vit.py --model vit_base --data_dir data/augmented --max_epochs 100 --patience 10
python src/train_yolo.py --model yolov8n-cls --data_dir data/augmented --epochs 100 --patience 10
```

Detailed example commands:

```bash
python src/train.py --model resnet50 --data_dir data/augmented --batch_size 64 --max_epochs 100 --patience 10 --lr 1e-3
```

```bash
python src/train_vit.py --model vit_base_patch16_224 --data_dir data/augmented --batch_size 32 --max_epochs 100 --patience 15 --lr 1e-4
```

```bash
yolo classify train model=yolov8n-cls.pt data=data/augmented epochs=100 patience=10 batch=128 imgsz=224
```

## Online Augmentation

Online augmentation applies only to training batches:

| Model family | Online augmentation |
|--------------|---------------------|
| ResNet-50 | MixUp and/or CutMix |
| Vision Transformer | MixUp and/or CutMix |
| YOLO-cls | Mosaic if supported |

Validation, test, and deployment inference use no augmentation.

## Evaluation Strategy

Official reporting order:

1. `test`: primary final metric source.
2. `valid_unseen`: secondary reference metric.
3. `valid_traincopy`: auxiliary reference only.

Recommended metrics:

- Accuracy.
- Macro F1.
- Per-class Precision, Recall, and F1-score.
- Confusion matrix.
- Classification report.

Evaluation preprocessing:

- Base Pipeline only.
- Resize with preserved aspect ratio.
- Zero-pad to **224x224**.

## Troubleshooting Signals

- If `valid_traincopy` is high but `valid_unseen` or `test` is low, treat it as overfitting or memorization risk.
- If minority class F1 is low, review class quotas and minority augmentation.
- If Flask + React deployment predictions differ from evaluation behavior, verify that the Flask API uses the same Base Pipeline.
- A legacy Streamlit `app.py`, if present, is prototype-only and not the official production demo.
