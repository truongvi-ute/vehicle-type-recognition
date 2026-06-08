# Model Comparison

This document compares the model families used by the project. The authoritative design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

## Model Roles

| Model | Architecture type | Approx. parameters | Project role |
|-------|-------------------|--------------------|--------------|
| ResNet-50 | CNN with residual connections | ~24.5M | Stable CNN baseline |
| Vision Transformer | Transformer with self-attention | ~86M for ViT-Base | Global-context comparison model |
| YOLO-cls | Lightweight classification model family | ~3M-7M for common small variants | Fast inference comparison model |

## ResNet-50

ResNet-50 is the stable CNN baseline. It is useful for comparison because residual connections make deep CNN training reliable on medium-sized image datasets.

Strengths:

- Stable training behavior.
- Strong baseline for image classification.
- Lower data appetite than many transformer models.

## Vision Transformer

Vision Transformer is used as a global-context comparison model. It processes image patches with self-attention and can capture long-range visual relationships.

Strengths:

- Strong global context modeling.
- Useful comparison point against CNN behavior.
- Can perform well with sufficient data and regularization.

## YOLO-cls

YOLO-cls is used as a fast inference comparison model. It is suitable for demo and latency-sensitive inference experiments.

Strengths:

- Fast inference.
- Lightweight model variants.
- Convenient training and deployment tooling.

## Shared Evaluation Policy

All model families must follow the same evaluation policy:

1. Report `test` as the official metric source.
2. Report `valid_unseen` as secondary reference.
3. Keep `valid_traincopy` as auxiliary only.

All models use the same Base Pipeline for validation, test, and deployment preprocessing.
