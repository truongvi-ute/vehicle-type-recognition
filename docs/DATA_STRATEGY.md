# Data Strategy

This document details split policy, class balancing, and leakage prevention. The authoritative design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

## Split First Protocol

All original images are split before balancing or augmentation.

Independent original split:

| Split | Ratio | Approx. count | Role | Augmentation |
|-------|-------|---------------|------|--------------|
| `train` | 85% | ~30,605 | Training source | Yes, after balancing |
| `valid_unseen` | 5% | ~1,800 | Independent validation | No |
| `test` | 10% | ~3,601 | Final evaluation | No |

Auxiliary validation:

| Set | Source | Role | Augmentation |
|-----|--------|------|--------------|
| `valid_traincopy` | Copied from `train` | Instructor-required reference only | No |

`valid_traincopy` is not an independent split and does not add another percentage to the original 85/5/10 split.

## Metric Reporting Policy

Official reporting order:

1. `test`: primary final metric source.
2. `valid_unseen`: secondary reference metric.
3. `valid_traincopy`: auxiliary reference only.

Do not use `test` for early stopping, hyperparameter tuning, augmentation design, or checkpoint selection.

## Class Balancing

Class balancing applies only to `train`.

Default target:

- About **7,000 images/class**.
- Configurable by class.
- Final dataset size depends on configured quotas.

Large classes:

- `boat`
- `car`

Policy:

- Avoid excessive generation.
- Select, cap, or distribute available images into environment buckets.
- Generate additional samples only for missing bucket quota.

Minority classes:

- `helicopter`
- `taxi`
- `train`
- `bicycle`
- `minibus`

Policy:

- Generate variants using Horizontal Flip, Rotation, Shift/Scale, Perspective Transform, Brightness/Contrast, and Coarse Dropout.
- Combine generated variants with environment simulation until quotas are filled.

`data/balanced/` is a train-only intermediate workspace. It must not contain validation or test data.

## Base Pipeline

Base Pipeline:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

The Base Pipeline is used for validation, test, and deployment normalization.

## Offline Augmentation Policy

Offline augmentation is quota-based and applies only to `train`.

| Bucket | Ratio | Example for 7,000 images/class |
|--------|-------|--------------------------------|
| `normal` | 70% | 4,900 |
| `rain` | 10% | 700 |
| `sun` | 10% | 700 |
| `night` | 10% | 700 |

Validation and test are pass-through Base Pipeline outputs and are not augmented.

## Leakage Prevention Checklist

- Split before balancing and augmentation.
- Keep `test` independent and untouched until final evaluation.
- Keep `valid_unseen` independent.
- Label `valid_traincopy` clearly as copied from train.
- Never augment validation or test sets.
- Never report `valid_traincopy` as official generalization performance.
