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

- Target images/class defaults to the largest train class size.
- Configurable by class.
- Final dataset size depends on configured quotas.

Large classes:

- `boat`
- `car`

Policy:

- Avoid excessive generation.
- Cap outputs to the configured class target.
- Keep 70% of the target as normal-condition outputs.
- Use available images above the 70% normal quota for weather outputs before generating additional samples.

Minority classes:

- `helicopter`
- `taxi`
- `train`
- `bicycle`
- `minibus`

Policy:

- If the class has fewer than 70% of the target, generate variants using Horizontal Flip, Rotation, Shift/Scale, Perspective Transform, Brightness/Contrast, and Coarse Dropout until the normal quota reaches 70%.
- After the normal quota is filled, combine available or generated variants with environment simulation until the remaining 30% weather quota is filled.
- If the class has more than 70% but less than 100% of the target, use the excess over 70% for weather outputs first, then generate only the remaining shortfall.

`data/balanced/` is a train-only intermediate workspace. It must not contain validation or test data.

## Base Pipeline

Base Pipeline:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

The Base Pipeline is used for validation, test, and deployment normalization.

## Offline Augmentation Policy

Offline augmentation is quota-based and applies only to `train`.

| Bucket | Ratio | Example for target class size |
|--------|-------|--------------------------------|
| `normal` | 70% | `round(target * 0.70)` |
| `rain` | 10% | `round(target * 0.10)` |
| `sun` | 10% | `round(target * 0.10)` |
| `night` | 10% | remaining images |

Class fill rule:

- Below 70% of target: fill `normal` to 70% with geometric/content variants first, then create 30% weather outputs.
- Between 70% and 100% of target: reserve 70% as `normal`, use the available excess for weather outputs, then generate only the missing weather outputs.
- At or above target: cap to the target distribution without over-generation.

Validation and test are pass-through Base Pipeline outputs and are not augmented.

## Leakage Prevention Checklist

- Split before balancing and augmentation.
- Keep `test` independent and untouched until final evaluation.
- Keep `valid_unseen` independent.
- Label `valid_traincopy` clearly as copied from train.
- Never augment validation or test sets.
- Never report `valid_traincopy` as official generalization performance.
