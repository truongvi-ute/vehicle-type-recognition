# Dataset Info

This document summarizes the Vehicle-10 dataset. The authoritative project design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

## Dataset

Vehicle-10 contains **36,006 RGB images** across 10 vehicle classes.

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

## Split Summary

Independent original split:

| Split | Ratio | Approx. count | Description |
|-------|-------|---------------|-------------|
| `train` | 85% | ~30,605 | Source for balancing, offline augmentation, and training |
| `valid_unseen` | 5% | ~1,800 | Independent validation |
| `test` | 10% | ~3,601 | Independent final evaluation |

Auxiliary set:

- `valid_traincopy` is copied from `train`.
- It is about the same working scale as validation unless configured otherwise.
- It is not an independent subset of the original dataset.
- It must not be used for official final evaluation.

## Data Risks

Class imbalance:

- Large classes: `boat`, `car`.
- Minority classes: `helicopter`, `taxi`, `train`, `bicycle`, `minibus`.

Data leakage rules:

- Do not copy `test` into any training or validation source.
- Do not augment `valid_unseen`, `valid_traincopy`, or `test`.
- Do not place augmented variants of validation or test images in train.
- Report `test` first, `valid_unseen` second, and `valid_traincopy` only as auxiliary reference.

## Canonical Data Structure

```text
data/
├── raw/<class>/
├── splits/
│   ├── train/<class>/
│   ├── valid_unseen/<class>/
│   ├── valid_traincopy/<class>/
│   └── test/<class>/
├── balanced/
│   └── <class>/
└── augmented/
    ├── train/<class>/<source>_<bucket>_<orig|geo>_<index>.jpg
    ├── valid_unseen/<class>/
    ├── valid_traincopy/<class>/
    └── test/<class>/
```

`data/augmented/` is the training-ready dataset root. Only `train/` contains offline augmentation. Validation and test folders are Base Pipeline outputs without augmentation.

The official deployment structure is Flask + React and lives outside the dataset folders:

```text
backend/
├── app.py
├── routes/
├── services/
├── utils/
└── requirements.txt

frontend/
├── package.json
├── src/
└── public/
```

If a legacy root-level Streamlit `app.py` still exists, it is prototype-only and not the official deployment path.

## Quota Summary

Default class quota:

- Target images/class defaults to the largest train class size.
- Configurable by class.
- Total image count after augmentation depends on configured quotas.

Default environment quota:

| Bucket | Ratio | Example for target class size |
|--------|-------|--------------------------------|
| `normal` | 70% | `round(target * 0.70)` |
| `rain` | 10% | `round(target * 0.10)` |
| `sun` | 10% | `round(target * 0.10)` |
| `night` | 10% | remaining images |

Class fill policy:

- Classes at or above target are capped to the target distribution and are not over-generated.
- Classes below 70% of target use geometric/content transforms to fill `normal` to 70% first, then create 30% weather outputs.
- Classes between 70% and 100% of target keep 70% as `normal`, use available excess images for weather outputs first, and generate only the remaining shortfall.
