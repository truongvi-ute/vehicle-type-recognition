# Documentation Audit Report

This report records the documentation audit and the fixes applied after the audit.

## Audit Scope

Reviewed Markdown files:

- `README.md`
- `docs/ARCHITECTURE_FINAL.md`
- `docs/DATA_STRATEGY.md`
- `docs/AUGMENTATION.md`
- `docs/DATASET_INFO.md`
- `docs/TRAINING_GUIDE.md`
- `docs/DEPLOYMENT.md`
- `docs/MODEL_COMPARISON.md`

## Critical Issues

Status: resolved in the current documentation set.

Resolved items:

- Split math is now explicit: independent original split is 85% `train`, 5% `valid_unseen`, and 10% `test`.
- `valid_traincopy` is documented as an auxiliary copy from `train`, not an independent split.
- Validation/test roles are aligned across all documents.
- Documentation text has been normalized to readable Markdown.

## Medium Issues

Status: resolved in the current documentation set.

Resolved items:

- Folder structure is standardized across documents.
- `data/augmented/` is defined as the training-ready dataset root.
- Only `data/augmented/train/` contains offline-augmented images.
- `valid_unseen`, `valid_traincopy`, and `test` under `data/augmented/` are Base Pipeline outputs without augmentation.
- Base Pipeline wording is standardized.
- Class balancing is defined as train-only quota preparation.
- Model roles are aligned across README and model comparison docs.
- Online augmentation is documented in both augmentation and training docs.

## Minor Issues

Status: resolved or documented.

Resolved items:

- Environment bucket names are standardized as path names `normal`, `rain`, `sun`, and `night`.
- Display labels may use `Normal`, `Rain`, `Sun`, and `Night`.
- README quick commands and TRAINING_GUIDE detailed commands are intentionally separated.
- Deployment documentation now describes the Flask + React inference flow and marks demo outputs as non-official metrics.

## Recommended Fixes

Applied:

1. Created [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md) as the Single Source of Truth.
2. Clarified canonical split math and the auxiliary nature of `valid_traincopy`.
3. Added a consistent metric reporting policy.
4. Standardized project and data folder structure.
5. Clarified `data/augmented/` as a training-ready root with non-augmented validation/test outputs.
6. Defined class balancing, class quota, large class handling, and minority class handling.
7. Added offline and online augmentation policies.
8. Aligned model roles across documents.
9. Standardized Base Pipeline terminology.
10. Updated deployment strategy to Flask + React while keeping inference preprocessing consistent with evaluation preprocessing.

## Current Source of Truth

Use [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md) for all future documentation and implementation decisions.

## Deployment Update

Current official deployment direction:

```text
React UI -> Flask API -> Base Pipeline preprocessing -> Model inference -> JSON response -> React visualization
```

Streamlit is no longer the primary deployment path. A legacy Streamlit `app.py`, if present, is prototype-only.
