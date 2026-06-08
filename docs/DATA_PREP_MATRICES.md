# Data Prep Matrix Guide

This guide explains the dataset statistics matrices generated after running
`src/data_prep.py`.

The JSON report is saved by default to:

```text
outputs/data_prep_counts.json
```

The matrix tables are stored under:

```text
matrices
```

Each matrix uses this shape:

```json
{
  "columns": ["class", "..."],
  "rows": [
    ["bicycle", "..."],
    ["boat", "..."],
    ["TOTAL", "..."]
  ]
}
```

The expected row count is 11:

- 10 vehicle classes
- 1 `TOTAL` row

## 1. `split_dataset_matrix`

This matrix summarizes the dataset before and after splitting.

Purpose:

- Check how many images each class has in the raw dataset.
- Check how many images each class has in each split.
- Confirm that independent validation, copied validation, and test are separated clearly.

Columns:

| Column | Meaning |
|--------|---------|
| `class` | Vehicle class name, plus a final `TOTAL` row. |
| `raw` | Number of original raw images in the class. |
| `train` | Number of images in the independent train split. |
| `valid_unseen` | Number of independent validation images. |
| `valid_traincopy` | Number of validation-reference images copied from train. |
| `test` | Number of independent test images. |

How to read it:

- `raw` is the original class distribution.
- `train`, `valid_unseen`, and `test` should reflect the split policy.
- `valid_traincopy` is not an independent split; it is copied from train for auxiliary reference.
- The `TOTAL` row gives the total number of images per column.

Recommended quick checks:

```text
raw total = total original images
train + valid_unseen + test = raw total
valid_traincopy is reported separately and should not be added to raw total
```

## 2. `augmentation_summary_matrix`

This matrix summarizes how train images are converted into the final augmented
training set.

Purpose:

- Check how many train images each class starts with.
- Check how many physical files must be added to reach the class target.
- Check how many outputs use geo/content generation.
- Check how each environment bucket is filled from original images or generated variants.

Columns:

| Column | Meaning |
|--------|---------|
| `class` | Vehicle class name, plus a final `TOTAL` row. |
| `train_source` | Number of train images used as the augmentation source. |
| `physical_added` | Number of physical files added beyond the train source to reach target. |
| `geo_generated` | Total number of output files marked as `geo`. |
| `normal_orig` | Normal/base outputs made from original train images. |
| `normal_geo` | Normal/base outputs made from geo/content-generated variants. |
| `rain_orig` | Rain outputs made from original train images. |
| `rain_geo` | Rain outputs made from geo/content-generated variants. |
| `sun_orig` | Sun outputs made from original train images. |
| `sun_geo` | Sun outputs made from geo/content-generated variants. |
| `night_orig` | Night outputs made from original train images. |
| `night_geo` | Night outputs made from geo/content-generated variants. |
| `final_total` | Final number of augmented train images for the class. |

How to read common cases:

| Case | Expected signal |
|------|-----------------|
| Class below 70% of target | `normal_geo > 0`; geo is needed to fill normal to 70%. |
| Class between 70% and target | `normal_geo = 0`; some weather can use original images, remaining weather uses geo. |
| Class at or above target | `physical_added = 0`; outputs are capped/distributed without over-generation. |

Recommended quick checks:

```text
final_total should equal the target for every class
normal_orig + normal_geo should equal 70% of target
rain_orig + rain_geo should equal about 10% of target
sun_orig + sun_geo should equal about 10% of target
night_orig + night_geo should fill the remaining target count
```

## Plotting The Matrices

To generate PNG heatmaps from the JSON report:

```powershell
python src/plot_data_prep_matrices.py
```

To plot only one matrix:

```powershell
python src/plot_data_prep_matrices.py --matrix split_dataset_matrix
```

Default output:

```text
outputs/figures/data_prep_<matrix_name>.png
```
