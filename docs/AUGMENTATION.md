# Augmentation Strategy

This document covers offline and online augmentation. The authoritative design is [ARCHITECTURE_FINAL.md](ARCHITECTURE_FINAL.md).

## Scope

Augmentation is applied only to training data.

No augmentation is applied to:

- `valid_unseen`
- `valid_traincopy`
- `test`
- deployment inference inputs

## Base Pipeline

Base Pipeline:

1. Resize while preserving aspect ratio.
2. Zero-pad to **224x224**.

`normal` bucket images are Base Pipeline outputs.

```python
def apply_pipeline_base(img):
    h, w = img.shape[:2]
    scale = 224 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top = (224 - new_h) // 2
    left = (224 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized
    return canvas
```

## Offline Augmentation

Offline augmentation is quota-based.

Default class target:

- Target images/class defaults to the largest train class size.
- Configurable per class.

Default environment distribution:

| Bucket | Ratio | Example for target class size |
|--------|-------|--------------------------------|
| `normal` | 70% | `round(target * 0.70)` |
| `rain` | 10% | `round(target * 0.10)` |
| `sun` | 10% | `round(target * 0.10)` |
| `night` | 10% | remaining images |

Total image count after augmentation depends on configured quotas.

Class fill policy:

- If a class already meets or exceeds the target, cap it to the target distribution and do not over-generate.
- If a class has fewer than 70% of the target, use geometric/content transforms to fill the `normal` bucket up to 70% first.
- After the 70% `normal` quota is filled, create the remaining 30% as weather outputs split across `rain`, `sun`, and `night`.
- If a class has more than 70% but less than 100% of the target, keep 70% as `normal`, use the available excess over 70% for weather outputs first, and generate only the remaining weather shortfall.

## Bucket Definitions

`normal`:

- Base Pipeline output.
- No environment simulation.

`rain`:

- Simulated rain, wet-looking conditions, slight blur, or reduced contrast.

`sun`:

- Simulated glare or strong sunlight.

`night`:

- Simulated low-light conditions.

## Large Classes

Large classes:

- `boat`
- `car`

Policy:

- Do not over-generate.
- Cap outputs to the configured class target.
- Fill environment buckets from available train images where possible.
- Generate only the missing samples needed to maintain the 70% normal and 30% weather distribution.

## Minority Classes

Minority classes:

- `helicopter`
- `taxi`
- `train`
- `bicycle`
- `minibus`

Allowed variant transforms:

- Horizontal Flip
- Rotation
- Shift/Scale
- Perspective Transform
- Brightness/Contrast
- Coarse Dropout

After the normal quota reaches 70% of target, combine available or generated variants with Rain, Sun, and Night simulation as needed to fill the remaining 30% weather quota.

## Online Augmentation

Online augmentation is optional and happens only inside training batches.

| Model family | Recommended online augmentation |
|--------------|---------------------------------|
| ResNet-50 | MixUp and/or CutMix |
| Vision Transformer | MixUp and/or CutMix |
| YOLO-cls | Mosaic if supported by the training configuration |

Online augmentation must be disabled for validation, test, and deployment inference.
