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

- About **7,000 images/class**.
- Configurable per class.

Default environment distribution:

| Bucket | Ratio | Example for 7,000 images/class |
|--------|-------|--------------------------------|
| `normal` | 70% | 4,900 |
| `rain` | 10% | 700 |
| `sun` | 10% | 700 |
| `night` | 10% | 700 |

Total image count after augmentation depends on configured quotas.

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
- Fill environment buckets mostly from available train images.
- Generate only missing bucket samples if needed.

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

After variant generation, combine samples with Rain, Sun, and Night simulation as needed to fill bucket quotas.

## Online Augmentation

Online augmentation is optional and happens only inside training batches.

| Model family | Recommended online augmentation |
|--------------|---------------------------------|
| ResNet-50 | MixUp and/or CutMix |
| Vision Transformer | MixUp and/or CutMix |
| YOLO-cls | Mosaic if supported by the training configuration |

Online augmentation must be disabled for validation, test, and deployment inference.
