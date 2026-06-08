from __future__ import annotations

import numpy as np
from PIL import Image


def apply_night_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.power(array, 1.75)
    array[..., 0] *= 0.72
    array[..., 1] *= 0.78
    array[..., 2] *= 0.95
    noise = rng.normal(0, 0.025, array.shape)
    array = np.clip(array + noise, 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))
