from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

from .base import TARGET_SIZE


def apply_rain_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    rng = random.Random(seed)
    array = np.array(image, dtype=np.float32)

    for _ in range(120):
        x = rng.randint(0, TARGET_SIZE - 1)
        y = rng.randint(0, TARGET_SIZE - 1)
        length = rng.randint(5, 12)
        for offset in range(length):
            yy = min(TARGET_SIZE - 1, y + offset)
            xx = min(TARGET_SIZE - 1, x + max(0, offset // 4))
            array[yy, xx] = np.maximum(array[yy, xx], np.array([190, 190, 190]))

    array = array * 0.88 - 8
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
