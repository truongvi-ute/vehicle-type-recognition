from __future__ import annotations

import random

import numpy as np
from PIL import Image

from .base import TARGET_SIZE


def apply_sun_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    rng = random.Random(seed)
    array = np.array(image, dtype=np.float32)
    yy, xx = np.mgrid[0:TARGET_SIZE, 0:TARGET_SIZE]
    center_x = rng.randint(46, 178)
    center_y = rng.randint(32, 126)
    radius = rng.randint(36, 64)
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    strength = np.clip(1 - distance / radius, 0, 1)[..., None]
    flare = np.array([255, 238, 190], dtype=np.float32)
    array = array * (1 + 0.12 * strength) + flare * (0.38 * strength)
    array = array * 1.08 + 8
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
