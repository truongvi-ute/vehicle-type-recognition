from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image


KERNEL_SIZES = (3, 5)
SIGMA_RANGE = (0.5, 1.2)
AMOUNT_RANGE = (0.3, 0.8)
THRESHOLD_RANGE = (0, 5)


def apply_unsharp_mask_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    """Apply reproducible, noise-aware unsharp masking without changing size."""
    rng = random.Random(seed)
    kernel_size = rng.choice(KERNEL_SIZES)
    sigma = rng.uniform(*SIGMA_RANGE)
    amount = rng.uniform(*AMOUNT_RANGE)
    threshold = rng.randint(*THRESHOLD_RANGE)

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = cv2.GaussianBlur(
        rgb,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    detail = rgb - blurred

    if threshold > 0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)

    sharpened = np.clip(rgb + amount * detail, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(sharpened, mode="RGB")
