from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image


KERNEL_SIZES = (3, 5)
SIGMA_RANGE = (0.4, 1.4)


def apply_gaussian_blur_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    """Apply reproducible, mild Gaussian blur while preserving image size."""
    rng = random.Random(seed)
    kernel_size = rng.choice(KERNEL_SIZES)
    sigma = rng.uniform(*SIGMA_RANGE)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(
        rgb,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(blurred, mode="RGB")
