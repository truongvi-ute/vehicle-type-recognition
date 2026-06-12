from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image


KERNEL_LENGTHS = (3, 5, 7)
ANGLE_RANGE = (0.0, 180.0)


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    if length < 3 or length % 2 == 0:
        raise ValueError("Motion blur kernel length must be an odd integer of at least 3.")

    kernel_mask = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    radius = center
    radians = np.deg2rad(angle)
    dx = np.cos(radians) * radius
    dy = np.sin(radians) * radius

    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel_mask, start, end, color=255, thickness=1, lineType=cv2.LINE_AA)

    kernel = kernel_mask.astype(np.float32) / 255.0
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        kernel[center, center] = 1.0
        kernel_sum = 1.0
    return kernel / kernel_sum


def apply_motion_blur_pipeline(image: Image.Image, seed: int = 42) -> Image.Image:
    """Apply reproducible linear motion blur while preserving image size."""
    rng = random.Random(seed)
    kernel_length = rng.choice(KERNEL_LENGTHS)
    angle = rng.uniform(*ANGLE_RANGE)
    kernel = _motion_kernel(kernel_length, angle)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.filter2D(
        rgb,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(blurred, mode="RGB")
