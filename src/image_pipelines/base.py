from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = 224


def resize_with_aspect_ratio(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def reflective_pad_to_square(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    width, height = image.size
    if width == target_size and height == target_size:
        return image

    pad_w = target_size - width
    pad_h = target_size - height

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img_np = np.array(image)
    padded_np = cv2.copyMakeBorder(
        img_np, top, bottom, left, right, cv2.BORDER_REFLECT_101
    )
    return Image.fromarray(padded_np)


def apply_base_pipeline(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    image = image.convert("RGB")
    resized = resize_with_aspect_ratio(image, target_size=target_size)
    return reflective_pad_to_square(resized, target_size=target_size)
