from __future__ import annotations

import base64
import io
import random
from typing import BinaryIO, Tuple

import numpy as np
from PIL import Image, ImageFilter


TARGET_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUPPORTED_PIPELINES = ["normal", "rain", "sun", "night"]


def resize_with_aspect_ratio(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def zero_pad_to_square(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - image.size[0]) // 2
    top = (target_size - image.size[1]) // 2
    canvas.paste(image, (left, top))
    return canvas


def apply_base_pipeline(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    image = image.convert("RGB")
    resized = resize_with_aspect_ratio(image, target_size=target_size)
    return zero_pad_to_square(resized, target_size=target_size)


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


def apply_environment_pipeline(image: Image.Image, pipeline: str | None) -> Tuple[Image.Image, str]:
    selected = (pipeline or "normal").strip().lower()
    if selected in {"base", "none"}:
        selected = "normal"
    if selected not in SUPPORTED_PIPELINES:
        raise ValueError(
            f"Unsupported pipeline '{pipeline}'. Choose one of: {', '.join(SUPPORTED_PIPELINES)}."
        )

    if selected == "normal":
        return image, selected
    if selected == "rain":
        return apply_rain_pipeline(image), selected
    if selected == "sun":
        return apply_sun_pipeline(image), selected
    return apply_night_pipeline(image), selected


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preprocess_image(image_stream: BinaryIO, pipeline: str | None = None) -> Tuple[torch.Tensor, Image.Image, str]:
    image = Image.open(image_stream)
    base = apply_base_pipeline(image)
    processed, selected_pipeline = apply_environment_pipeline(base, pipeline)
    return image_to_tensor(processed), processed, selected_pipeline


def preprocess_preview_image(image_stream: BinaryIO, pipeline: str | None = None) -> Tuple[Image.Image, str]:
    image = Image.open(image_stream)
    base = apply_base_pipeline(image)
    return apply_environment_pipeline(base, pipeline)
