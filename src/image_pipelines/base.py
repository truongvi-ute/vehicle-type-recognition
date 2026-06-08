from __future__ import annotations

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
