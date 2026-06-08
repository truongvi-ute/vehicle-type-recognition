from __future__ import annotations

import base64
import io
from typing import BinaryIO, Tuple

import numpy as np
from PIL import Image

from src.image_pipelines import apply_base_pipeline, apply_environment_pipeline


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
