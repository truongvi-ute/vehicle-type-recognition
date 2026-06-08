from __future__ import annotations

import time
from typing import BinaryIO, Dict, List

from utils.class_names import CLASS_NAMES
from utils.preprocessing import image_to_data_url, preprocess_image, preprocess_preview_image


class InferenceError(ValueError):
    pass


def preview_pipeline_image(
    image_stream: BinaryIO,
    pipeline: str | None = None,
) -> Dict[str, object]:
    processed_image, selected_pipeline = preprocess_preview_image(
        image_stream,
        pipeline=pipeline,
    )
    return {
        "pipeline": selected_pipeline,
        "processed_image": image_to_data_url(processed_image),
    }


def predict_image(
    image_stream: BinaryIO,
    models_dir: str,
    model_name: str | None = None,
    pipeline: str | None = None,
    top_k: int = 3,
) -> Dict[str, object]:
    if top_k <= 0:
        raise InferenceError("top_k must be positive.")

    import torch
    from services.model_loader import get_model
    from ultralytics import YOLO

    started = time.perf_counter()
    loaded = get_model(models_dir=models_dir, model_name=model_name)
    image_tensor, processed_image, selected_pipeline = preprocess_image(
        image_stream,
        pipeline=pipeline,
    )

    predictions: List[Dict[str, object]] = []
    is_yolo = isinstance(loaded.model, YOLO)

    if is_yolo:
        results = loaded.model(processed_image, verbose=False)
        probs = results[0].probs
        topk_conf = probs.top5conf.tolist() if hasattr(probs, 'top5conf') else []
        topk_idx = probs.top5.tolist() if hasattr(probs, 'top5') else []

        if not topk_idx:
            probs_tensor = probs.data
            k = min(top_k, probs_tensor.numel())
            topk_conf_tensor, topk_idx_tensor = torch.topk(probs_tensor, k=k)
            topk_conf = topk_conf_tensor.tolist()
            topk_idx = topk_idx_tensor.tolist()

        for confidence, index in zip(topk_conf, topk_idx):
            predictions.append(
                {
                    "class_name": CLASS_NAMES[index],
                    "confidence": round(float(confidence), 6),
                }
            )
    else:
        image_tensor = image_tensor.to(loaded.device)
        with torch.no_grad():
            logits = loaded.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probabilities.numel(), len(CLASS_NAMES))
        confidences, indices = torch.topk(probabilities, k=k)

        for confidence, index in zip(confidences.tolist(), indices.tolist()):
            predictions.append(
                {
                    "class_name": CLASS_NAMES[index],
                    "confidence": round(float(confidence), 6),
                }
            )

    predictions = predictions[:top_k]
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {
        "model_name": loaded.name,
        "pipeline": selected_pipeline,
        "processed_image": image_to_data_url(processed_image),
        "predictions": predictions,
        "processing_time_ms": elapsed_ms,
    }
