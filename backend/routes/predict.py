from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request
from PIL import UnidentifiedImageError

from services.inference_service import InferenceError, predict_image, preview_pipeline_image


predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict() -> tuple[object, int]:
    if "image" not in request.files:
        return jsonify({"error": "Missing multipart/form-data field: image"}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({"error": "No image file was uploaded."}), 400

    model_name = request.form.get("model_name") or None
    pipeline = request.form.get("pipeline") or "normal"

    try:
        result = predict_image(
            image_stream=image_file.stream,
            models_dir=current_app.config["MODELS_DIR"],
            model_name=model_name,
            pipeline=pipeline,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400
    except InferenceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result), 200


@predict_bp.post("/preprocess")
def preprocess_preview() -> tuple[object, int]:
    if "image" not in request.files:
        return jsonify({"error": "Missing multipart/form-data field: image"}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({"error": "No image file was uploaded."}), 400

    pipeline = request.form.get("pipeline") or "normal"

    try:
        result = preview_pipeline_image(
            image_stream=image_file.stream,
            pipeline=pipeline,
        )
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception("Preprocess preview failed")
        return jsonify({"error": f"Preprocess preview failed: {exc}"}), 500

    return jsonify(result), 200
