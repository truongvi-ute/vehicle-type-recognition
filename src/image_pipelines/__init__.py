from .base import TARGET_SIZE, apply_base_pipeline
from .gaussian_blur import apply_gaussian_blur_pipeline
from .motion_blur import apply_motion_blur_pipeline
from .night import apply_night_pipeline
from .rain import apply_rain_pipeline
from .sun import apply_sun_pipeline
from .unsharp_mask import apply_unsharp_mask_pipeline

SUPPORTED_PIPELINES = ["normal", "rain", "sun", "night"]
SUPPORTED_V2_PIPELINES = ["normal", "gaussian_blur", "motion_blur", "unsharp_mask"]


def apply_environment_pipeline(image, pipeline: str | None, seed: int = 42):
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
        return apply_rain_pipeline(image, seed=seed), selected
    if selected == "sun":
        return apply_sun_pipeline(image, seed=seed), selected
    return apply_night_pipeline(image, seed=seed), selected


def apply_v2_pipeline(image, pipeline: str | None, seed: int = 42):
    selected = (pipeline or "normal").strip().lower()
    if selected in {"base", "none"}:
        selected = "normal"
    if selected not in SUPPORTED_V2_PIPELINES:
        raise ValueError(
            f"Unsupported V2 pipeline '{pipeline}'. "
            f"Choose one of: {', '.join(SUPPORTED_V2_PIPELINES)}."
        )

    rgb_image = image.convert("RGB")
    if selected == "normal":
        return rgb_image, selected
    if selected == "gaussian_blur":
        return apply_gaussian_blur_pipeline(rgb_image, seed=seed), selected
    if selected == "motion_blur":
        return apply_motion_blur_pipeline(rgb_image, seed=seed), selected
    return apply_unsharp_mask_pipeline(rgb_image, seed=seed), selected
