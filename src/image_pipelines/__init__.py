from .base import TARGET_SIZE, apply_base_pipeline
from .night import apply_night_pipeline
from .rain import apply_rain_pipeline
from .sun import apply_sun_pipeline

SUPPORTED_PIPELINES = ["normal", "rain", "sun", "night"]


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
