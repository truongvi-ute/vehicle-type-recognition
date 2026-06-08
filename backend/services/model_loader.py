from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_for_inference  # noqa: E402
from utils.class_names import CLASS_NAMES  # noqa: E402


@dataclass(frozen=True)
class LoadedModel:
    name: str
    path: Path
    model: nn.Module
    device: torch.device


_MODEL_CACHE: Dict[str, LoadedModel] = {}


def _candidate_paths(models_dir: Path, model_name: Optional[str]) -> list[Path]:
    if model_name:
        raw = Path(model_name)
        names = [raw.name]
        if raw.suffix != ".pth":
            names.extend([f"{raw.name}.pth", f"{raw.name}_best.pth"])
        return [models_dir / name for name in names]

    return sorted(
        models_dir.glob("*.pth"),
        key=lambda path: (0 if path.name.endswith("_best.pth") else 1, path.name.lower()),
    )


def resolve_model_path(models_dir: str | Path, model_name: Optional[str] = None) -> Path:
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Models directory not found: {root}")

    for path in _candidate_paths(root, model_name):
        if path.is_file():
            return path

    if model_name:
        raise FileNotFoundError(
            f"Model '{model_name}' was not found in {root}. "
            "Expected a .pth checkpoint such as resnet50_best.pth."
        )

    raise FileNotFoundError(
        f"No .pth checkpoint found in {root}. Train a model first or copy a checkpoint into models/."
    )


def get_model(models_dir: str | Path, model_name: Optional[str] = None) -> LoadedModel:
    model_path = resolve_model_path(models_dir, model_name)
    cache_key = str(model_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_inference(
        checkpoint_path=str(model_path),
        num_classes=len(CLASS_NAMES),
        device=device,
    )
    loaded = LoadedModel(
        name=model_path.stem,
        path=model_path,
        model=model,
        device=device,
    )
    _MODEL_CACHE[cache_key] = loaded
    return loaded
