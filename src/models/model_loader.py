from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

MODEL_FOLDER_MAP = {
    "assignment": "model_assignment",
    "AYBULabs": "model_aybu-labs",
    "AYBUExams": "model_aybu-exams",
}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_assessment_type(assessment_type: str) -> str:
    assessment_type = str(assessment_type or "").strip()

    aliases = {
   
        # new UI values
        "assignment": "assignment",
        "labs-aybu": "AYBULabs",
        "exams-aybu": "AYBUExams",
    }

    if assessment_type in MODEL_FOLDER_MAP:
        return assessment_type

    normalized = aliases.get(assessment_type.lower())
    if normalized:
        return normalized

    allowed = ", ".join([
        "assignment",
        "labs-aybu",
        "exams-aybu",

    ])
    raise ValueError(
        f"Unknown assessment_type: {assessment_type}. Allowed values: {allowed}"
    )
def _validate_required_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


@lru_cache(maxsize=None)
def _load_model_assets_cached(normalized_assessment_type: str) -> Dict[str, Any]:
    base_dir = _get_project_root()
    model_dir = base_dir / "saved_models" / MODEL_FOLDER_MAP[normalized_assessment_type]

    model_path = model_dir / "model.pkl"
    feature_order_path = model_dir / "feature_order.json"
    background_path = model_dir / "X_background.npy"

    _validate_required_file(model_path, "Model file")
    _validate_required_file(feature_order_path, "Feature order file")
    _validate_required_file(background_path, "Background file")

    model = joblib.load(model_path)

    try:
        clf = model[-1]
        if not hasattr(clf, "multi_class"):
            clf.multi_class = "auto"
    except Exception:
        pass

    with open(feature_order_path, "r", encoding="utf-8") as input_file:
        feature_order = json.load(input_file)

    if not isinstance(feature_order, list):
        raise ValueError(f"feature_order.json must contain a list. Got: {type(feature_order).__name__}")

    X_background = np.load(background_path)

    if X_background.ndim != 2:
        raise ValueError(f"X_background.npy must be 2D. Got shape: {X_background.shape}")

    return {
        "model": model,
        "feature_order": feature_order,
        "X_background": X_background,
        "model_dir": str(model_dir),
        "assessment_type": normalized_assessment_type,
    }


def load_model_assets(assessment_type: str) -> Dict[str, Any]:
    normalized_assessment_type = _normalize_assessment_type(assessment_type)
    return _load_model_assets_cached(normalized_assessment_type)


def clear_model_asset_cache() -> None:
    _load_model_assets_cached.cache_clear()
