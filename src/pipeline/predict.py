from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from src.embeddings.embedding_extractor import get_codebert_embedding
from src.explainability.shap_explainer import explain_prediction
from src.features.feature_extractor import CodeMetricsExtractor
from src.models.model_loader import load_model_assets


def clean_numeric_value(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        return float(value)
    except Exception:
        return 0.0


def extract_manual_features(code: str, feature_order: Iterable[str]) -> np.ndarray:
    extractor = CodeMetricsExtractor(code)
    metrics = extractor.metrics

    if extractor.tree is None:
        raise ValueError("Manual feature extraction failed: the input is not valid Python syntax.")
    if extractor.tokens is None:
        raise ValueError("Manual feature extraction failed: tokenization failed.")
    if extractor.radon_raw_metrics is None:
        raise ValueError("Manual feature extraction failed: Radon could not analyze the code.")
    if not metrics:
        raise ValueError("Manual feature extraction failed: no metrics were produced.")

    vector = [clean_numeric_value(metrics.get(feature_name, 0.0)) for feature_name in feature_order]
    return np.array(vector, dtype=np.float32)


def build_feature_vector(code: str, feature_order: Iterable[str]) -> np.ndarray:
    manual_features = extract_manual_features(code, feature_order)
    embedding = np.array(get_codebert_embedding(code), dtype=np.float32)

    final_vector = np.concatenate([manual_features, embedding], axis=0)
    return final_vector.reshape(1, -1)


def map_prediction_to_label(prediction: Any) -> str:
    label_map = {
        0: "Human-written",
        1: "AI-generated",
        "0": "Human-written",
        "1": "AI-generated",
    }
    return label_map.get(prediction, str(prediction))


def round_floats(obj: Any, decimals: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, np.floating):
        return round(float(obj), decimals)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {key: round_floats(value, decimals) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value, decimals) for value in obj]
    return obj


def _validate_input_vector(X: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"Model input must be 2D, got shape {X.shape}")
    if X.shape[0] != 1:
        raise ValueError(f"Prediction expects a single sample, got shape {X.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Model input contains invalid numeric values.")


def predict_code(code_text: str, assessment_type: str) -> Dict[str, Any]:
    code_text = str(code_text or "").strip()
    if not code_text:
        raise ValueError("No code was provided.")

    assets = load_model_assets(assessment_type)
    model = assets["model"]
    feature_order: List[str] = assets["feature_order"]

    X = build_feature_vector(code_text, feature_order)
    _validate_input_vector(X)

    prediction = model.predict(X)[0]
    label = map_prediction_to_label(prediction)

    predicted_class_confidence = 0.0
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        predicted_class_confidence = float(np.max(probabilities))

    explanation = explain_prediction(
        model=model,
        X_sample=X,
        handcrafted_feature_order=feature_order,
        predicted_label=label,
        background_data=assets.get("X_background"),
        top_k=30,
    )

    result = {
        "assessment_type": assessment_type,
        "label": label,
        "predicted_class_confidence": predicted_class_confidence,
        "prediction": str(prediction),
        "model_used": str(assets["model_dir"]),
        "num_manual_features": len(feature_order),
        "total_input_size": int(X.shape[1]),
        "probabilities": probabilities.tolist() if probabilities is not None else None,
        "top_features": explanation["top_features"],
        "grouped_influences": explanation["grouped_influences"],
    }

    return round_floats(result, decimals=4)
