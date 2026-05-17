from typing import Any, Dict, List, Optional

import numpy as np
import shap
from sklearn.pipeline import Pipeline

from src.explainability.group_mapper import map_any_feature_to_group


def build_feature_names(handcrafted_feature_order: List[str], embedding_size: int) -> List[str]:
    print("\n========== STEP 1: BUILD FEATURE NAMES ==========")
    print("Number of handcrafted features:", len(handcrafted_feature_order))
    print("Embedding size:", embedding_size)

    embedding_names = [f"emb_{i}" for i in range(embedding_size)]
    feature_names = list(handcrafted_feature_order) + embedding_names

    print("Total feature names built:", len(feature_names))
    print("First 10 Software engineering feature names:", handcrafted_feature_order[:10])
    print("First 10 embedding names:", embedding_names[:10])
    print("First 20 final feature names:", feature_names[:20])

    return feature_names


def _extract_shap_for_sample(shap_values: Any, predicted_label: str, sample_index: int = 0) -> np.ndarray:
    print("\n========== STEP 4: EXTRACT SHAP VECTOR FOR ONE SAMPLE ==========")
    # Always explain the AI class so the sign meaning stays stable:
    # positive SHAP = supports AI
    # negative SHAP = supports Human
    normalized_label = str(predicted_label).strip().lower()
    class_idx = 1

    print("Predicted label:", predicted_label)
    print("Normalized label:", normalized_label)
    print("Selected class index:", class_idx)
    print("Sample index:", sample_index)
    print("Raw SHAP python type:", type(shap_values))

    if isinstance(shap_values, list):
        print("SHAP is a list.")
        print("Number of list items:", len(shap_values))
        print("List item shapes:", [np.array(x).shape for x in shap_values])

        if len(shap_values) > class_idx:
            arr = np.array(shap_values[class_idx], dtype=np.float32)
            print(f"Using class_idx={class_idx}")
        else:
            arr = np.array(shap_values[0], dtype=np.float32)
            print("class_idx out of range, using first SHAP array")

        print("Chosen SHAP array shape:", arr.shape)

        if arr.ndim == 2:
            result = arr[sample_index]
            print("Returning 2D array row. Result shape:", result.shape)
            print("Max abs SHAP in extracted vector:", float(np.max(np.abs(result))))
            return result

        raise ValueError(f"Unexpected list-based SHAP shape: {arr.shape}")

    arr = np.array(shap_values, dtype=np.float32)
    print("SHAP converted to ndarray shape:", arr.shape)
    print("SHAP ndim:", arr.ndim)

    if arr.ndim == 3:
        if arr.shape[-1] > class_idx:
            result = arr[sample_index, :, class_idx]
            print("Using 3D format with selected class. Result shape:", result.shape)
        else:
            result = arr[sample_index, :, 0]
            print("3D format but class_idx out of range, using class 0. Result shape:", result.shape)

        print("Max abs SHAP in extracted vector:", float(np.max(np.abs(result))))
        return result

    if arr.ndim == 2:
        result = arr[sample_index]
        print("Using 2D format. Result shape:", result.shape)
        print("Max abs SHAP in extracted vector:", float(np.max(np.abs(result))))
        return result

    if arr.ndim == 1:
        print("Using 1D SHAP vector directly.")
        print("Max abs SHAP in extracted vector:", float(np.max(np.abs(arr))))
        return arr

    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


def _compute_shap_for_pipeline_logreg(model: Pipeline, X_sample: np.ndarray, background_data: Optional[np.ndarray] = None):
    print("\n========== STEP 2A: PIPELINE LOGISTIC BRANCH ==========")
    print("Model type:", type(model))
    print("Pipeline steps:", list(model.named_steps.keys()))
    print("X_sample shape:", X_sample.shape)

    scaler = model.named_steps.get("standardscaler")
    lr_model = model.named_steps.get("logisticregression")

    print("Scaler found:", scaler is not None)
    print("LogisticRegression found:", lr_model is not None)

    if lr_model is None:
        raise ValueError("Pipeline does not contain a 'logisticregression' step.")

    if background_data is None:
        raise ValueError("background_data is None. You must load and pass X_background.npy for the selected model.")

    print("background_data shape before transform:", background_data.shape)

    if scaler is not None:
        X_sample_transformed = scaler.transform(X_sample)
        background_transformed = scaler.transform(background_data)
        print("Applied StandardScaler.")
    else:
        X_sample_transformed = X_sample
        background_transformed = background_data
        print("No scaler found, using raw features.")

    print("X_sample_transformed shape:", X_sample_transformed.shape)
    print("background_transformed shape:", background_transformed.shape)

    explainer = shap.LinearExplainer(lr_model, background_transformed)
    print("Created shap.LinearExplainer")

    shap_values = explainer.shap_values(X_sample_transformed)
    print("Computed SHAP values")

    if isinstance(shap_values, list):
        print("Raw SHAP list shapes:", [np.array(x).shape for x in shap_values])
    else:
        print("Raw SHAP array shape:", np.array(shap_values).shape)

    return shap_values


def compute_shap_values_for_sample(model, X_sample: np.ndarray, background_data: Optional[np.ndarray] = None):
    print("\n========== STEP 2: COMPUTE SHAP VALUES ==========")
    print("Model type:", type(model))
    print("Model class name:", model.__class__.__name__)
    print("X_sample shape:", X_sample.shape)
    print("background_data is None:", background_data is None)
    if background_data is not None:
        print("background_data shape:", background_data.shape)

    if isinstance(model, Pipeline):
        step_names = {name.lower() for name in model.named_steps.keys()}
        print("Detected sklearn Pipeline")
        print("Pipeline step names:", step_names)

        if "logisticregression" in step_names:
            print("Routing to pipeline logistic regression handler")
            return _compute_shap_for_pipeline_logreg(model=model, X_sample=X_sample, background_data=background_data)

    model_name = model.__class__.__name__.lower()
    print("Non-pipeline model name used for routing:", model_name)

    if any(name in model_name for name in ["forest", "tree", "xgb", "lgbm", "boost"]):
        print("Routing to TreeExplainer")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values

    if "logisticregression" in model_name or "linear" in model_name:
        print("Routing to raw LinearExplainer")

        if background_data is None:
            raise ValueError("background_data is None. You must load and pass X_background.npy for the selected model.")

        explainer = shap.LinearExplainer(model, background_data)
        shap_values = explainer.shap_values(X_sample)
        return shap_values

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model type {model.__class__.__name__} is not supported by the fallback SHAP explainer because it does not have predict_proba()."
        )

    print("Routing to KernelExplainer fallback")

    if background_data is None:
        raise ValueError("background_data is None. You must load and pass X_background.npy for the selected model.")

    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(X_sample)
    return shap_values

def summarize_top_shap_features(shap_values, feature_names: List[str], predicted_label: str, top_k: int = 30) -> List[Dict[str, Any]]:
    print("\n========== STEP 5: SUMMARIZE TOP SHAP FEATURES ==========")
    print("Number of feature names:", len(feature_names))
    print("Requested top_k:", top_k)

    shap_vector = _extract_shap_for_sample(
        shap_values=shap_values,
        predicted_label=predicted_label,
        sample_index=0
    )

    if len(shap_vector) != len(feature_names):
        raise ValueError(
            f"Feature name count ({len(feature_names)}) does not match SHAP vector size ({len(shap_vector)})."
        )

    ranked_idx = np.argsort(np.abs(shap_vector))[::-1]

    positive_idx = [idx for idx in ranked_idx if float(shap_vector[idx]) > 0]
    negative_idx = [idx for idx in ranked_idx if float(shap_vector[idx]) < 0]

    # Keep a raw pool from both sides.
    # Later, app.py decides whether to show only predicted-class features
    # or mixed AI/Human features when confidence is low.
    selected_idx = positive_idx[:top_k] + negative_idx[:top_k]
    selected_idx = sorted(
        selected_idx,
        key=lambda idx: abs(float(shap_vector[idx])),
        reverse=True
    )

    results = []

    for idx in selected_idx:
        feature_name = feature_names[idx]
        shap_value = float(shap_vector[idx])

        mapped = map_any_feature_to_group(feature_name, predicted_label)
        group = mapped.get("group")

        if group in [None, "", "NULL", "null", "UNMAPPED", "nan"]:
            continue

        direction = "supports_ai" if shap_value > 0 else "supports_human"

        results.append({
            "feature": feature_name,
            "feature_type": mapped.get("feature_type"),
            "group": group,
            "shap_value": shap_value,
            "direction": direction,

            "group_corr_sum": mapped.get("group_corr_sum"),
            "n_pairs": mapped.get("n_pairs"),
            "max_corr": mapped.get("max_corr"),
            "group_rank": mapped.get("group_rank"),
            "group_support": mapped.get("group_support"),
        })

    print("Final number of kept top features:", len(results))
    return results


def _is_manual_feature_type(feature_type: Optional[str]) -> bool:
    normalized = str(feature_type or "").strip().lower()
    return normalized in {"manual", "software_engineering_features", "software engineering features"}


def aggregate_groups(top_features: List[Dict[str, Any]], predicted_label: str) -> List[Dict[str, Any]]:
    print("\n========== STEP 6: AGGREGATE GROUPS ==========")
    print("Number of input top_features:", len(top_features))
    print("Predicted label for grouping:", predicted_label)

    normalized_label = str(predicted_label).strip().lower()
    is_ai_prediction = normalized_label in {"ai-generated", "ai", "1"}

    group_scores: Dict[str, Dict[str, Any]] = {}

    for item in top_features:
        group = item["group"] if item["group"] else "UNMAPPED"
        shap_value = float(item["shap_value"])
        feature_type = item.get("feature_type")

        if group not in group_scores:
            group_scores[group] = {
                "group": group,
                "group_support_score": 0.0,
                "count": 0,
                "manual_count": 0,
                "embedding_count": 0,
            }

        group_scores[group]["count"] += 1

        if _is_manual_feature_type(feature_type):
            group_scores[group]["manual_count"] += 1
        elif str(feature_type).strip().lower() == "embedding":
            group_scores[group]["embedding_count"] += 1

        if is_ai_prediction and shap_value > 0:
            group_scores[group]["group_support_score"] += shap_value
        elif not is_ai_prediction and shap_value < 0:
            group_scores[group]["group_support_score"] += abs(shap_value)

    grouped = sorted(group_scores.values(), key=lambda x: x["group_support_score"], reverse=True)
    return grouped


def explain_prediction(
    model,
    X_sample: np.ndarray,
    handcrafted_feature_order: List[str],
    predicted_label: str,
    background_data: Optional[np.ndarray] = None,
    top_k: int = 30,
) -> Dict[str, Any]:
    print("\n========== STEP 0: START EXPLAIN PREDICTION ==========")

    if X_sample.ndim != 2:
        raise ValueError(f"X_sample must be 2D, got shape {X_sample.shape}")

    total_features = X_sample.shape[1]
    handcrafted_count = len(handcrafted_feature_order)
    embedding_size = total_features - handcrafted_count

    if embedding_size < 0:
        raise ValueError(
            f"Total feature count ({total_features}) is smaller than handcrafted feature count ({handcrafted_count})."
        )

    feature_names = build_feature_names(handcrafted_feature_order, embedding_size)

    shap_values = compute_shap_values_for_sample(model=model, X_sample=X_sample, background_data=background_data)

    top_features = summarize_top_shap_features(
        shap_values=shap_values,
        feature_names=feature_names,
        predicted_label=predicted_label,
        top_k=top_k,
    )

    grouped_influences = aggregate_groups(top_features, predicted_label)

    return {
        "top_features": top_features,
        "grouped_influences": grouped_influences,
        "feature_names": feature_names,
    }
