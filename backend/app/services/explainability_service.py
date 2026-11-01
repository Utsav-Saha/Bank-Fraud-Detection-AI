"""SHAP explanations for compatible tree models. Kept separate from predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.preprocessing_service import align_features
from app.utils.model_loader import ModelNotAvailableError, get_registry

_EXPLAINER = None
_EXPLAINER_ERROR: str | None = None


def _tree_explainer():
    global _EXPLAINER, _EXPLAINER_ERROR
    if _EXPLAINER is not None or _EXPLAINER_ERROR:
        return _EXPLAINER
    registry = get_registry()
    model = registry.supervised.get("random_forest")
    if model is None:
        _EXPLAINER_ERROR = "random_forest.pkl is required for SHAP TreeExplainer."
        return None
    try:
        import shap

        _EXPLAINER = shap.TreeExplainer(model)
    except Exception as exc:
        _EXPLAINER_ERROR = f"Unable to initialize SHAP TreeExplainer: {exc}"
        _EXPLAINER = None
    return _EXPLAINER


def _fraud_shap_vector(shap_output: Any) -> np.ndarray:
    """Select the fraud-class SHAP vector across SHAP version differences."""
    values = getattr(shap_output, "values", shap_output)
    if isinstance(values, list):
        array = np.asarray(values[1] if len(values) > 1 else values[0])
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        return array
    if array.ndim == 2:
        # (n_features, n_classes) or (n_samples, n_features)
        if array.shape[0] == 1:
            return array[0]
        if array.shape[-1] == 2 and array.shape[0] != 2:
            return array[:, 1]
        return array[0]
    if array.ndim == 3:
        # (n_samples, n_features, n_classes)
        if array.shape[-1] == 2:
            return array[0, :, 1]
        return array[0, :, 0]
    raise RuntimeError(f"Unsupported SHAP value shape: {array.shape}")


def explain_transaction(
    payload: dict[str, float],
    *,
    already_scaled: bool = False,
    top_n: int = 10,
    model_name: str = "random_forest",
) -> dict[str, Any]:
    if model_name != "random_forest":
        raise ModelNotAvailableError(
            "SHAP explanations in this prototype are implemented for the saved Random Forest only."
        )
    registry = get_registry()
    explainer = _tree_explainer()
    if explainer is None:
        raise ModelNotAvailableError(_EXPLAINER_ERROR or "SHAP explainer is unavailable.")

    raw_frame = pd.DataFrame([payload], columns=registry.feature_columns)
    scaled = align_features(payload, already_scaled=already_scaled)
    shap_output = explainer.shap_values(scaled)
    fraud_values = _fraud_shap_vector(shap_output)

    contributions = []
    for feature, shap_value, raw_value, scaled_value in zip(
        registry.feature_columns,
        fraud_values,
        raw_frame.iloc[0].tolist(),
        scaled.iloc[0].tolist(),
    ):
        shap_float = float(shap_value)
        contributions.append(
            {
                "feature": feature,
                "shap_value": round(shap_float, 6),
                "input_value": float(raw_value),
                "scaled_value": float(scaled_value),
                "direction": "increases_fraud_score" if shap_float > 0 else "decreases_fraud_score",
                "note": f"Feature {feature} contributed to the model's prediction.",
            }
        )
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    global_importance = []
    importances = getattr(registry.supervised["random_forest"], "feature_importances_", None)
    if importances is not None:
        global_importance = [
            {"feature": feature, "importance": float(score)}
            for feature, score in sorted(
                zip(registry.feature_columns, importances),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
    return {
        "model": "random_forest",
        "explanation_type": "shap_tree_explainer",
        "class_explained": "fraud",
        "disclaimer": (
            "V1-V28 are anonymized PCA components from the Kaggle dataset. "
            "This explanation reports contribution magnitude only and does not assign real-world meaning to those features."
        ),
        "contributions": contributions[:top_n],
        "all_contributions": contributions,
        "global_feature_importance": global_importance[:15],
    }
