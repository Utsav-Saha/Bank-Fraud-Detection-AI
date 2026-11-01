"""Isolation Forest scoring. An anomaly is not confirmed fraud."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.model_loader import get_registry


def score_anomaly(scaled_features: pd.DataFrame) -> dict[str, Any]:
    registry = get_registry()
    model = registry.isolation_forest
    if model is None:
        return {
            "anomaly_flag": 0,
            "anomaly_available": False,
            "raw_score": None,
            "normalized_score": None,
            "note": "Isolation Forest artifact is not loaded. Anomaly scoring skipped.",
        }

    raw_pred = int(model.predict(scaled_features)[0])
    raw_score = float(model.decision_function(scaled_features)[0])
    is_anomaly = raw_pred == -1
    return {
        "anomaly_flag": 1 if is_anomaly else 0,
        "anomaly_available": True,
        "raw_score": round(raw_score, 6),
        "normalized_score": round(-raw_score, 6),
        "note": "Anomaly flag indicates unusual structure versus the fitted baseline, not a confirmed fraud label.",
    }
