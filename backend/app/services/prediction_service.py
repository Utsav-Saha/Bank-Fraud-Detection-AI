"""Supervised prediction using cached project artifacts only."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import REQUIRED_FEATURES
from app.services.anomaly_service import score_anomaly
from app.services.preprocessing_service import align_features, align_frame
from app.services.risk_engine import RiskPolicy, calculate_risk
from app.services.rule_engine import apply_rules
from app.utils.helpers import make_transaction_id
from app.utils.model_loader import ModelNotAvailableError, get_registry


def _predict_proba(model_name: str, model: Any, features: pd.DataFrame) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        fraud_probability = float(np.asarray(proba)[0, 1])
        return 1, fraud_probability
    if model_name == "tensorflow_ann":
        raw = np.asarray(model.predict(features, verbose=0))
        fraud_probability = float(raw.reshape(-1)[0])
        return 1, fraud_probability
    if hasattr(model, "decision_function"):
        score = float(np.asarray(model.decision_function(features)).reshape(-1)[0])
        fraud_probability = float(1.0 / (1.0 + np.exp(-score)))
        return 1, fraud_probability
    raise ModelNotAvailableError(f"Model '{model_name}' does not expose a fraud probability.")


def predict_transaction(
    payload: dict[str, float],
    *,
    model_name: str = "random_forest",
    already_scaled: bool = False,
    threshold: float = 0.50,
    transaction_id: str | None = None,
    risk_policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    registry = get_registry()
    model = registry.require(model_name)
    started = time.perf_counter()
    scaled = align_features(payload, already_scaled=already_scaled)
    _, fraud_probability = _predict_proba(model_name, model, scaled)
    prediction = 1 if fraud_probability >= threshold else 0
    anomaly = score_anomaly(scaled)
    risk = calculate_risk(fraud_probability, policy=risk_policy)
    alerts = apply_rules(
        fraud_probability=fraud_probability,
        prediction=prediction,
        anomaly_flag=int(anomaly["anomaly_flag"]),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    amount = float(payload.get("Amount", 0.0))
    return {
        "transaction_id": transaction_id or make_transaction_id(),
        "model": model_name,
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(fraud_probability, 6),
        "classification_threshold": threshold,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "recommendation": risk["recommendation"],
        "anomaly_flag": anomaly["anomaly_flag"],
        "anomaly": anomaly,
        "alerts": [item["message"] for item in alerts],
        "triggered_rules": alerts,
        "inference_time_ms": elapsed_ms,
        "transaction_amount_input": amount,
        "already_scaled": already_scaled,
        "feature_order": registry.feature_columns or REQUIRED_FEATURES,
        "probability_notes": risk["probability_notes"],
        "risk_policy": risk["policy"],
    }


def predict_batch_frame(
    frame: pd.DataFrame,
    *,
    model_name: str = "random_forest",
    already_scaled: bool = False,
    threshold: float = 0.50,
    risk_policy: RiskPolicy | None = None,
) -> list[dict[str, Any]]:
    registry = get_registry()
    model = registry.require(model_name)
    results: list[dict[str, Any]] = []
    scaled_all = align_frame(frame, already_scaled=already_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(scaled_all))[:, 1]
    elif model_name == "tensorflow_ann":
        probabilities = np.asarray(model.predict(scaled_all, verbose=0)).reshape(-1)
    else:
        raise ModelNotAvailableError(f"Model '{model_name}' cannot score batches.")

    iso = registry.isolation_forest
    if iso is not None:
        iso_pred = np.asarray(iso.predict(scaled_all))
        iso_scores = np.asarray(iso.decision_function(scaled_all))
    else:
        iso_pred = np.ones(len(frame), dtype=int)
        iso_scores = np.zeros(len(frame), dtype=float)

    for index in range(len(frame)):
        started = time.perf_counter()
        fraud_probability = float(probabilities[index])
        prediction = 1 if fraud_probability >= threshold else 0
        anomaly_flag = 1 if int(iso_pred[index]) == -1 else 0
        anomaly = {
            "anomaly_flag": anomaly_flag,
            "anomaly_available": iso is not None,
            "raw_score": round(float(iso_scores[index]), 6) if iso is not None else None,
            "normalized_score": round(-float(iso_scores[index]), 6) if iso is not None else None,
            "note": "Anomaly flag indicates unusual structure versus the fitted baseline, not a confirmed fraud label.",
        }
        risk = calculate_risk(fraud_probability, policy=risk_policy)
        alerts = apply_rules(
            fraud_probability=fraud_probability,
            prediction=prediction,
            anomaly_flag=anomaly_flag,
        )
        txn_id = None
        if "transaction_id" in frame.columns:
            txn_id = str(frame.iloc[index]["transaction_id"])
        results.append(
            {
                "row_index": int(index),
                "transaction_id": txn_id or make_transaction_id(),
                "model": model_name,
                "prediction": prediction,
                "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
                "fraud_probability": round(fraud_probability, 6),
                "classification_threshold": threshold,
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "recommendation": risk["recommendation"],
                "anomaly_flag": anomaly_flag,
                "anomaly": anomaly,
                "alerts": [item["message"] for item in alerts],
                "triggered_rules": alerts,
                "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "transaction_amount_input": float(frame.iloc[index]["Amount"]),
                "already_scaled": already_scaled,
                "valid": True,
                "error": None,
            }
        )
    return results
