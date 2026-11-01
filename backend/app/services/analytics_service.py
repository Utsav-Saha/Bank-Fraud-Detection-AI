"""Operational analytics from persisted predictions and saved evaluation files."""

from __future__ import annotations

from typing import Any

import pandas as pd
from flask import current_app

from app.models.database_models import FraudAlert, ModelMetadata, Transaction


def seed_model_metadata() -> None:
    csv_path = current_app.config["METRICS_CSV"]
    if not csv_path.exists():
        return
    frame = pd.read_csv(csv_path)
    name_map = {
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
    }
    for _, row in frame.iterrows():
        source_name = str(row.get("Model", "")).strip()
        model_name = name_map.get(source_name, source_name.lower().replace(" ", "_"))
        existing = ModelMetadata.query.filter_by(model_name=model_name).one_or_none()
        payload = dict(
            model_version="notebook_ml_algo",
            accuracy=_maybe_float(row.get("Accuracy")),
            precision=_maybe_float(row.get("Precision")),
            recall=_maybe_float(row.get("Recall")),
            f1_score=_maybe_float(row.get("F1 Score")),
            roc_auc=_maybe_float(row.get("ROC-AUC")),
        )
        if existing is None:
            current_app.extensions["sqlalchemy"].session.add(  # type: ignore[index]
                ModelMetadata(model_name=model_name, **payload)
            )
        else:
            for key, value in payload.items():
                setattr(existing, key, value)
    from app.extensions import db

    db.session.commit()


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def comparison_payload() -> dict[str, Any]:
    csv_path = current_app.config["METRICS_CSV"]
    rows = []
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        rows = frame.to_dict(orient="records")
    db_rows = [
        {
            "model_name": item.model_name,
            "model_version": item.model_version,
            "accuracy": item.accuracy,
            "precision": item.precision,
            "recall": item.recall,
            "f1_score": item.f1_score,
            "roc_auc": item.roc_auc,
        }
        for item in ModelMetadata.query.order_by(ModelMetadata.id).all()
    ]
    return {
        "source_file": str(csv_path) if csv_path.exists() else None,
        "note": "Metrics are copied from outputs/metrics/ml_model_comparison.csv. No values are invented.",
        "training_note": (
            "Classifiers were trained on StandardScaler features after SMOTE oversampling of the training split. "
            "Reported test metrics use the original imbalanced test distribution. "
            "No TensorFlow ANN metrics file exists in this workspace."
        ),
        "evaluation_csv": rows,
        "database_records": db_rows,
        "pr_auc": None,
        "pr_auc_note": "PR-AUC was not stored in the saved comparison CSV.",
    }


def summary_payload() -> dict[str, Any]:
    total = Transaction.query.count()
    predicted_fraud = Transaction.query.filter(Transaction.prediction == 1).count()
    confirmed = Transaction.query.filter(Transaction.review_status == "CONFIRMED_FRAUD").count()
    high_risk = Transaction.query.filter(Transaction.risk_score > 60).count()
    avg_risk = Transaction.query.with_entities(Transaction.risk_score).all()
    average = round(sum(item[0] for item in avg_risk) / len(avg_risk), 2) if avg_risk else 0.0
    recent = (
        Transaction.query.order_by(Transaction.id.desc())
        .limit(15)
        .all()
    )
    alerts = FraudAlert.query.order_by(FraudAlert.id.desc()).limit(8).all()
    distribution = {
        level: Transaction.query.filter_by(risk_level=level).count()
        for level in ("Low", "Medium", "High", "Critical")
    }
    return {
        "total_analyzed_transactions": total,
        "predicted_fraud_cases": predicted_fraud,
        "confirmed_fraud_cases": confirmed,
        "fraud_prediction_rate": round((predicted_fraud / total), 6) if total else 0.0,
        "high_risk_transactions": high_risk,
        "average_risk_score": average,
        "risk_distribution": distribution,
        "distinction": "predicted_fraud_cases are model outputs; confirmed_fraud_cases require analyst review.",
        "recent_transactions": [_txn_dict(item) for item in recent],
        "recent_alerts": [
            {
                "id": alert.id,
                "transaction_id": alert.transaction_id,
                "severity": alert.severity,
                "description": alert.description,
                "review_status": alert.review_status,
            }
            for alert in alerts
        ],
    }


def _txn_dict(item: Transaction) -> dict[str, Any]:
    return {
        "transaction_id": item.transaction_id,
        "transaction_amount": item.transaction_amount,
        "prediction": item.prediction,
        "prediction_label": item.prediction_label,
        "fraud_probability": item.fraud_probability,
        "risk_score": item.risk_score,
        "risk_level": item.risk_level,
        "model_name": item.model_name,
        "anomaly_flag": item.anomaly_flag,
        "review_status": item.review_status,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


def recent_high_risk(limit: int = 25) -> list[dict[str, Any]]:
    rows = (
        Transaction.query.filter(Transaction.risk_score > 60)
        .order_by(Transaction.risk_score.desc())
        .limit(limit)
        .all()
    )
    return [_txn_dict(item) for item in rows]
