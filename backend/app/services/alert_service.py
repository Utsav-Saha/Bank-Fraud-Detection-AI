"""Persist predictions and manage alerts."""

from __future__ import annotations

from typing import Any

from app.extensions import db
from app.models.database_models import FraudAlert, InvestigationRecord, Transaction


def persist_prediction(result: dict[str, Any]) -> Transaction:
    existing = Transaction.query.filter_by(transaction_id=result["transaction_id"]).one_or_none()
    if existing is None:
        record = Transaction(
            transaction_id=result["transaction_id"],
            transaction_amount=float(result.get("transaction_amount_input") or 0.0),
            prediction=int(result["prediction"]),
            prediction_label=result["prediction_label"],
            fraud_probability=float(result["fraud_probability"]),
            risk_score=float(result["risk_score"]),
            risk_level=result["risk_level"],
            model_name=result["model"],
            anomaly_flag=int(result["anomaly_flag"]),
        )
        db.session.add(record)
    else:
        record = existing
        record.transaction_amount = float(result.get("transaction_amount_input") or 0.0)
        record.prediction = int(result["prediction"])
        record.prediction_label = result["prediction_label"]
        record.fraud_probability = float(result["fraud_probability"])
        record.risk_score = float(result["risk_score"])
        record.risk_level = result["risk_level"]
        record.model_name = result["model"]
        record.anomaly_flag = int(result["anomaly_flag"])

    for rule in result.get("triggered_rules") or []:
        db.session.add(
            FraudAlert(
                transaction_id=result["transaction_id"],
                alert_type=rule.get("rule_id") or "RULE",
                severity=rule.get("severity") or "INFO",
                description=rule.get("explanation") or rule.get("message") or "",
                review_status="NEW",
            )
        )
    db.session.commit()
    return record


def list_alerts(severity: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    query = FraudAlert.query.order_by(FraudAlert.id.desc())
    if severity:
        query = query.filter(FraudAlert.severity == severity.upper())
    if status:
        query = query.filter(FraudAlert.review_status == status)
    rows = query.limit(500).all()
    payload = []
    for row in rows:
        txn = Transaction.query.filter_by(transaction_id=row.transaction_id).one_or_none()
        payload.append(
            {
                "id": row.id,
                "transaction_id": row.transaction_id,
                "alert_type": row.alert_type,
                "severity": row.severity,
                "description": row.description,
                "review_status": row.review_status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "risk_score": txn.risk_score if txn else None,
                "risk_level": txn.risk_level if txn else None,
                "prediction_label": txn.prediction_label if txn else None,
            }
        )
    return payload


def patch_alert(alert_id: int, review_status: str, investigator_note: str = "") -> dict[str, Any] | None:
    alert = db.session.get(FraudAlert, alert_id)
    if alert is None:
        return None
    alert.review_status = review_status
    if investigator_note:
        note = InvestigationRecord(
            transaction_id=alert.transaction_id,
            investigator_note=investigator_note,
            review_status=review_status,
        )
        db.session.add(note)
        txn = Transaction.query.filter_by(transaction_id=alert.transaction_id).one_or_none()
        if txn is not None:
            txn.review_status = review_status
    db.session.commit()
    return {
        "id": alert.id,
        "transaction_id": alert.transaction_id,
        "review_status": alert.review_status,
    }


def save_investigation(transaction_id: str, investigator_note: str, review_status: str) -> dict[str, Any]:
    txn = Transaction.query.filter_by(transaction_id=transaction_id).one_or_none()
    if txn is None:
        raise ValueError(f"Unknown transaction_id: {transaction_id}")
    txn.review_status = review_status
    record = InvestigationRecord(
        transaction_id=transaction_id,
        investigator_note=investigator_note,
        review_status=review_status,
    )
    db.session.add(record)
    db.session.commit()
    return {
        "id": record.id,
        "transaction_id": transaction_id,
        "investigator_note": investigator_note,
        "review_status": review_status,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }
