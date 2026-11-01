"""Relational schema for prediction history, alerts, investigations, and metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(db.Model):
    __tablename__ = "transactions"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    transaction_amount = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Integer, nullable=False, index=True)
    fraud_probability = db.Column(db.Float, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(32), nullable=False, index=True)
    model_name = db.Column(db.String(64), nullable=False, index=True)
    anomaly_flag = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    alerts = db.relationship(
        "FraudAlert",
        back_populates="transaction",
        cascade="all, delete-orphan",
        lazy="select",
    )
    investigations = db.relationship(
        "InvestigationRecord",
        back_populates="transaction",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        db.Index("ix_transactions_risk_created", "risk_level", "created_at"),
    )


class FraudAlert(db.Model):
    __tablename__ = "fraud_alerts"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(
        db.String(64),
        db.ForeignKey("transactions.transaction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alert_type = db.Column(db.String(64), nullable=False)
    severity = db.Column(db.String(32), nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    review_status = db.Column(db.String(32), nullable=False, default="NEW", index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    transaction = db.relationship("Transaction", back_populates="alerts")


class InvestigationRecord(db.Model):
    __tablename__ = "investigation_records"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(
        db.String(64),
        db.ForeignKey("transactions.transaction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    investigator_note = db.Column(db.Text, nullable=False, default="")
    review_status = db.Column(db.String(32), nullable=False, default="Pending", index=True)
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    transaction = db.relationship("Transaction", back_populates="investigations")


class ModelMetadata(db.Model):
    __tablename__ = "model_metadata"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(128), nullable=False, unique=True, index=True)
    model_version = db.Column(db.String(64), nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    roc_auc = db.Column(db.Float, nullable=True)
    pr_auc = db.Column(db.Float, nullable=True)
    training_time_sec = db.Column(db.Float, nullable=True)
    source_file = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
