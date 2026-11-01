"""Service health and model-artifact inventory."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify
from sqlalchemy import text

from app.extensions import db
from app.utils.model_loader import inspect_artifacts

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    models_dir = current_app.config["MODELS_DIR"]
    artifacts = inspect_artifacts(models_dir)
    missing_required = [item.name for item in artifacts if item.required and not item.available]

    database_status = "disconnected"
    try:
        db.session.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception:
        current_app.logger.exception("Health check could not reach the database.")
        database_status = "error"

    overall = "ok" if database_status == "connected" and not missing_required else "degraded"

    payload = {
        "status": overall,
        "service": "banking-fraud-detection-api",
        "environment": current_app.config.get("ENV_NAME", "development"),
        "database": database_status,
        "models_dir": str(models_dir),
        "models": [item.as_dict() for item in artifacts],
        "missing_required": missing_required,
    }
    status_code = 200 if overall == "ok" else 503
    return jsonify(payload), status_code
