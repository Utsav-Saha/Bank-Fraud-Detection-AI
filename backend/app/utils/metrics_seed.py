"""Load evaluated model metrics from existing CSV artifacts. Never invent scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app.extensions import db
from app.models.database_models import ModelMetadata

METRIC_CSV = "ml_model_comparison.csv"

COLUMN_MAP = {
    "Model": "model_name",
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1_score",
    "ROC-AUC": "roc_auc",
    "Training Time (sec)": "training_time_sec",
}


def seed_model_metadata_from_csv(metrics_dir: Path) -> int:
    """Insert comparison-CSV rows if the metadata table is empty.

    Returns the number of rows inserted. PR-AUC is left null because it is
    not present in the saved comparison file.
    """
    existing = db.session.execute(select(ModelMetadata.id).limit(1)).first()
    if existing is not None:
        return 0

    csv_path = metrics_dir / METRIC_CSV
    if not csv_path.is_file():
        return 0

    frame = pd.read_csv(csv_path)
    inserted = 0
    for _, row in frame.iterrows():
        name = str(row.get("Model", "")).strip()
        if not name:
            continue
        record = ModelMetadata(
            model_name=name,
            model_version="notebook-ml-algo",
            accuracy=_optional_float(row.get("Accuracy")),
            precision=_optional_float(row.get("Precision")),
            recall=_optional_float(row.get("Recall")),
            f1_score=_optional_float(row.get("F1 Score")),
            roc_auc=_optional_float(row.get("ROC-AUC")),
            pr_auc=None,
            training_time_sec=_optional_float(row.get("Training Time (sec)")),
            source_file=str(csv_path.name),
        )
        db.session.add(record)
        inserted += 1
    if inserted:
        db.session.commit()
    return inserted


def _optional_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
