"""Reconstruct the original StandardScaler transform. Never resample incoming requests."""

from __future__ import annotations

import pandas as pd

from app.models.schemas import REQUIRED_FEATURES
from app.utils.model_loader import ModelNotAvailableError, get_registry


def align_features(payload: dict, already_scaled: bool = False) -> pd.DataFrame:
    registry = get_registry()
    columns = registry.feature_columns or REQUIRED_FEATURES
    frame = pd.DataFrame([{name: float(payload[name]) for name in columns}], columns=columns)
    if already_scaled:
        return frame
    if registry.scaler is None:
        raise ModelNotAvailableError("scaler.pkl is required to transform raw transaction features.")
    scaled = registry.scaler.transform(frame)
    return pd.DataFrame(scaled, columns=columns)


def align_frame(frame: pd.DataFrame, already_scaled: bool = False) -> pd.DataFrame:
    registry = get_registry()
    columns = registry.feature_columns or REQUIRED_FEATURES
    ordered = frame[columns].astype(float)
    if already_scaled:
        return ordered
    if registry.scaler is None:
        raise ModelNotAvailableError("scaler.pkl is required to transform raw transaction features.")
    return pd.DataFrame(registry.scaler.transform(ordered), columns=columns)


def infer_already_scaled(frame: pd.DataFrame) -> bool:
    """Heuristic used only as a warning, never as silent substitution.

    Processed training/test files store StandardScaler output. Raw Kaggle Time is
    seconds from the first transaction (typically thousands), while scaled Time
    is near zero with |z| usually under 3.
    """
    if "Time" not in frame.columns:
        return False
    time_abs = frame["Time"].abs()
    amount_abs = frame["Amount"].abs() if "Amount" in frame.columns else time_abs
    return bool((time_abs.max() < 10) and (amount_abs.max() < 50))
