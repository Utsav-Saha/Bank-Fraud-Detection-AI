"""Short-horizon trend aggregation. The Kaggle Time column spans about two days."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from flask import current_app


def trend_payload() -> dict[str, Any]:
    path = current_app.config["TRENDS_CSV"]
    if not path.exists():
        return {
            "available": False,
            "message": "data/processed/hourly_fraud_trends.csv was not found.",
            "disclaimer": (
                "The source Time column is seconds from the first transaction, covering roughly two days. "
                "This is not a long-term calendar forecast."
            ),
        }
    frame = pd.read_csv(path)
    frame = frame.sort_values(frame.columns[0])
    y = frame["fraud_tx"].astype(float).to_numpy() if "fraud_tx" in frame.columns else np.array([])
    forecast = _naive_forecast(y, horizon=6)
    return {
        "available": True,
        "interval": "hour_elapsed_from_dataset_start",
        "disclaimer": (
            "Aggregations use the Kaggle Time column (seconds from the first transaction), about two days of data. "
            "No calendar dates are invented. Forecasts are a simple baseline, not operational predictions."
        ),
        "points": frame.to_dict(orient="records"),
        "forecast_baseline": forecast,
        "lstm_experiment": {
            "ran": False,
            "reason": "Only about 48 hourly points exist. That sequence is too short for a reliable LSTM experiment.",
        },
    }


def _naive_forecast(series: np.ndarray, horizon: int = 6) -> dict[str, Any]:
    if series.size == 0:
        return {"method": "none", "values": []}
    window = min(6, series.size)
    last_mean = float(np.mean(series[-window:]))
    values = [round(last_mean, 4) for _ in range(horizon)]
    mae = None
    if series.size > window:
        naive = np.full(window, float(np.mean(series[-(window * 2) : -window])))
        actual = series[-window:]
        mae = round(float(np.mean(np.abs(actual - naive))), 4)
    return {
        "method": "moving_average_naive",
        "horizon_intervals": horizon,
        "values": values,
        "holdout_mae_last_window": mae,
        "note": "Chronological holdout on the last window only. Not a production forecast.",
    }
