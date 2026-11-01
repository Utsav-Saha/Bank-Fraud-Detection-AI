"""CSV and numeric validation helpers."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from marshmallow import ValidationError
from werkzeug.datastructures import FileStorage

from app.models.schemas import OPTIONAL_IGNORED, REQUIRED_FEATURES, _finite_number


ALLOWED_CSV_EXTENSIONS = {".csv"}


def validate_upload(file: FileStorage | None, max_bytes: int) -> bytes:
    if file is None or not file.filename:
        raise ValidationError("A CSV file named 'file' is required.")
    name = file.filename.lower()
    if not any(name.endswith(ext) for ext in ALLOWED_CSV_EXTENSIONS):
        raise ValidationError("Only .csv files are accepted.")
    payload = file.read()
    if not payload:
        raise ValidationError("The uploaded file is empty.")
    if len(payload) > max_bytes:
        raise ValidationError(f"File exceeds the maximum size of {max_bytes} bytes.")
    return payload


def parse_transaction_csv(payload: bytes, max_rows: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as exc:
        raise ValidationError(f"Unable to parse CSV: {exc}") from exc

    if frame.empty:
        raise ValidationError("CSV contains no rows.")
    if len(frame) > max_rows:
        raise ValidationError(f"CSV exceeds the maximum of {max_rows} rows.")

    missing = [col for col in REQUIRED_FEATURES if col not in frame.columns]
    if missing:
        raise ValidationError({"missing_features": missing})

    unexpected = [
        col for col in frame.columns if col not in REQUIRED_FEATURES and col not in OPTIONAL_IGNORED
    ]
    if unexpected:
        raise ValidationError({"unexpected_features": unexpected})

    for col in REQUIRED_FEATURES:
        coerced = pd.to_numeric(frame[col], errors="coerce")
        if coerced.isna().any() or (~coerced.apply(lambda x: x == x and abs(x) != float("inf"))).any():
            bad_idx = coerced.index[coerced.isna() | ~coerced.apply(lambda x: abs(x) != float("inf"))].tolist()[:10]
            raise ValidationError(
                {
                    "invalid_numeric_values": {
                        "column": col,
                        "row_indexes": bad_idx,
                        "message": "NaN, Infinity, and non-numeric values are not allowed.",
                    }
                }
            )
        frame[col] = coerced
    return frame


def features_from_mapping(data: dict[str, Any]) -> dict[str, float]:
    return {name: _finite_number(data[name]) for name in REQUIRED_FEATURES}
