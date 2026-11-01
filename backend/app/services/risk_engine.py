"""Prototype policy risk score. Distinct from raw model probability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskPolicy:
    low_max: float = 30.0
    medium_max: float = 60.0
    high_max: float = 80.0


def calculate_risk(
    fraud_probability: float,
    policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    """Risk Score = fraud probability × 100 (prototype policy, not industry standard)."""
    policy = policy or RiskPolicy()
    probability = max(0.0, min(1.0, float(fraud_probability)))
    risk_score = round(probability * 100.0, 2)

    if risk_score <= policy.low_max:
        level = "Low"
        recommendation = "Normal monitoring"
    elif risk_score <= policy.medium_max:
        level = "Medium"
        recommendation = "Additional verification recommended"
    elif risk_score <= policy.high_max:
        level = "High"
        recommendation = "Priority manual review"
    else:
        level = "Critical"
        recommendation = "Urgent manual review"

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "recommendation": recommendation,
        "policy": {
            "formula": "risk_score = fraud_probability * 100",
            "note": "Prototype policy thresholds, not industry-standard cutoffs.",
            "low_max": policy.low_max,
            "medium_max": policy.medium_max,
            "high_max": policy.high_max,
        },
        "probability_notes": {
            "raw_model_probability": round(probability, 6),
            "calibrated_probability": None,
            "calibration_available": False,
        },
    }
