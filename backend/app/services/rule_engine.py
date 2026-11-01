"""Forward-chaining expert rules over transaction facts."""

from __future__ import annotations

from typing import Any


def apply_rules(
    *,
    fraud_probability: float,
    prediction: int,
    anomaly_flag: int,
) -> list[dict[str, Any]]:
    facts = {
        "fraud_probability": float(fraud_probability),
        "prediction": int(prediction),
        "anomaly_flag": int(anomaly_flag),
    }
    triggered: list[dict[str, Any]] = []

    rules = [
        {
            "rule_id": "R102",
            "description": "High supervised fraud probability",
            "condition": facts["fraud_probability"] >= 0.80 and facts["fraud_probability"] < 0.95,
            "severity": "HIGH",
            "explanation": "Fraud probability is greater than or equal to 0.80.",
            "recommended_action": "Priority analyst review",
            "message": "High fraud probability detected",
        },
        {
            "rule_id": "R101",
            "description": "Critical supervised fraud probability",
            "condition": facts["fraud_probability"] >= 0.95,
            "severity": "CRITICAL",
            "explanation": "Fraud probability is greater than or equal to 0.95.",
            "recommended_action": "Urgent manual review",
            "message": "Critical fraud probability detected",
        },
        {
            "rule_id": "R201",
            "description": "Isolation Forest unusual-pattern warning",
            "condition": facts["anomaly_flag"] == 1,
            "severity": "HIGH",
            "explanation": "The anomaly detector flagged this transaction as unusual relative to the legitimate training baseline.",
            "recommended_action": "Investigate unusual feature pattern. An anomaly is not confirmed fraud.",
            "message": "Unusual transaction pattern detected",
        },
        {
            "rule_id": "R501",
            "description": "Combined supervised and anomaly warning",
            "condition": facts["prediction"] == 1 and facts["anomaly_flag"] == 1,
            "severity": "CRITICAL",
            "explanation": "The selected ML model predicted fraud and the anomaly detector also flagged an unusual pattern.",
            "recommended_action": "Urgent combined-signal review",
            "message": "ML fraud prediction and anomaly detector both flagged this transaction",
        },
    ]

    for rule in rules:
        if rule["condition"]:
            triggered.append(
                {
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "trigger_condition": rule["explanation"],
                    "severity": rule["severity"],
                    "explanation": rule["explanation"],
                    "recommended_action": rule["recommended_action"],
                    "message": rule["message"],
                }
            )
    return triggered
