"""
Rule-Based Expert System for Financial Fraud Detection.
Implements forward-chaining expert rules to produce clear,
interpretable alerts and regulatory audit explanations.
"""

from typing import List, Dict, Any, Optional

def apply_fraud_rules(
    fraud_probability: float,
    anomaly_flag: int,
    amount_usd: float = 0.0,
    hour_of_day: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates facts against expert rules.
    
    Args:
        fraud_probability: Supervised ML prediction probability [0, 1].
        anomaly_flag: 1 if Isolation Forest flagged the transaction as an outlier, else 0.
        amount_usd: Transaction monetary amount.
        hour_of_day: Integer hour [0, 23] if available.
        
    Returns:
        List of triggered rule dictionaries with rule_id, severity, and explanation.
    """
    alerts = []
    
    # Rule 1: Extreme Supervised Probability
    if fraud_probability >= 0.95:
        alerts.append({
            "rule_id": "R101",
            "name": "CRITICAL_FRAUD_PROBABILITY",
            "severity": "CRITICAL",
            "message": "Ensemble ML model detected severe fraud probability (≥ 95%). High confidence malicious pattern."
        })
    elif fraud_probability >= 0.80:
        alerts.append({
            "rule_id": "R102",
            "name": "HIGH_FRAUD_PROBABILITY",
            "severity": "HIGH",
            "message": "Ensemble ML model flagged elevated fraud probability (≥ 80%). Requires operational attention."
        })
    elif fraud_probability >= 0.50:
        alerts.append({
            "rule_id": "R103",
            "name": "ELEVATED_SUSPICION",
            "severity": "MEDIUM",
            "message": "Supervised model classification exceeds standard 50% decision threshold."
        })

    # Rule 2: Unsupervised Structural Anomaly
    if anomaly_flag == 1:
        alerts.append({
            "rule_id": "R201",
            "name": "UNUSUAL_TRANSACTION_PATTERN",
            "severity": "HIGH",
            "message": "Isolation Forest identified multidimensional feature space anomaly. Transaction deviates from normal baseline."
        })

    # Rule 3: High-Value Exposure
    if amount_usd >= 2500.0:
        alerts.append({
            "rule_id": "R301",
            "name": "HIGH_MONETARY_EXPOSURE",
            "severity": "MEDIUM",
            "message": f"Transaction volume (${amount_usd:,.2f}) significantly exceeds median credit card expenditure."
        })

    # Rule 4: Night-time Vulnerability Window (00:00 to 05:00)
    if hour_of_day is not None and (0 <= hour_of_day <= 5):
        if fraud_probability >= 0.40 or anomaly_flag == 1:
            alerts.append({
                "rule_id": "R401",
                "name": "NIGHT_HOURS_HIGH_VELOCITY",
                "severity": "HIGH",
                "message": f"Suspicious transaction originated during high-risk overnight window ({hour_of_day:02d}:00 hours)."
            })

    # Rule 5: Dual Flag Convergence
    if fraud_probability >= 0.70 and anomaly_flag == 1:
        alerts.append({
            "rule_id": "R501",
            "name": "DUAL_ENGINE_CONVERGENCE",
            "severity": "CRITICAL",
            "message": "CONVERGENCE ALERT: Supervised ML classifier and Unsupervised Anomaly detector both flagged this transaction independently."
        })

    if not alerts:
        alerts.append({
            "rule_id": "R000",
            "name": "NORMAL_PROFILE",
            "severity": "INFO",
            "message": "No configured fraud rules triggered. All parameters conform to expected baseline."
        })

    return alerts
