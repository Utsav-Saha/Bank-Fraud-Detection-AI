"""
Risk Intelligence Scoring Engine for SecureBank AI.
Calculates a calibrated 0-100 Risk Score combining:
- Supervised ML Fraud Probability (65% base weight)
- Isolation Forest Anomaly Detection (20% structural weight)
- Transaction Monetary Exposure (15% monetary exposure weight)
"""

from typing import Dict, Any

def calculate_risk_score(
    fraud_probability: float,
    anomaly_flag: int = 0,
    anomaly_score: float = 0.0,
    amount_usd: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate calibrated 0-100 risk score and operational recommendations.
    
    Args:
        fraud_probability: Float between 0.0 and 1.0 from supervised ML model.
        anomaly_flag: 1 if flagged as anomaly by Isolation Forest, else 0.
        anomaly_score: Normalized anomaly severity score (higher = more anomalous).
        amount_usd: Transaction amount in USD.
        
    Returns:
        Dictionary containing risk_score, risk_level, action_code, recommendation, and breakdown.
    """
    # 1. Supervised ML probability component (0 - 65 points)
    ml_component = float(fraud_probability) * 100.0 * 0.65
    
    # 2. Unsupervised Anomaly component (0 - 20 points)
    # If anomaly_flag is set, allocate minimum 12 points + bonus based on score
    if anomaly_flag == 1:
        scaled_anomaly = min(20.0, 12.0 + max(0.0, anomaly_score) * 20.0)
    else:
        scaled_anomaly = 0.0
        
    # 3. Financial Exposure component (0 - 15 points)
    if amount_usd > 5000.0:
        amount_penalty = 15.0
    elif amount_usd > 1500.0:
        amount_penalty = 10.0
    elif amount_usd > 500.0:
        amount_penalty = 5.0
    else:
        amount_penalty = 0.0
        
    raw_score = ml_component + scaled_anomaly + amount_penalty
    risk_score = round(min(100.0, max(0.0, raw_score)), 2)
    
    # Operational Action Levels
    if risk_score <= 30.0:
        risk_level = "Low"
        action_code = "APPROVE"
        badge_color = "#10b981"  # Emerald Green
        recommendation = "Normal monitoring. Transaction cleared for automatic processing."
    elif risk_score <= 60.0:
        risk_level = "Medium"
        action_code = "STEP_UP_AUTH"
        badge_color = "#f59e0b"  # Amber Orange
        recommendation = "Additional verification required. Trigger 2-Factor Authentication (OTP/SMS)."
    elif risk_score <= 80.0:
        risk_level = "High"
        action_code = "MANUAL_REVIEW"
        badge_color = "#f97316"  # Deep Orange
        recommendation = "Priority manual review. Place temporary hold on transaction settlement."
    else:
        risk_level = "Critical"
        action_code = "BLOCK"
        badge_color = "#ef4444"  # Crimson Red
        recommendation = "Urgent action! Immediately block transaction and freeze card for fraud inspection."
        
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action_code": action_code,
        "recommendation": recommendation,
        "badge_color": badge_color,
        "breakdown": {
            "ml_probability_pct": round(fraud_probability * 100.0, 2),
            "ml_points": round(ml_component, 2),
            "anomaly_flag": bool(anomaly_flag),
            "anomaly_points": round(scaled_anomaly, 2),
            "exposure_penalty_points": round(amount_penalty, 2),
            "amount_usd": round(amount_usd, 2)
        }
    }
