"""
Alert Management Service for SecureBank AI.
Categorizes, formats, and prioritizes fraud alerts for security analysts.
"""

from typing import List, Dict, Any
from src.database import get_active_alerts, update_transaction_status

def format_alert_badge(severity: str) -> Dict[str, str]:
    """Returns color styling for alert severity levels."""
    sev = severity.upper()
    if sev == "CRITICAL":
        return {"color": "#ef4444", "bg": "#450a0a", "label": "CRITICAL"}
    elif sev == "HIGH":
        return {"color": "#f97316", "bg": "#431407", "label": "HIGH"}
    elif sev == "MEDIUM":
        return {"color": "#f59e0b", "bg": "#451a03", "label": "MEDIUM"}
    else:
        return {"color": "#3b82f6", "bg": "#172554", "label": "INFO"}

def resolve_alert(transaction_id: str, resolution: str, notes: str = "") -> bool:
    """
    Resolves an alert. Resolution can be:
    - 'Confirmed Fraud'
    - 'False Positive / Cleared'
    - 'Under Escalation'
    """
    return update_transaction_status(transaction_id, resolution, notes)
