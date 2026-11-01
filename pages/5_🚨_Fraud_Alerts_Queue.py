import streamlit as st
import pandas as pd
import numpy as np
import os

from src.database import get_active_alerts, get_recent_transactions, update_transaction_status
from src.alerts import format_alert_badge

st.set_page_config(page_title="Fraud Alerts Queue — SecureBank AI", page_icon="🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
.alert-box {
    background: #111827;
    border: 1px solid #263345;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚨 Live Fraud Alerts & Security Triage Queue")
st.markdown("Security Operations Center (SOC) review queue connected to persistent SQLite audit storage.")

# Fetch alerts
alerts_df = get_active_alerts()

if alerts_df.empty:
    # If no alerts in live DB yet, seed a few realistic operational alerts for demonstration
    from src.database import log_transaction
    from src.rules import apply_fraud_rules
    
    # Seed 3 demo alerts so the table is immediately functional
    demo_alerts_1 = apply_fraud_rules(0.96, 1, 3200.0, 2)
    log_transaction("TXN-SEED-01", 3200.0, 0.96, 94.0, "Critical", "Fraud", 1, "BLOCK", hour_elapsed=2, alerts=demo_alerts_1)
    
    demo_alerts_2 = apply_fraud_rules(0.82, 0, 1450.0, 14)
    log_transaction("TXN-SEED-02", 1450.0, 0.82, 68.0, "High", "Fraud", 0, "MANUAL_REVIEW", hour_elapsed=14, alerts=demo_alerts_2)
    
    demo_alerts_3 = apply_fraud_rules(0.45, 1, 890.0, 3)
    log_transaction("TXN-SEED-03", 890.0, 0.45, 52.0, "Medium", "Legitimate", 1, "STEP_UP_AUTH", hour_elapsed=3, alerts=demo_alerts_3)
    
    alerts_df = get_active_alerts()

# KPI Row
n_total = len(alerts_df)
n_crit = (alerts_df["severity"] == "CRITICAL").sum()
n_high = (alerts_df["severity"] == "HIGH").sum()
total_exposure = alerts_df["amount"].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Active Alerts", f"{n_total}", "Pending Review")
with kpi2:
    st.metric("Critical Alerts", f"{n_crit}", "Immediate Action", delta_color="inverse")
with kpi3:
    st.metric("High Severity Alerts", f"{n_high}", "Priority Queue")
with kpi4:
    st.metric("Total Flagged Exposure", f"${total_exposure:,.2f}", "Protected Assets")

st.markdown("---")

# Filter Controls
f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    sev_filter = st.selectbox("Filter by Severity:", ["ALL", "CRITICAL", "HIGH", "MEDIUM"])

filtered_df = alerts_df if sev_filter == "ALL" else alerts_df[alerts_df["severity"] == sev_filter]

st.subheader(f"Alert Records ({len(filtered_df)} showing)")

# Interactive Triage Table
display_cols = ["alert_id", "transaction_id", "severity", "amount", "risk_score", "risk_level", "message", "created_at"]
available = [c for c in display_cols if c in filtered_df.columns]
st.dataframe(filtered_df[available], use_container_width=True, hide_index=True)

st.markdown("---")

# Analyst Action Console
st.subheader("🛠️ Security Analyst Incident Resolution")
t_col1, t_col2, t_col3 = st.columns([1, 1, 2])

with t_col1:
    target_txn = st.selectbox("Select Transaction to Resolve:", filtered_df["transaction_id"].unique())

with t_col2:
    action_decision = st.selectbox("Triage Resolution:", [
        "Confirmed Fraud (Permanently Block)",
        "False Positive (Authorize Transaction)",
        "Escalate to Special Investigations Unit (SIU)"
    ])

with t_col3:
    analyst_note = st.text_input("Analyst Audit Notes:", value="Manual verification completed per protocol.")

if st.button("💾 Commit Resolution & Update Audit Trail", type="primary"):
    res_code = "Confirmed Fraud" if "Confirmed" in action_decision else ("False Positive" if "False" in action_decision else "Escalated")
    success = update_transaction_status(target_txn, res_code, analyst_note)
    if success:
        st.success(f"Transaction {target_txn} successfully updated to status: '{res_code}' in SQLite database.")
        st.rerun()
    else:
        st.error("Failed to commit status change to SQLite.")
