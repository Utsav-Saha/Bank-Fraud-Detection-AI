import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.risk_engine import calculate_risk_score

st.set_page_config(page_title="Risk & Anomaly Analysis — SecureBank AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
.tier-card {
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Multi-Factor Risk & Anomaly Analysis")
st.markdown("Understand how SecureBank AI calibrates multi-engine signals into an unified 0–100 operational risk score.")

# Interactive Risk Calculator Simulator
st.subheader("🎛️ Live Risk Engine Signal Simulator")
st.caption("Adjust detection signals to observe how the multi-factor risk engine reacts in real time:")

calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    sim_prob = st.slider("Supervised ML Fraud Probability", 0.0, 1.0, 0.75, 0.01)
with calc_col2:
    sim_anomaly = st.selectbox("Isolation Forest Anomaly Detected?", [1, 0], format_func=lambda x: "🚨 YES (Structural Outlier)" if x == 1 else "✅ NO (Normal Pattern)")
    sim_score = st.slider("Anomaly Severity Distance", 0.0, 1.0, 0.50 if sim_anomaly == 1 else 0.0, 0.05)
with calc_col3:
    sim_amount = st.number_input("Transaction Monetary Amount ($ USD)", 0.0, 25000.0, 1850.0, 50.0)

# Compute composite score
calc_res = calculate_risk_score(sim_prob, sim_anomaly, sim_score, sim_amount)

# Visual Gauge
gauge_col, info_col = st.columns([1, 1])

with gauge_col:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=calc_res["risk_score"],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Calibrated Risk Score ({calc_res['risk_level']})", 'font': {'size': 20, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': calc_res['badge_color']},
            'bgcolor': "#111827",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': '#064e3b'},
                {'range': [30, 60], 'color': '#78350f'},
                {'range': [60, 80], 'color': '#7c2d12'},
                {'range': [80, 100], 'color': '#450a0a'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': calc_res['risk_score']
            }
        }
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        height=320,
        margin=dict(t=30, b=10, l=30, r=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with info_col:
    st.markdown(f"""
    <div style="background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 22px; height: 90%;">
        <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase;">Direct Operational Protocol</div>
        <div style="font-size: 26px; font-weight: 800; color: {calc_res['badge_color']}; margin-top: 5px;">
            Action: {calc_res['action_code']}
        </div>
        <p style="color: #cbd5e1; margin-top: 10px; font-size: 14px; line-height: 1.5;">
            {calc_res['recommendation']}
        </p>
        <hr style="border-color: #1f2937;">
        <div style="font-size: 13px; color: #94a3b8;">
            <b>Scoring Breakdown:</b><br>
            • Supervised ML Signal (65%): <b>+{calc_res['breakdown']['ml_points']} pts</b><br>
            • Unsupervised Anomaly Signal (20%): <b>+{calc_res['breakdown']['anomaly_points']} pts</b><br>
            • Financial Loss Exposure Penalty: <b>+{calc_res['breakdown']['exposure_penalty_points']} pts</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Framework Table
st.subheader("🏛️ Operational Risk Governance Framework")
st.markdown("""
| Tier | Score Range | Operational Code | System Protocol | Action Description |
| :---: | :---: | :---: | :---: | :--- |
| 🟢 **Low Risk** | 0 – 30 | `APPROVE` | Automatic Clearing | High confidence legitimate activity. Cleared through standard processing. |
| 🟡 **Medium Risk** | 31 – 60 | `STEP_UP_AUTH` | 2-Factor Challenge | Marginal suspicion. Trigger SMS One-Time Password (OTP) or Biometric re-verification. |
| 🟠 **High Risk** | 61 – 80 | `MANUAL_REVIEW` | Operational Hold | Suspicious pattern detected. Placed in tier-1 analyst queue with transaction hold. |
| 🔴 **Critical Risk** | 81 – 100 | `BLOCK` | Automated Interception | Extreme fraud likelihood or dual-engine convergence. Immediate payment decline and card lock. |
""")
