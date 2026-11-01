import streamlit as st
import pandas as pd
import numpy as np
import os
import json

from src.database import get_database_kpis, get_recent_transactions
from src.prediction import get_predictor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SecureBank AI — Fraud Intelligence Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL CUSTOM STYLING (DARK ENTERPRISE FINTECH AESTHETICS)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #0b1120;
    color: #f8fafc;
}

/* Hide default streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1f2937;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Typography */
.portal-header {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.portal-sub {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Badges */
.badge-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    background: #064e3b;
    color: #34d399;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid #059669;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}

/* Metric Cards */
.kpi-card {
    background: linear-gradient(145deg, #131c31, #18243c);
    border: 1px solid #26354f;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
}

.kpi-title {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 800;
    margin-top: 6px;
    letter-spacing: -0.5px;
}

.kpi-subtext {
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
}

/* Feature grid card */
.feature-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}

.feature-box h4 {
    color: #38bdf8;
    margin-bottom: 8px;
    font-weight: 700;
}

.feature-box p {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR BRANDING
# =========================================================
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px;">
    <div style="background: linear-gradient(135deg, #2563eb, #06b6d4); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px;">
        🏦
    </div>
    <div>
        <div style="font-size: 20px; font-weight: 800; color: #ffffff;">SecureBank AI</div>
        <div style="font-size: 11px; color: #64748b; letter-spacing: 0.5px;">HYBRID FRAUD INTELLIGENCE</div>
    </div>
</div>
<div class="badge-status">
    <span class="pulse-dot"></span> SYSTEM ONLINE (V2.0.0)
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("ACTIVE AI SUBSYSTEMS")
st.sidebar.markdown("""
- 🌲 **Supervised ML**: Random Forest
- 🔎 **Anomaly Engine**: Isolation Forest
- ⚙️ **Expert Rules**: Forward Chaining
- 📊 **Risk Engine**: 0–100 Multi-Factor
- 🧠 **Explainability**: SHAP Attribution
- 🕸️ **Graph Analytics**: NetworkX (BFS/DFS)
""")

# =========================================================
# MAIN LANDING PORTAL
# =========================================================
st.markdown("""
<div class="portal-header">Enterprise Fraud Intelligence & Risk Platform</div>
<div class="portal-sub">Hybrid AI Architecture combining Supervised Machine Learning, Unsupervised Anomaly Detection, Rule-Based Reasoning, and SHAP Explainability.</div>
""", unsafe_allow_html=True)

# Fetch KPIs
kpis = get_database_kpis()

# Check dataset metrics
trends_file = "data/processed/hourly_fraud_trends.csv"
historical_total = 284807
historical_fraud = 492
historical_rate = 0.173
if os.path.exists(trends_file):
    trends_df = pd.read_csv(trends_file)
    historical_total = int(trends_df["total_tx"].sum())
    historical_fraud = int(trends_df["fraud_tx"].sum())
    historical_rate = round((historical_fraud / historical_total) * 100, 3)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Evaluated Portfolio Volume</div>
        <div class="kpi-value">{historical_total:,}</div>
        <div class="kpi-subtext">Credit card transactions scanned</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Detected Fraud Incidents</div>
        <div class="kpi-value" style="color: #ef4444;">{historical_fraud:,}</div>
        <div class="kpi-subtext">Confirmed fraudulent attempts</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Portfolio Fraud Rate</div>
        <div class="kpi-value" style="color: #f59e0b;">{historical_rate}%</div>
        <div class="kpi-subtext">Extreme class imbalance benchmark</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Active Alerts Queue</div>
        <div class="kpi-value" style="color: #38bdf8;">{kpis['urgent_reviews_pending']}</div>
        <div class="kpi-subtext">High/Critical risk pending review</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick Navigation Section
st.subheader("🚀 Platform Modules & Operational Cockpit")
st.caption("Select a module below or use the sidebar navigation:")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="feature-box">
        <h4>1. 📊 Executive Dashboard</h4>
        <p>Comprehensive transaction monitoring KPIs, monetary volume distributions, and high-level risk distribution analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <h4>4. 🧠 Explainable AI (SHAP)</h4>
        <p>Demystify model classifications with Shapley values. Inspect which specific features drove a high fraud probability score.</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div class="feature-box">
        <h4>2. 🔍 Real-Time Fraud Predictor</h4>
        <p>Single transaction inference with 1-click test presets and vectorized bulk CSV uploading with instant risk classification.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <h4>5. 🚨 Fraud Alerts & Triage</h4>
        <p>Interactive security analyst queue. Review, approve, block, or mark false positives with persistent audit logging.</p>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div class="feature-box">
        <h4>3. ⚡ Multi-Factor Risk Engine</h4>
        <p>Holistic 0-100 risk scoring blending Supervised ML confidence, Isolation Forest anomaly flags, and monetary loss exposure.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <h4>6. 📈 Fraud Trend Forecasting</h4>
        <p>Temporal analysis over 48 hours, highlighting diurnal night-time fraud rate spikes and Holt-Winters trend forecasting.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Recent Evaluated Transactions Feed
st.subheader("📋 Recent Transaction Audit Feed")
recent_df = get_recent_transactions(limit=10)

if not recent_df.empty:
    display_cols = ["transaction_id", "created_at", "amount", "prediction", "risk_score", "risk_level", "anomaly_flag", "review_status"]
    available_cols = [c for c in display_cols if c in recent_df.columns]
    st.dataframe(
        recent_df[available_cols],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No live transactions evaluated in this session yet. Run a prediction in '🔍 Fraud Prediction' to populate the audit log.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("SecureBank AI • Semester Project — Advanced Hybrid Intelligence System • Built with Streamlit, Scikit-Learn, SHAP, and NetworkX")
