import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from src.prediction import get_predictor

st.set_page_config(page_title="Fraud Predictor — SecureBank AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
.result-card {
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
    border: 1px solid #334155;
}
.alert-pill {
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Real-Time Transaction Fraud Predictor")
st.markdown("Evaluate individual transactions or upload batch CSV files for instant multi-engine hybrid risk scoring.")

predictor = get_predictor()

# Tabs: Single Transaction vs Batch Upload
tab_single, tab_batch = st.tabs(["⚡ Single Transaction Evaluation", "📁 Batch CSV Scoring"])

with tab_single:
    st.subheader("1. Preset Test Scenarios (1-Click Fill)")
    preset_choice = st.selectbox(
        "Select an automated test scenario:",
        [
            "Custom Manual Entry",
            "✅ Verified Legitimate Transaction (Grocery Purchase $45.20)",
            "🚨 Confirmed High-Risk Attack (Outlier V14/V10, $1,850.00)",
            "⚠️ Suspicious Night-time Anomaly (High Isolation Score, $420.00)"
        ]
    )
    
    # Preset definitions
    defaults = {f"V{i}": 0.0 for i in range(1, 29)}
    defaults["Time"] = 36000.0
    defaults["Amount"] = 120.0
    
    if "Verified Legitimate" in preset_choice:
        defaults["Time"] = 52000.0
        defaults["Amount"] = 45.20
        defaults["V14"] = 0.42
        defaults["V10"] = 0.15
        defaults["V12"] = 0.28
    elif "Confirmed High-Risk" in preset_choice:
        defaults["Time"] = 7200.0
        defaults["Amount"] = 1850.00
        # Highly distinctive fraud PCA values from known frauds
        defaults["V14"] = -8.24
        defaults["V10"] = -6.12
        defaults["V12"] = -5.48
        defaults["V17"] = -7.15
        defaults["V4"] = 4.35
    elif "Suspicious Night-time" in preset_choice:
        defaults["Time"] = 8400.0  # 2:20 AM
        defaults["Amount"] = 420.00
        defaults["V14"] = -3.20
        defaults["V10"] = -2.10
        defaults["V4"] = 2.80

    st.markdown("#### Transaction Parameters")
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        in_time = st.number_input("Time (Seconds from start)", value=float(defaults["Time"]), step=100.0)
    with p_col2:
        in_amount = st.number_input("Transaction Amount ($ USD)", value=float(defaults["Amount"]), min_value=0.0, step=10.0)

    with st.expander("🔬 Advanced PCA Latent Components (V1 – V28)", expanded=False):
        st.caption("PCA transformed latent features from bank anonymization pipeline.")
        v_inputs = {}
        cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with cols[col_idx]:
                key = f"V{i}"
                v_inputs[key] = st.number_input(key, value=float(defaults[key]), step=0.1, format="%.4f")

    if st.button("🚀 Analyze Transaction Risk", type="primary", use_container_width=True):
        input_data = {"Time": in_time, "Amount": in_amount}
        input_data.update(v_inputs)
        
        with st.spinner("Running Supervised ML, Isolation Forest, and Expert Rule Engine..."):
            result = predictor.predict_single(input_data, persist_to_db=True)
            
        st.markdown("---")
        
        # Result Header Banner
        is_fraud = (result["prediction"] == "Fraud")
        bg_card = "#450a0a" if is_fraud else "#064e3b"
        border_card = "#dc2626" if is_fraud else "#10b981"
        icon = "🚨" if is_fraud else "✅"
        
        st.markdown(f"""
        <div style="background: {bg_card}; border: 2px solid {border_card}; border-radius: 16px; padding: 22px; text-align: center; margin-bottom: 25px;">
            <div style="font-size: 38px;">{icon}</div>
            <div style="font-size: 26px; font-weight: 800; color: #ffffff; margin-top: 6px;">
                Prediction: {result['prediction'].upper()}
            </div>
            <div style="font-size: 14px; color: #e2e8f0; margin-top: 4px;">
                Reference ID: <b>{result['transaction_id']}</b> • Operational Action: <b>{result['action_code']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk & Anomaly Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%", "Supervised ML")
        with m2:
            st.metric("Composite Risk Score", f"{result['risk_score']} / 100", f"Tier: {result['risk_level']}")
        with m3:
            st.metric("Structural Anomaly", "FLAGGED" if result['anomaly_detected'] else "NORMAL", f"Score: {result['anomaly_score']}")
        with m4:
            st.metric("Action Directive", result['action_code'], "Security Protocol")

        # Operational Recommendation Box
        st.info(f"**Operational Protocol**: {result['recommendation']}")

        # Triggered Expert Rules
        st.subheader("📋 Triggered Expert System Rules")
        for alert in result["alerts"]:
            sev = alert.get("severity", "INFO")
            color_map = {
                "CRITICAL": ("#ef4444", "#450a0a"),
                "HIGH": ("#f97316", "#431407"),
                "MEDIUM": ("#f59e0b", "#451a03"),
                "INFO": ("#3b82f6", "#172554")
            }
            fg, bg = color_map.get(sev, ("#94a3b8", "#1e293b"))
            st.markdown(f"""
            <div style="background: {bg}; border: 1px solid {fg}; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
                <b style="color: {fg};">[{alert.get('rule_id', 'RULE')}] {sev}</b>: {alert.get('message', '')}
            </div>
            """, unsafe_allow_html=True)

with tab_batch:
    st.subheader("📁 Vectorized Multi-Record Batch CSV Scoring")
    st.markdown("Upload a CSV file containing transactions. The file will be standardized and scored across all engines.")
    
    sample_file = "data/sample/sample_transactions.csv"
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            st.download_button("📥 Download Benchmark Sample CSV (10 Transactions)", f, file_name="benchmark_sample.csv", mime="text/csv")
            
    uploaded_file = st.file_uploader("Upload Transaction CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(batch_raw):,} records successfully.")
            
            if st.button("⚡ Score Entire Batch Dataset", type="primary"):
                with st.spinner("Scoring batch transactions..."):
                    scored_df = predictor.predict_batch(batch_raw, persist_to_db=False)
                    
                st.subheader("Scored Results Preview")
                st.dataframe(scored_df, use_container_width=True)
                
                # Summary metrics
                n_flagged = (scored_df["prediction"] == "Fraud").sum()
                st.metric("Flagged Fraud Records in Batch", f"{n_flagged} / {len(scored_df)}", f"{n_flagged/len(scored_df)*100:.2f}%")
                
                csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Scored Results CSV",
                    csv_bytes,
                    file_name="securebank_scored_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
