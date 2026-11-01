import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Executive Dashboard — SecureBank AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
.metric-box {
    background: linear-gradient(145deg, #111827, #1f293d);
    border: 1px solid #283548;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Executive Fraud Intelligence Dashboard")
st.markdown("High-level portfolio performance, transaction volume metrics, and financial risk distributions.")

# Load trends data
trends_path = "data/processed/hourly_fraud_trends.csv"
if not os.path.exists(trends_path):
    st.error("Hourly fraud trends dataset not found. Please verify Phase 1 execution.")
    st.stop()

trends_df = pd.read_csv(trends_path)

total_tx = int(trends_df["total_tx"].sum())
total_fraud = int(trends_df["fraud_tx"].sum())
total_legit = total_tx - total_fraud
fraud_rate = (total_fraud / total_tx) * 100
total_volume = trends_df["total_amount"].sum()
fraud_volume = trends_df["fraud_amount"].sum()

# Top metric row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{total_tx:,}", "2-Day Portfolio")
with col2:
    st.metric("Detected Fraud", f"{total_fraud:,}", f"{fraud_rate:.3f}% Fraud Rate", delta_color="inverse")
with col3:
    st.metric("Total Monetary Volume", f"${total_volume:,.2f}", "Total Transacted")
with col4:
    st.metric("Fraud Exposure Prevented", f"${fraud_volume:,.2f}", "Flagged Exposure", delta_color="inverse")

st.markdown("---")

# Visualizations Row 1
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Transaction Class Split")
    fig_donut = go.Figure(data=[go.Pie(
        labels=["Legitimate (99.83%)", "Fraudulent (0.17%)"],
        values=[total_legit, total_fraud],
        hole=0.6,
        marker=dict(colors=["#3b82f6", "#ef4444"])
    )])
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with c2:
    st.subheader("Hourly Fraud Rate Velocity (%)")
    fig_rate = px.bar(
        trends_df,
        x="Hour_Elapsed",
        y="fraud_rate_pct",
        color="fraud_rate_pct",
        color_continuous_scale="Reds",
        labels={"Hour_Elapsed": "Elapsed Time (Hours)", "fraud_rate_pct": "Fraud Rate (%)"}
    )
    fig_rate.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_rate, use_container_width=True)

# Visualizations Row 2
st.subheader("Transaction Monetary Volume Distribution")
col_vol1, col_vol2 = st.columns(2)

with col_vol1:
    fig_hourly_vol = go.Figure()
    fig_hourly_vol.add_trace(go.Scatter(
        x=trends_df["Hour_Elapsed"],
        y=trends_df["total_amount"],
        mode="lines+markers",
        name="Total Volume ($)",
        line=dict(color="#3b82f6", width=2)
    ))
    fig_hourly_vol.update_layout(
        title="Hourly Gross Monetary Volume ($)",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827"
    )
    st.plotly_chart(fig_hourly_vol, use_container_width=True)

with col_vol2:
    fig_fraud_vol = go.Figure()
    fig_fraud_vol.add_trace(go.Bar(
        x=trends_df["Hour_Elapsed"],
        y=trends_df["fraud_amount"],
        name="Fraudulent Loss ($)",
        marker_color="#ef4444"
    ))
    fig_fraud_vol.update_layout(
        title="Hourly Fraudulent Volume Exposure ($)",
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827"
    )
    st.plotly_chart(fig_fraud_vol, use_container_width=True)
