import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from src.explainability import get_explainer

st.set_page_config(page_title="Explainable AI (SHAP) — SecureBank AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1120; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Explainable AI (SHAP Interpretability)")
st.markdown("Auditable machine learning decisions using **Shapley Additive exPlanations (SHAP)** to explain *why* transactions receive their fraud probability.")

explainer = get_explainer()

# Global Feature Importance Section
st.subheader("🌐 Global Feature Attribution (Macro Model Drivers)")
st.caption("Which latent components dominate the model's global fraud detection criteria across 284k transactions:")

global_df = explainer.get_global_feature_importance(top_n=10)

if not global_df.empty:
    fig_global = px.bar(
        global_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Global Feature Importance (Gini)", "feature": "PCA Feature Component"}
    )
    fig_global.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        height=380,
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_global, use_container_width=True)

st.markdown("---")

# Local Single Transaction Explanation
st.subheader("🎯 Local Transaction Waterfall Attribution")
st.markdown("Inspect how each individual feature nudges the fraud probability higher or lower for a specific transaction.")

sample_choice = st.selectbox(
    "Select a case study transaction to explain:",
    [
        "🚨 High-Risk Fraud Attack Case (V14 = -8.24, V10 = -6.12)",
        "✅ Clean Everyday Grocery Transaction (V14 = +0.42, Normal V1-V28)",
        "⚠️ Borderline Night-time Transaction (V14 = -3.20, Outlier V4)"
    ]
)

# Construct feature row
sample_row = {f"V{i}": 0.0 for i in range(1, 29)}
sample_row["Time"] = 36000.0
sample_row["Amount"] = 120.0

if "High-Risk Fraud" in sample_choice:
    sample_row["Time"] = 7200.0
    sample_row["Amount"] = 1850.0
    sample_row["V14"] = -8.24
    sample_row["V10"] = -6.12
    sample_row["V12"] = -5.48
    sample_row["V17"] = -7.15
    sample_row["V4"] = 4.35
elif "Clean Everyday" in sample_choice:
    sample_row["Time"] = 52000.0
    sample_row["Amount"] = 45.20
    sample_row["V14"] = 0.42
    sample_row["V10"] = 0.15
    sample_row["V12"] = 0.28
elif "Borderline" in sample_choice:
    sample_row["Time"] = 8400.0
    sample_row["Amount"] = 420.0
    sample_row["V14"] = -3.20
    sample_row["V10"] = -2.10
    sample_row["V4"] = 2.80

sample_df = pd.DataFrame([sample_row])

with st.spinner("Computing Shapley additive values..."):
    contributions = explainer.explain_transaction(sample_df, top_n=8)

cont_df = pd.DataFrame(contributions)

if not cont_df.empty:
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        # Plotly horizontal bar chart of local contributions
        chart_data = cont_df.sort_values("shap_value", ascending=True)
        colors = ["#ef4444" if val > 0 else "#3b82f6" for val in chart_data["shap_value"]]
        
        fig_waterfall = go.Figure(go.Bar(
            x=chart_data["shap_value"],
            y=chart_data["feature"],
            orientation="h",
            marker=dict(color=colors)
        ))
        fig_waterfall.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig_waterfall.update_layout(
            title="Local SHAP Risk Contributions (Red = Increases Risk, Blue = Reduces Risk)",
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            height=380,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis_title="SHAP Value Contribution to Fraud Score"
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
    with col_table:
        st.markdown("#### Feature Influence Table")
        display_tbl = cont_df[["feature", "feature_value", "shap_value", "impact"]].copy()
        display_tbl.columns = ["Feature", "Value", "SHAP Impact", "Direction"]
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)
        
    st.markdown("""
    > 💡 **Compliance Note (SAR Filing)**: Under regulatory frameworks (e.g. BSA, FinCEN SAR, and GDPR Article 22), 
    > models must not operate as opaque black boxes. SHAP provides the quantitative evidence demonstrating exactly 
    > which deviations triggered automated fraud alerts.
    """)
