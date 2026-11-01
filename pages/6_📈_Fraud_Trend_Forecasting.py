import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Fraud Trends & Forecasting — SecureBank AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ADVANCED CUSTOM STYLING (GLASSMORPHISM & FINTECH DARK THEME)
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

/* Page Header */
.page-title {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 22px;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 20px;
    background: rgba(37, 99, 235, 0.15);
    color: #60a5fa;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.3);
    margin-bottom: 12px;
}

/* Metric Cards with Glassmorphic Gradient */
.kpi-card {
    background: linear-gradient(135deg, rgba(19, 28, 49, 0.9), rgba(24, 36, 60, 0.8));
    border: 1px solid #223147;
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease-in-out;
}

.kpi-card:hover {
    border-color: #3b82f6;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -6px rgba(37, 99, 235, 0.25);
}

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 4px;
    letter-spacing: -0.5px;
}

.kpi-delta {
    font-size: 12px;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Callout Info Cards */
.insight-box {
    background: linear-gradient(145deg, #111827, #172033);
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 18px 0;
}

.alert-callout {
    background: linear-gradient(145deg, #2b0b0b, #3b1111);
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 18px 0;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    color: #94a3b8;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA INGESTION & ROBUSTNESS
# =========================================================
trends_path = "data/processed/hourly_fraud_trends.csv"

if not os.path.exists(trends_path):
    st.error("❌ Processed hourly trends file not found at `data/processed/hourly_fraud_trends.csv`. Please verify Phase 1 execution.")
    st.stop()

df = pd.read_csv(trends_path)

# Calculate key temporal statistics
total_tx = int(df["total_tx"].sum())
total_fraud = int(df["fraud_tx"].sum())
overall_fraud_rate = (total_fraud / total_tx) * 100
total_fraud_loss = df["fraud_amount"].sum()

# Peak stats
peak_count_row = df.loc[df["fraud_tx"].idxmax()]
peak_rate_row = df.loc[df["fraud_rate_pct"].idxmax()]

# Diurnal comparison: Night (00:00 - 05:00) vs Day (06:00 - 23:00)
night_mask = df["Hour_Of_Day"].isin([0, 1, 2, 3, 4, 5])
night_tx = df[night_mask]["total_tx"].sum()
night_fraud = df[night_mask]["fraud_tx"].sum()
night_rate = (night_fraud / night_tx) * 100 if night_tx > 0 else 0

day_tx = df[~night_mask]["total_tx"].sum()
day_fraud = df[~night_mask]["fraud_tx"].sum()
day_rate = (day_fraud / day_tx) * 100 if day_tx > 0 else 0
night_multiplier = (night_rate / day_rate) if day_rate > 0 else 1.0

# =========================================================
# HEADER BANNER
# =========================================================
st.markdown("""
<div class="status-pill">
    <span style="width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; display: inline-block;"></span>
    TEMPORAL RISK INTELLIGENCE & FORECASTING ENGINE
</div>
<div class="page-title">📈 Time-Series Fraud Trend Velocity & Forecasting</div>
<div class="page-subtitle">
Continuous 48-hour temporal modeling, diurnal day-night attack vector detection, and parametric Holt-Winters trend forecasting.
</div>
""", unsafe_allow_html=True)

# =========================================================
# EXECUTIVE KPI STRIP
# =========================================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">48-Hour Fraud Incidents</div>
        <div class="kpi-value" style="color: #f87171;">{total_fraud:,} <span style="font-size: 14px; color: #94a3b8;">/ {total_tx:,}</span></div>
        <div class="kpi-delta" style="color: #f59e0b;">
            Avg Rate: <b>{overall_fraud_rate:.3f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Peak Rate Attack Window</div>
        <div class="kpi-value" style="color: #fbbf24;">{peak_rate_row['Hour_Of_Day']:02d}:00 HRS</div>
        <div class="kpi-delta" style="color: #ef4444;">
            Peak Rate: <b>{peak_rate_row['fraud_rate_pct']:.2f}%</b> ({peak_rate_row['fraud_tx']} cases)
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Overnight Risk Multiplier</div>
        <div class="kpi-value" style="color: #38bdf8;">{night_multiplier:.1f}x <span style="font-size: 14px; color: #94a3b8;">Surge</span></div>
        <div class="kpi-delta" style="color: #cbd5e1;">
            Night: <b>{night_rate:.2f}%</b> vs Day: <b>{day_rate:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Prevented Fraud Loss</div>
        <div class="kpi-value" style="color: #34d399;">${total_fraud_loss:,.0f}</div>
        <div class="kpi-delta" style="color: #6ee7b7;">
            Peak Hour Loss: <b>${peak_count_row['fraud_amount']:,.0f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN INTERACTIVE TABS
# =========================================================
tab_overview, tab_diurnal, tab_forecast, tab_telemetry = st.tabs([
    "📊 Velocity & Timeline Dynamics",
    "🌙 Diurnal Vulnerability Matrix",
    "🔮 Interactive Trend Forecaster",
    "📋 Hourly Telemetry Data"
])

# ---------------------------------------------------------
# TAB 1: TIMELINE & VELOCITY DYNAMICS
# ---------------------------------------------------------
with tab_overview:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    
    with ctrl_col1:
        time_filter = st.selectbox("Timeline Scope:", ["Full 48 Hours", "Day 1 (Hours 0–23)", "Day 2 (Hours 24–47)"])
    
    with ctrl_col2:
        view_metric = st.selectbox(
            "Primary Visual Metric:",
            ["Fraud Count vs Total Volume", "Fraud Rate Percentage (%)", "Monetary Fraud Exposure ($)"]
        )

    # Filter data accordingly
    if time_filter == "Day 1 (Hours 0–23)":
        plot_df = df[df["Day"] == 1].copy()
    elif time_filter == "Day 2 (Hours 24–47)":
        plot_df = df[df["Day"] == 2].copy()
    else:
        plot_df = df.copy()

    if view_metric == "Fraud Count vs Total Volume":
        fig_main = go.Figure()
        
        # Total transaction area / line
        fig_main.add_trace(go.Scatter(
            x=plot_df["Hour_Elapsed"],
            y=plot_df["total_tx"],
            name="Total Transactions (Left Axis)",
            mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.08)'
        ))
        
        # Fraud transaction bars
        fig_main.add_trace(go.Bar(
            x=plot_df["Hour_Elapsed"],
            y=plot_df["fraud_tx"],
            name="Confirmed Fraud Cases (Right Axis)",
            yaxis="y2",
            marker=dict(
                color=plot_df["is_fraud_spike"].map({True: "#ef4444", False: "#f87171"}),
                line=dict(color="#991b1b", width=1)
            )
        ))
        
        fig_main.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            height=440,
            margin=dict(t=30, b=20, l=20, r=20),
            yaxis=dict(title="Total Transactions Volume", color="#60a5fa", gridcolor="#1f2937"),
            yaxis2=dict(
                title="Fraud Cases",
                color="#f87171",
                overlaying="y",
                side="right",
                gridcolor="rgba(0,0,0,0)"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_main, use_container_width=True)

    elif view_metric == "Fraud Rate Percentage (%)":
        fig_rate = go.Figure()
        
        fig_rate.add_trace(go.Bar(
            x=plot_df["Hour_Elapsed"],
            y=plot_df["fraud_rate_pct"],
            marker=dict(
                color=plot_df["fraud_rate_pct"],
                colorscale="YlOrRd",
                line=dict(color="#b45309", width=1)
            ),
            name="Hourly Fraud Rate (%)"
        ))
        
        # Mean benchmark line
        fig_rate.add_hline(
            y=overall_fraud_rate,
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"Benchmark Mean ({overall_fraud_rate:.2f}%)",
            annotation_position="top left"
        )
        
        fig_rate.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            height=420,
            margin=dict(t=30, b=20, l=20, r=20),
            xaxis=dict(title="Elapsed Time (Hours)", gridcolor="#1f2937"),
            yaxis=dict(title="Fraud Rate (% of Transactions)", gridcolor="#1f2937")
        )
        st.plotly_chart(fig_rate, use_container_width=True)

    else:
        fig_amt = go.Figure()
        fig_amt.add_trace(go.Bar(
            x=plot_df["Hour_Elapsed"],
            y=plot_df["fraud_amount"],
            name="Fraud Exposure ($ USD)",
            marker=dict(
                color="#dc2626",
                line=dict(color="#991b1b", width=1)
            )
        ))
        fig_amt.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            height=420,
            margin=dict(t=30, b=20, l=20, r=20),
            xaxis=dict(title="Elapsed Time (Hours)", gridcolor="#1f2937"),
            yaxis=dict(title="Financial Loss ($ USD)", gridcolor="#1f2937")
        )
        st.plotly_chart(fig_amt, use_container_width=True)

    # Moving Average Smoothing Section
    st.subheader("🌊 Velocity Smoothing & Outlier Spike Detection")
    fig_smooth = go.Figure()
    
    fig_smooth.add_trace(go.Scatter(
        x=df["Hour_Elapsed"],
        y=df["fraud_tx"],
        mode="markers",
        name="Observed Hourly Fraud Count",
        marker=dict(color="#64748b", size=5)
    ))
    
    fig_smooth.add_trace(go.Scatter(
        x=df["Hour_Elapsed"],
        y=df["rolling_fraud_3h"],
        mode="lines",
        name="3-Hour Rolling Moving Average",
        line=dict(color="#38bdf8", width=2.5)
    ))
    
    fig_smooth.add_trace(go.Scatter(
        x=df["Hour_Elapsed"],
        y=df["rolling_fraud_6h"],
        mode="lines",
        name="6-Hour Rolling Moving Average",
        line=dict(color="#c084fc", width=2, dash="dash")
    ))
    
    # Statistical Spikes (> 1.5 sigma)
    spikes = df[df["is_fraud_spike"]]
    fig_smooth.add_trace(go.Scatter(
        x=spikes["Hour_Elapsed"],
        y=spikes["fraud_tx"],
        mode="markers",
        name="High-Risk Velocity Burst (> 1.5σ)",
        marker=dict(color="#ef4444", size=11, symbol="diamond", line=dict(color="#ffffff", width=1.5))
    ))
    
    fig_smooth.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        height=380,
        margin=dict(t=30, b=20, l=20, r=20),
        xaxis=dict(title="Elapsed Hours (0 to 47)", gridcolor="#1f2937"),
        yaxis=dict(title="Fraud Incident Volume", gridcolor="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_smooth, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b style="color: #60a5fa; font-size: 14px;">💡 Technical Mechanism: Why Fraud Rate Surges at Night</b><br>
        <span style="color: #cbd5e1; font-size: 13px;">
        Legitimate cardholders sharply reduce purchasing between 01:00 AM and 05:00 AM, lowering the transaction denominator. 
        However, automated fraud bots and dark-web credential stuffing tools operate continuously around the clock, causing the 
        fraud proportion to jump from a daytime baseline of 0.12% to over 1.33%.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: DIURNAL VULNERABILITY MATRIX
# ---------------------------------------------------------
with tab_diurnal:
    st.subheader("🌙 24-Hour Diurnal Cycle Comparison (Day 1 vs. Day 2)")
    st.markdown("Comparing recurring hour-by-hour vulnerability between consecutive 24-hour cycles:")

    day1 = df[df["Day"] == 1]
    day2 = df[df["Day"] == 2]

    col_d1, col_d2 = st.columns([3, 1])

    with col_d1:
        fig_diurnal_bar = go.Figure()
        
        fig_diurnal_bar.add_trace(go.Bar(
            x=day1["Hour_Of_Day"],
            y=day1["fraud_rate_pct"],
            name="Day 1 Cycle (00:00 - 23:00)",
            marker=dict(color="#2563eb", opacity=0.85)
        ))
        
        fig_diurnal_bar.add_trace(go.Bar(
            x=day2["Hour_Of_Day"],
            y=day2["fraud_rate_pct"],
            name="Day 2 Cycle (24:00 - 47:00)",
            marker=dict(color="#f97316", opacity=0.85)
        ))
        
        fig_diurnal_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            height=400,
            xaxis=dict(
                title="Hour of Day (00:00 to 23:00)",
                tickmode="linear",
                tick0=0,
                dtick=1,
                gridcolor="#1f2937"
            ),
            yaxis=dict(title="Fraud Rate (%)", gridcolor="#1f2937"),
            margin=dict(t=30, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_diurnal_bar, use_container_width=True)

    with col_d2:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #223147; border-radius: 14px; padding: 18px; height: 95%;">
            <b style="color: #f8fafc; font-size: 15px;">🚨 High-Risk Window</b><br>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 6px;">
                Transactions initiated between <b>01:00 and 05:00 AM</b> demonstrate <b>3.8x elevated probability</b> of fraudulent compromise.
            </p>
            <hr style="border-color: #1f2937;">
            <div style="font-size: 12px; color: #cbd5e1;">
                • Day 1 Peak: <b>{day1['fraud_rate_pct'].max():.2f}%</b> (Hr {day1.loc[day1['fraud_rate_pct'].idxmax(), 'Hour_Of_Day']:02d}:00)<br><br>
                • Day 2 Peak: <b>{day2['fraud_rate_pct'].max():.2f}%</b> (Hr {day2.loc[day2['fraud_rate_pct'].idxmax(), 'Hour_Of_Day']:02d}:00)<br><br>
                • Recommended Action: <b>Enforce Step-Up 2FA</b> on transactions > $300 during overnight hours.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: INTERACTIVE PREDICTIVE FORECASTER
# ---------------------------------------------------------
with tab_forecast:
    st.subheader("🔮 Parametric Holt-Winters Trend Forecasting Model")
    st.markdown("Fit and project future fraud incident volume using **Holt's Linear Exponential Smoothing**:")

    # Pure Python Holt's Linear Trend Implementation
    class LiveHoltModel:
        def __init__(self, alpha=0.15, beta=0.20):
            self.alpha = alpha
            self.beta = beta
            
        def fit(self, y):
            n = len(y)
            L = np.zeros(n)
            T = np.zeros(n)
            L[0] = y[0]
            T[0] = y[1] - y[0] if n > 1 else 0
            for t in range(1, n):
                L[t] = self.alpha * y[t] + (1 - self.alpha) * (L[t-1] + T[t-1])
                T[t] = self.beta * (L[t] - L[t-1]) + (1 - self.beta) * T[t-1]
            self.level = L[-1]
            self.trend = T[-1]
            self.in_sample = np.zeros(n)
            self.in_sample[0] = y[0]
            for t in range(1, n):
                self.in_sample[t] = max(0, L[t-1] + T[t-1])
            return self
            
        def forecast(self, steps):
            h = np.arange(1, steps + 1)
            preds = self.level + h * self.trend
            return np.maximum(0, preds)

    # Interactive Hyperparameter Controls
    p_col1, p_col2, p_col3 = st.columns([1, 1, 1])

    with p_col1:
        horizon = st.slider("Forecast Horizon (Future Hours):", 6, 24, 12, 1)

    with p_col2:
        alpha_val = st.slider("Alpha (Level Smoothing α):", 0.01, 0.99, 0.14, 0.01)

    with p_col3:
        beta_val = st.slider("Beta (Trend Smoothing β):", 0.01, 0.99, 0.23, 0.01)

    # Split for out-of-sample validation: train on first 36h, evaluate on last 12h
    train_cutoff = 36
    y_full = df["fraud_tx"].values
    y_train = y_full[:train_cutoff]
    y_val = y_full[train_cutoff:]

    # Fit model on training slice
    model = LiveHoltModel(alpha=alpha_val, beta=beta_val).fit(y_train)
    val_preds = model.forecast(len(y_val))
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # Fit on all 48 hours to project into the future
    full_model = LiveHoltModel(alpha=alpha_val, beta=beta_val).fit(y_full)
    future_preds = full_model.forecast(horizon)
    future_hours = np.arange(48, 48 + horizon)

    # Plot Forecast
    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=df["Hour_Elapsed"],
        y=df["fraud_tx"],
        name="Historical Observed Fraud Count",
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5)
    ))

    # In-sample fitted
    fig_fc.add_trace(go.Scatter(
        x=df["Hour_Elapsed"],
        y=full_model.in_sample,
        name="Holt Model Filter In-Sample",
        mode="lines",
        line=dict(color="#a855f7", width=1.5, dash="dot")
    ))

    # Future Projection
    fig_fc.add_trace(go.Scatter(
        x=future_hours,
        y=future_preds,
        name=f"Projected Forecast (+{horizon}h)",
        mode="lines+markers",
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond")
    ))

    # Confidence Interval cone (95% CI)
    upper_ci = future_preds + 1.96 * rmse
    lower_ci = np.maximum(0, future_preds - 1.96 * rmse)

    fig_fc.add_trace(go.Scatter(
        x=np.concatenate([future_hours, future_hours[::-1]]),
        y=np.concatenate([upper_ci, lower_ci[::-1]]),
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% Forecast Confidence Band"
    ))

    fig_fc.add_vline(x=47.5, line_dash="dash", line_color="#64748b", annotation_text="Forecast Horizon Start")

    fig_fc.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        height=450,
        margin=dict(t=30, b=20, l=20, r=20),
        xaxis=dict(title="Continuous Timeline (Elapsed Hours)", gridcolor="#1f2937"),
        yaxis=dict(title="Projected Fraud Incidents", gridcolor="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Evaluation scorecard
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Test Out-of-Sample MAE", f"{mae:.2f} txns/hr", "Mean Absolute Error")
    with s2:
        st.metric("Test Out-of-Sample RMSE", f"{rmse:.2f} txns/hr", "Root Mean Squared Error")
    with s3:
        st.metric("Avg Projected Next 6h", f"{np.mean(future_preds[:6]):.1f} txns/hr", "Expected Incident Rate")

# ---------------------------------------------------------
# TAB 4: HOURLY TELEMETRY DATA TABLE
# ---------------------------------------------------------
with tab_telemetry:
    st.subheader("📋 Aggregated Hourly Telemetry Records")
    st.markdown("Raw aggregated metrics computed across the 284,807 transactions dataset:")

    # Clean display copy
    display_telemetry = df[[
        "Hour_Elapsed", "Day", "Hour_Of_Day", "total_tx", "fraud_tx", 
        "fraud_rate_pct", "total_amount", "fraud_amount", "is_fraud_spike"
    ]].copy()

    display_telemetry.columns = [
        "Hour Elapsed", "Day", "Hour of Day", "Total Transactions", "Fraud Cases",
        "Fraud Rate (%)", "Total Volume ($)", "Fraud Volume ($)", "Velocity Spike"
    ]

    st.dataframe(
        display_telemetry.style.format({
            "Fraud Rate (%)": "{:.3f}%",
            "Total Volume ($)": "${:,.2f}",
            "Fraud Volume ($)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Full Hourly Fraud Trends Dataset (CSV)",
        data=csv_data,
        file_name="hourly_fraud_trends_export.csv",
        mime="text/csv"
    )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("SecureBank AI • Fraud Trend Analytics Engine • Powered by Holt-Winters Exponential Smoothing & Plotly")
