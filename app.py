import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SecureBank AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
}

/* Hide Streamlit default elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263244;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}


/* Main title */

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.main-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}


/* Logo */

.logo-box {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 35px;
}

.logo-icon {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
}

.logo-text {
    font-size: 21px;
    font-weight: 800;
    color: white;
}

.logo-sub {
    color: #64748b;
    font-size: 11px;
}


/* Status */

.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 20px;
    background: #052e2b;
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
}


/* Cards */

.metric-card {
    background: linear-gradient(
        145deg,
        #111827,
        #172033
    );

    border: 1px solid #263244;
    border-radius: 18px;
    padding: 22px;
    min-height: 135px;
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-icon {
    font-size: 25px;
}


/* Section */

.section-title {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
}


/* Analysis box */

.analysis-card {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 18px;
    padding: 25px;
}


/* Prediction cards */

.safe-card {
    background: linear-gradient(
        145deg,
        #052e2b,
        #064e3b
    );

    border: 1px solid #047857;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}

.fraud-card {
    background: linear-gradient(
        145deg,
        #450a0a,
        #7f1d1d
    );

    border: 1px solid #dc2626;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}

.prediction-icon {
    font-size: 55px;
}

.prediction-title {
    color: white;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
}

.prediction-text {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 8px;
}


/* Risk */

.risk-high {
    color: #f87171;
    font-size: 20px;
    font-weight: 800;
}

.risk-medium {
    color: #fbbf24;
    font-size: 20px;
    font-weight: 800;
}

.risk-low {
    color: #34d399;
    font-size: 20px;
    font-weight: 800;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #0891b2
    );
    color: white;
    border: none;
    font-weight: 700;
    font-size: 15px;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #0e7490
    );
    color: white;
}


/* Input fields */

.stNumberInput input {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px;
    margin-top: 40px;
    border-top: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/test_processed.csv"


# =========================================================
# LOAD MODEL
# =========================================================

try:
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
    else:
        raise FileNotFoundError
except FileNotFoundError:
    st.error(
        "❌ random_forest.pkl not found inside models folder."
    )
    st.stop()


# =========================================================
# LOAD SCALER
# =========================================================

try:
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            with open(SCALER_PATH, "rb") as file:
                scaler = pickle.load(file)
    else:
        raise FileNotFoundError
except FileNotFoundError:
    st.error(
        "❌ scaler.pkl not found inside models folder."
    )
    st.stop()


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        "❌ test_processed.csv not found inside data/processed folder."
    )
    st.stop()


# =========================================================
# FEATURES
# =========================================================

if "Class" in df.columns:
    feature_names = [
        col for col in df.columns
        if col != "Class"
    ]
else:
    feature_names = list(df.columns)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div class="logo-box">

<div class="logo-icon">
🏦
</div>

<div>
<div class="logo-text">
SecureBank AI
</div>

<div class="logo-sub">
FRAUD INTELLIGENCE SYSTEM
</div>
</div>

</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("""
<div class="status">
<span class="status-dot"></span>
AI SYSTEM ONLINE
</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Dashboard",
        "🔍 Analyze Transaction",
        "🤖 Model Information"
    ]
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Powered by Machine Learning"
)

st.sidebar.caption(
    "Random Forest Classifier"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">Fraud Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Real-time machine learning based banking transaction analysis'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # DATASET STATISTICS
    # -----------------------------------------------------

    total_transactions = len(df)

    if "Class" in df.columns:

        fraud_transactions = int(
            df["Class"].sum()
        )

        legitimate_transactions = (
            total_transactions - fraud_transactions
        )

        fraud_rate = (
            fraud_transactions /
            total_transactions *
            100
        )

    else:

        fraud_transactions = 0
        legitimate_transactions = total_transactions
        fraud_rate = 0


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-icon">💳</div>

        <div class="metric-title">
        TRANSACTIONS
        </div>

        <div class="metric-value">
        {total_transactions:,}
        </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-icon">🚨</div>

        <div class="metric-title">
        FRAUD CASES
        </div>

        <div class="metric-value">
        {fraud_transactions:,}
        </div>

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-icon">✅</div>

        <div class="metric-title">
        LEGITIMATE
        </div>

        <div class="metric-value">
        {legitimate_transactions:,}
        </div>

        </div>
        """, unsafe_allow_html=True)


    with col4:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-icon">📈</div>

        <div class="metric-title">
        FRAUD RATE
        </div>

        <div class="metric-value">
        {fraud_rate:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        if "Class" in df.columns:

            chart_data = pd.DataFrame({
                "Transaction Type": [
                    "Legitimate",
                    "Fraud"
                ],

                "Count": [
                    legitimate_transactions,
                    fraud_transactions
                ]
            })

            st.bar_chart(
                chart_data.set_index(
                    "Transaction Type"
                )
            )


    with col2:

        st.markdown("""
        <div class="analysis-card">

        <h3 style="color:white;">
        🛡️ Security Overview
        </h3>

        <p style="color:#94a3b8;">
        The AI system analyzes transaction patterns
        and identifies potentially fraudulent activity.
        </p>

        <br>

        <b style="color:#34d399;">
        ● Model Status: Active
        </b>

        <br><br>

        <b style="color:#60a5fa;">
        ● Detection Engine: Random Forest
        </b>

        <br><br>

        <b style="color:#fbbf24;">
        ● Feature Processing: Enabled
        </b>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TRANSACTION ANALYSIS
# =========================================================

elif page == "🔍 Analyze Transaction":

    st.markdown(
        '<div class="main-title">Analyze Transaction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Enter transaction information and let the AI model assess the risk.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">💳 Transaction Features</div>',
        unsafe_allow_html=True
    )


    input_data = {}


    # Split features into columns

    for i in range(0, len(feature_names), 2):

        col1, col2 = st.columns(2)


        # First feature

        feature = feature_names[i]

        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        mean_value = float(df[feature].mean())


        with col1:

            input_data[feature] = st.number_input(
                feature,
                min_value=min_value,
                max_value=max_value,
                value=mean_value,
                key=f"input_{feature}"
            )


        # Second feature

        if i + 1 < len(feature_names):

            feature2 = feature_names[i + 1]

            min_value2 = float(
                df[feature2].min()
            )

            max_value2 = float(
                df[feature2].max()
            )

            mean_value2 = float(
                df[feature2].mean()
            )


            with col2:

                input_data[feature2] = st.number_input(
                    feature2,
                    min_value=min_value2,
                    max_value=max_value2,
                    value=mean_value2,
                    key=f"input_{feature2}"
                )


    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    st.markdown("")


    analyze = st.button(
        "🔎 ANALYZE TRANSACTION"
    )


    if analyze:

        try:

            # Create dataframe

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_names
            )


            # -------------------------------------------------
            # SCALE DATA
            # -------------------------------------------------

            input_scaled = scaler.transform(
                input_df
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(
                input_scaled
            )


            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_scaled
                )

                fraud_probability = float(
                    probabilities[0][1]
                )

            else:

                fraud_probability = float(
                    prediction[0]
                )


            fraud_percentage = (
                fraud_probability * 100
            )


            # -------------------------------------------------
            # RISK LEVEL
            # -------------------------------------------------

            if fraud_percentage >= 70:

                risk = "HIGH RISK"
                risk_class = "risk-high"

            elif fraud_percentage >= 30:

                risk = "MEDIUM RISK"
                risk_class = "risk-medium"

            else:

                risk = "LOW RISK"
                risk_class = "risk-low"


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '🤖 AI Prediction'
                '</div>',
                unsafe_allow_html=True
            )


            if prediction[0] == 1:

                st.markdown(f"""
                <div class="fraud-card">

                <div class="prediction-icon">
                🚨
                </div>

                <div class="prediction-title">
                FRAUDULENT TRANSACTION
                </div>

                <div class="prediction-text">
                The AI model has detected suspicious
                transaction characteristics.
                </div>

                <br>

                <div class="{risk_class}">
                {risk}
                </div>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="safe-card">

                <div class="prediction-icon">
                🛡️
                </div>

                <div class="prediction-title">
                TRANSACTION APPEARS LEGITIMATE
                </div>

                <div class="prediction-text">
                No significant fraudulent pattern was
                detected by the AI model.
                </div>

                <br>

                <div class="{risk_class}">
                {risk}
                </div>

                </div>
                """, unsafe_allow_html=True)


            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📈 Fraud Probability'
                '</div>',
                unsafe_allow_html=True
            )


            col1, col2 = st.columns([2, 1])


            with col1:

                st.progress(
                    fraud_probability
                )


            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{fraud_percentage:.2f}%"
                )


            # -------------------------------------------------
            # TRANSACTION SUMMARY
            # -------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '📋 Transaction Summary'
                '</div>',
                unsafe_allow_html=True
            )


            st.dataframe(
                input_df,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"❌ Prediction Error: {e}"
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.markdown(
        '<div class="main-title">AI Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Machine learning architecture used for fraud detection'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-icon">🤖</div>

        <div class="metric-title">
        ALGORITHM
        </div>

        <div class="metric-value"
        style="font-size:22px;">
        Random Forest
        </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-icon">🔢</div>

        <div class="metric-title">
        FEATURES
        </div>

        <div class="metric-value">
        {len(feature_names)}
        </div>

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown("""
        <div class="metric-card">

        <div class="metric-icon">🎯</div>

        <div class="metric-title">
        TASK
        </div>

        <div class="metric-value"
        style="font-size:22px;">
        Classification
        </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">⚙️ Detection Pipeline</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="analysis-card">

    <h3 style="color:white;">
    Transaction
    </h3>

    <p style="color:#94a3b8;">
    ↓
    </p>

    <h3 style="color:#60a5fa;">
    Data Preprocessing
    </h3>

    <p style="color:#94a3b8;">
    ↓
    </p>

    <h3 style="color:#38bdf8;">
    Feature Scaling
    </h3>

    <p style="color:#94a3b8;">
    ↓
    </p>

    <h3 style="color:#a78bfa;">
    Random Forest Classifier
    </h3>

    <p style="color:#94a3b8;">
    ↓
    </p>

    <h3 style="color:#34d399;">
    Fraud Probability
    </h3>

    <p style="color:#94a3b8;">
    ↓
    </p>

    <h3 style="color:#fbbf24;">
    Risk Assessment
    </h3>

    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">🔐 System Capabilities</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("""
        <div class="analysis-card">

        <h3 style="color:white;">
        ✅ Fraud Detection
        </h3>

        <p style="color:#94a3b8;">
        Identifies potentially fraudulent transactions
        using the trained Random Forest classifier.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="analysis-card">

        <h3 style="color:white;">
        📊 Probability Analysis
        </h3>

        <p style="color:#94a3b8;">
        Provides a probability score to help evaluate
        transaction risk.
        </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🏦 <b>SecureBank AI</b> &nbsp; | &nbsp;
AI-Powered Bank Fraud Detection System
<br><br>
Machine Learning • Random Forest • Streamlit

</div>
""", unsafe_allow_html=True)
