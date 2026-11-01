"""
Hybrid Fraud Prediction Service for SecureBank AI.
Integrates:
1. Feature Validation & Alignment (models/feature_columns.json)
2. Standardization Transform (models/scaler.pkl)
3. Supervised Classification (Random Forest / Logistic Regression)
4. Unsupervised Anomaly Detection (Isolation Forest)
5. Risk Intelligence Scoring Engine (0-100 calibrated score)
6. Rule-Based Expert System (Forward-chaining rules)
7. SQLite Audit Logging
"""

import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Optional

from src.risk_engine import calculate_risk_score
from src.rules import apply_fraud_rules
from src.anomaly import score_anomaly
from src.database import log_transaction

class HybridFraudPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            alt_dir = os.path.join("..", models_dir)
            if os.path.exists(alt_dir):
                self.models_dir = alt_dir
                
        # 1. Load Feature Schema
        feature_path = os.path.join(self.models_dir, "feature_columns.json")
        with open(feature_path, "r") as f:
            self.feature_columns = json.load(f)
            
        # 2. Load Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # 3. Load Supervised Model (Random Forest default)
        rf_path = os.path.join(self.models_dir, "random_forest.pkl")
        self.rf_model = joblib.load(rf_path)
        
        # 4. Load Isolation Forest
        self.iso_path = os.path.join(self.models_dir, "isolation_forest.pkl")
        self.has_isolation_forest = os.path.exists(self.iso_path)

    def prepare_dataframe(self, input_data: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Aligns input data with the exact 30 feature schema."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            df = pd.DataFrame([input_data.to_dict()])
        else:
            df = input_data.copy()
            
        # Impute missing feature columns with 0.0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
                
        # Return in exact column order
        return df[self.feature_columns]

    def predict_single(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        transaction_id: Optional[str] = None,
        persist_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Runs complete hybrid fraud inference on a single transaction record.
        """
        if transaction_id is None:
            transaction_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
            
        df = self.prepare_dataframe(input_data)
        
        # Extract metadata
        amount_usd = float(df["Amount"].iloc[0]) if "Amount" in df.columns else 0.0
        time_sec = float(df["Time"].iloc[0]) if "Time" in df.columns else 0.0
        hour_elapsed = int(time_sec // 3600)
        hour_of_day = hour_elapsed % 24
        
        # Standardize features and wrap in DataFrame to preserve feature names
        scaled_array = self.scaler.transform(df)
        scaled_features = pd.DataFrame(scaled_array, columns=self.feature_columns)
        
        # 1. Supervised Machine Learning (Random Forest)
        rf_probs = self.rf_model.predict_proba(scaled_features)[0]
        fraud_prob = float(rf_probs[1]) if len(rf_probs) > 1 else float(rf_probs[0])
        prediction_label = "Fraud" if fraud_prob >= 0.50 else "Legitimate"
        
        # 2. Unsupervised Anomaly Detection (Isolation Forest)
        anomaly_res = score_anomaly(scaled_features, self.iso_path)
        anomaly_flag = anomaly_res["anomaly_flag"]
        anomaly_score = anomaly_res["normalized_score"]
        
        # 3. Risk Intelligence Scoring Engine (0 - 100)
        risk_report = calculate_risk_score(
            fraud_probability=fraud_prob,
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            amount_usd=amount_usd
        )
        
        # 4. Expert Rule Engine
        alerts = apply_fraud_rules(
            fraud_probability=fraud_prob,
            anomaly_flag=anomaly_flag,
            amount_usd=amount_usd,
            hour_of_day=hour_of_day
        )
        
        result = {
            "transaction_id": transaction_id,
            "prediction": prediction_label,
            "fraud_probability": round(fraud_prob, 4),
            "anomaly_detected": bool(anomaly_flag),
            "anomaly_score": round(anomaly_score, 4),
            "risk_score": risk_report["risk_score"],
            "risk_level": risk_report["risk_level"],
            "action_code": risk_report["action_code"],
            "recommendation": risk_report["recommendation"],
            "badge_color": risk_report["badge_color"],
            "amount_usd": round(amount_usd, 2),
            "hour_of_day": hour_of_day,
            "alerts": alerts,
            "breakdown": risk_report["breakdown"]
        }
        
        # 5. Persist to SQLite Database
        if persist_to_db:
            try:
                log_transaction(
                    transaction_id=transaction_id,
                    amount=amount_usd,
                    fraud_probability=fraud_prob,
                    risk_score=risk_report["risk_score"],
                    risk_level=risk_report["risk_level"],
                    prediction=prediction_label,
                    anomaly_flag=anomaly_flag,
                    action_code=risk_report["action_code"],
                    hour_elapsed=hour_elapsed,
                    features_json=df.to_json(orient="records"),
                    alerts=alerts
                )
            except Exception as e:
                print(f"Warning: Failed to persist to database: {e}")
                
        return result

    def predict_batch(self, batch_df: pd.DataFrame, persist_to_db: bool = False) -> pd.DataFrame:
        """
        Runs vectorized hybrid inference on a batch DataFrame (e.g. uploaded CSV).
        """
        aligned_df = self.prepare_dataframe(batch_df)
        scaled_array = self.scaler.transform(aligned_df)
        scaled_features = pd.DataFrame(scaled_array, columns=self.feature_columns)
        
        # Supervised probabilities
        probs = self.rf_model.predict_proba(scaled_features)[:, 1]
        
        results = []
        for i in range(len(batch_df)):
            row_dict = batch_df.iloc[i].to_dict()
            txn_id = row_dict.get("transaction_id", f"TXN-BATCH-{i+1:04d}")
            res = self.predict_single(row_dict, transaction_id=txn_id, persist_to_db=persist_to_db)
            results.append({
                "transaction_id": res["transaction_id"],
                "prediction": res["prediction"],
                "fraud_probability": res["fraud_probability"],
                "risk_score": res["risk_score"],
                "risk_level": res["risk_level"],
                "action_code": res["action_code"],
                "anomaly_detected": res["anomaly_detected"],
                "amount": res["amount_usd"],
                "recommendation": res["recommendation"]
            })
            
        return pd.DataFrame(results)

# Global singleton predictor instance
_PREDICTOR_INSTANCE = None

def get_predictor() -> HybridFraudPredictor:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = HybridFraudPredictor()
    return _PREDICTOR_INSTANCE
