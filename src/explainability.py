"""
Explainable AI (XAI) Service for SecureBank AI.
Provides local and global feature explanations using SHAP (SHapley Additive exPlanations)
and feature contribution breakdowns for financial audits.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class FraudExplainer:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            alt_dir = os.path.join("..", models_dir)
            if os.path.exists(alt_dir):
                self.models_dir = alt_dir
                
        self.rf_model = joblib.load(os.path.join(self.models_dir, "random_forest.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        
        # Load feature names
        import json
        with open(os.path.join(self.models_dir, "feature_columns.json"), "r") as f:
            self.feature_columns = json.load(f)
            
        self.tree_explainer = None
        self._init_shap()

    def _init_shap(self):
        """Attempts to initialize SHAP TreeExplainer with graceful fallback."""
        try:
            import shap
            # Use small background sample if needed
            self.tree_explainer = shap.TreeExplainer(self.rf_model)
        except Exception as e:
            print(f"SHAP explainer init info: {e}. Utilizing feature contribution fallback.")
            self.tree_explainer = None

    def explain_transaction(
        self,
        transaction_df: pd.DataFrame,
        top_n: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Calculates local feature contribution values for a single transaction.
        Returns top_n positive (fraud-increasing) and negative (fraud-decreasing) features.
        """
        # Ensure exact columns
        df = transaction_df.copy()
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_columns]
        
        scaled_features = self.scaler.transform(df)
        
        contributions = []
        
        if self.tree_explainer is not None:
            try:
                shap_vals = self.tree_explainer.shap_values(scaled_features)
                # Binary classification: index 1 is the Fraud class
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    vals = shap_vals[1][0]
                elif hasattr(shap_vals, "values"):
                    vals = shap_vals.values[0, :, 1] if len(shap_vals.values.shape) > 2 else shap_vals.values[0]
                else:
                    vals = shap_vals[0]
                    
                for col, val, raw_val in zip(self.feature_columns, vals, df.iloc[0]):
                    contributions.append({
                        "feature": col,
                        "shap_value": float(val),
                        "feature_value": float(raw_val),
                        "impact": "INCREASES_RISK" if val > 0 else "DECREASES_RISK",
                        "abs_impact": abs(float(val))
                    })
            except Exception as e:
                print(f"SHAP runtime exception: {e}. Using tree feature importances.")
                contributions = self._fallback_contributions(scaled_features[0], df.iloc[0])
        else:
            contributions = self._fallback_contributions(scaled_features[0], df.iloc[0])
            
        # Sort by absolute impact descending
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return contributions[:top_n]

    def _fallback_contributions(self, scaled_vector: np.ndarray, raw_row: pd.Series) -> List[Dict[str, Any]]:
        """
        Calculates feature influence based on Random Forest feature importances
        weighted by the standardized distance from the mean.
        """
        importances = getattr(self.rf_model, "feature_importances_", np.ones(len(self.feature_columns)) / len(self.feature_columns))
        results = []
        for col, imp, z_score, raw_val in zip(self.feature_columns, importances, scaled_vector, raw_row):
            # Directional impact: outlier in critical components contributes to fraud score
            # In credit card fraud, extreme negative values in V14, V10, V12, V17 strongly indicate fraud
            is_negative_indicator = col in ["V14", "V10", "V12", "V17", "V11", "V4"]
            if is_negative_indicator:
                impact_dir = -1.0 * z_score * imp
            else:
                impact_dir = z_score * imp
                
            results.append({
                "feature": col,
                "shap_value": round(float(impact_dir), 4),
                "feature_value": round(float(raw_val), 4),
                "impact": "INCREASES_RISK" if impact_dir > 0 else "DECREASES_RISK",
                "abs_impact": abs(float(impact_dir))
            })
        return results

    def get_global_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Returns the top global features driving model decisions."""
        importances = getattr(self.rf_model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values("importance", ascending=False)
        return df.head(top_n)

# Global singleton
_EXPLAINER_INSTANCE = None

def get_explainer() -> FraudExplainer:
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = FraudExplainer()
    return _EXPLAINER_INSTANCE
