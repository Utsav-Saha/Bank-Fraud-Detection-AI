"""
Anomaly Detection Module for SecureBank AI.
Loads and executes Isolation Forest models on transaction feature vectors.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union

_MODEL_CACHE = {}

def get_isolation_model(model_path: str = "models/isolation_forest.pkl"):
    """Caches and returns the loaded Isolation Forest artifact."""
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            alt_path = os.path.join("..", model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                return None
        _MODEL_CACHE[model_path] = joblib.load(model_path)
    return _MODEL_CACHE[model_path]

def score_anomaly(
    features_array: Union[np.ndarray, pd.DataFrame],
    model_path: str = "models/isolation_forest.pkl"
) -> Dict[str, Any]:
    """
    Evaluates anomaly likelihood for a given pre-scaled feature vector.
    
    Returns:
        Dict with anomaly_flag (1 or 0), raw_score, normalized_score, and is_outlier bool.
    """
    model = get_isolation_model(model_path)
    if model is None:
        return {
            "anomaly_flag": 0,
            "raw_score": 0.0,
            "normalized_score": 0.0,
            "is_outlier": False
        }
        
    if isinstance(features_array, pd.DataFrame):
        arr = features_array.values
    else:
        arr = features_array
        
    if len(arr.shape) == 1:
        arr = arr.reshape(1, -1)
        
    raw_pred = model.predict(arr)[0]
    raw_score = model.decision_function(arr)[0]
    
    # Invert decision function: negative -> anomalous, so -raw_score makes higher = more abnormal
    norm_score = -float(raw_score)
    is_anomaly = (raw_pred == -1)
    
    return {
        "anomaly_flag": 1 if is_anomaly else 0,
        "raw_score": round(float(raw_score), 4),
        "normalized_score": round(norm_score, 4),
        "is_outlier": bool(is_anomaly)
    }
