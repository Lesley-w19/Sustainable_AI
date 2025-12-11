# anomaly_detector.py
from __future__ import annotations

from pyexpat import model
from typing import Dict, Tuple, Optional
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# These are the same numeric fields you log in logging_utils.log_energy_event
# :contentReference[oaicite:0]{index=0}
FEATURE_COLUMNS = [
    "energy_kwh",
    "layers",
    "training_hours",
    "flops_hr",
    "tokens",
    "type_token_ratio",
    "avg_sentence_len",
    "stopword_ratio",
    "sections",
]


def _load_log_dataframe(log_path: str = "energy_logs.csv") -> Optional[pd.DataFrame]:
    """Load the historical energy log as a DataFrame, if it exists."""
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    # Need at least some history to train anything meaningful
    if df.empty or len(df) < 30:
        return None
    return df


def _train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> IsolationForest:
    """
    Train an Isolation Forest on past usage patterns.
    Assumes most historical prompts are 'normal'.
    """
    X = df[FEATURE_COLUMNS]
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    model.fit(X)
    return model


def detect_energy_anomaly(
    features: Dict[str, float],
    energy_kwh: float,
    layers: int,
    training_hours: float,
    flops_hr: float,
    log_path: str = "energy_logs.csv",
) -> Dict[str, object]:
    """
    Run anomaly detection for the current prompt.

    Returns a dict with:
      - is_anomaly: bool
      - score: float (lower = more abnormal)
      - reason: str (human-readable explanation)
    """
    df = _load_log_dataframe(log_path)
    if df is None:
        # Not enough history yet
        return {
            "is_anomaly": False,
            "score": 0.0,
            "reason": "Not enough historical data to train anomaly model yet.",
        }

    model = _train_isolation_forest(df)

    # Build a single-row vector in the same feature order as the training set
    row = {
        "energy_kwh": float(energy_kwh),
        "layers": float(layers),
        "training_hours": float(training_hours),
        "flops_hr": float(flops_hr),
        "tokens": float(features.get("tokens", 0.0)),
        "type_token_ratio": float(features.get("type_token_ratio", 0.0)),
        "avg_sentence_len": float(features.get("avg_sentence_len", 0.0)),
        "stopword_ratio": float(features.get("stopword_ratio", 0.0)),
        "sections": float(features.get("sections", 0.0)),
    }

    x = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    
    label = model.predict(x)[0]         # -1 = anomaly, 1 = normal
    score = float(model.decision_function(x)[0])
    is_anomaly = (label == -1)

    # --- Explain WHY it's abnormal using simple statistics ---
    reasons = []

    # Pre-compute some thresholds on historical data
    q95_tokens = df["tokens"].quantile(0.95)
    q95_energy = df["energy_kwh"].quantile(0.95)
    q95_flops = df["flops_hr"].quantile(0.95)
    q95_layers = df["layers"].quantile(0.95)

    # Optional “FLOPs per layer” ratio
    df["flops_per_layer"] = df["flops_hr"] / df["layers"].clip(lower=1)
    current_fpl = row["flops_hr"] / max(row["layers"], 1.0)
    q95_fpl = df["flops_per_layer"].quantile(0.95)

    if row["tokens"] > q95_tokens:
        reasons.append("token count is in the top 5% of all logged prompts.")
    if row["energy_kwh"] > q95_energy:
        reasons.append("predicted energy consumption is in the top 5% of historical usage.")
    if row["flops_hr"] > q95_flops:
        reasons.append("FLOPs/hr is in the top 5% of past configurations.")
    if row["layers"] > q95_layers:
        reasons.append("number of layers is unusually high compared to past prompts.")
    if current_fpl > q95_fpl:
        reasons.append("FLOPs-per-layer ratio is abnormally high.")

    if not reasons and is_anomaly:
        reasons.append("overall usage pattern is unusual compared to normal prompts.")

    reason_text = " ".join(reasons) if reasons else "Within normal range based on current history."

    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "reason": reason_text,
    }
