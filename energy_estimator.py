# energy_estimator.py
from typing import Dict

from prompt_engine import predict_energy_kwh


# class EnergyEstimator:
#     """
#     Backwards-compatible wrapper.

#     If you call estimate_energy_kwh(layers, training_hours, flops_per_hour)
#     WITHOUT features, it assumes a neutral feature profile.
#     If you have text features, pass them as features=... for better estimates.
#     """

#     @staticmethod
#     def estimate_energy_kwh(
#         layers: int,
#         training_hours: float,
#         flops_per_hour: float,
#         features: Dict[str, float] | None = None,
#     ) -> float:
#         if features is None:
#             # Neutral / average feature profile
#             features = {
#                 "tokens": 100.0,
#                 "type_token_ratio": 0.5,
#                 "avg_sentence_len": 20.0,
#                 "stopword_ratio": 0.4,
#                 "sections": 2.0,
#             }
#         return predict_energy_kwh(features, layers, training_hours, flops_per_hour)

import joblib
import numpy as np

class EnergyEstimator:

    def __init__(self):
        self.model = joblib.load("energy_predictor.pkl")

    def estimate_energy_kwh(
        self, layers, training_hours, flops_per_hour, features=None
    ):
        if features is None:
            features = {
                "tokens": 100.0,
                "type_token_ratio": 0.5,
                "avg_sentence_len": 20.0,
                "stopword_ratio": 0.4,
                "sections": 2,
            }

        row = np.array([
            features["tokens"],
            features["type_token_ratio"],
            features["avg_sentence_len"],
            features["stopword_ratio"],
            features["sections"],
            layers,
            training_hours,
            flops_per_hour,
        ]).reshape(1, -1)

        return float(self.model.predict(row)[0])
