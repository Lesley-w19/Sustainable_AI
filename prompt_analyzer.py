# services.py
from typing import Dict

from prompt_engine import compute_features, jaccard_similarity, predict_energy_kwh


class PromptAnalyzer:
    """
    Thin wrapper around the richer utilities in prompt_engine.
    """

    @staticmethod
    def compute_features(text: str) -> Dict[str, float]:
        return compute_features(text)

    @staticmethod
    def jaccard_similarity(a: str, b: str) -> float:
        return jaccard_similarity(a, b)


def run_check_prompt(
    analyzer: PromptAnalyzer,
    combined_prompt: str,
    layers: int,
    training_hours: float,
    flops_hr: float,
) -> Dict:
    """
    Updated 'Check Prompt' pipeline using richer features and energy model.
    """
    features = analyzer.compute_features(combined_prompt)
    energy = predict_energy_kwh(features, layers, training_hours, flops_hr)

    return {
        "features": features,
        "energy": energy,
    }
