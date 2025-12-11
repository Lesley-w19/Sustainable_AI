# prompt_improver.py
from typing import Dict

from models import PromptData
from prompt_engine import PromptVariant, optimize_prompt


class PromptImprover:
    """
    Uses the multi-variant optimizer from prompt_engine to
    generate an improved version of the prompt.
    """

    @staticmethod
    def improve(
        prompt: PromptData,
        layers: int,
        training_hours: float,
        flops_hr: float,
        similarity_min: float = 0.80,
    ) -> Dict:
        """
        Build a PromptVariant from the original PromptData and
        run the optimizer.
        """
        # ✅ Convert PromptData -> PromptVariant (strings only)
        base_variant = PromptVariant(
            role=prompt.role,
            context=prompt.context,
            expectations=prompt.expectations,
        )

        # ✅ Now pass the variant to the optimizer
        return optimize_prompt(
            prompt=base_variant,
            layers=layers,
            training_hours=training_hours,
            flops_per_hour=flops_hr,
            similarity_min=similarity_min,
        )


def run_improve_prompt(
    analyzer,              # kept for backwards compatibility, not used
    improver: PromptImprover,
    energy_estimator,      # kept for backwards compatibility, not used
    prompt: PromptData,    # ✅ full PromptData, not a combined string
    layers: int,
    training_hours: float,
    flops_hr: float,
) -> Dict:
    """
    Wrapper with a similar signature to the old version, but it now
    uses the full PromptData object instead of a single combined string.
    """
    result = improver.improve(
        prompt=prompt,
        layers=layers,
        training_hours=training_hours,
        flops_hr=flops_hr,
    )
    return result
