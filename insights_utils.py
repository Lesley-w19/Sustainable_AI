# insight_utils.py
from helpers import _pct_change

def build_improvement_insights(
    feats_before: dict,
    feats_after: dict,
    energy_before: float,
    energy_after: float,
    similarity: float,
) -> str:
    """
    Create a short natural-language explanation of what changed
    and why it matters for the model.
    """
    tokens_before = feats_before["tokens"]
    tokens_after = feats_after["tokens"]

    token_pct = _pct_change(tokens_before, tokens_after)        # expect negative
    stop_pct = _pct_change(feats_before["stopword_ratio"], feats_after["stopword_ratio"])
    ttr_pct = _pct_change(feats_before["type_token_ratio"], feats_after["type_token_ratio"])
    sentlen_pct = _pct_change(feats_before["avg_sentence_len"], feats_after["avg_sentence_len"])
    energy_pct = _pct_change(energy_before, energy_after)

    # Make them human-friendly
    def fmt_pct(x: float) -> str:
        sign = "↓" if x < 0 else "↑"
        return f"{sign}{abs(x):.1f}%"

    return (
        "### What changed and why it matters\n"
        f"- **Length:** Tokens went from `{tokens_before:.0f}` to `{tokens_after:.0f}` "
        f"({fmt_pct(token_pct)}), meaning the prompt is more concise and avoids redundancy.\n"
        f"- **Information density:** Type–token ratio increased from "
        f"`{feats_before['type_token_ratio']:.3f}` to `{feats_after['type_token_ratio']:.3f}` "
        f"({fmt_pct(ttr_pct)}), so each word carries more unique information.\n"
        f"- **Clarity:** Average sentence length changed from "
        f"`{feats_before['avg_sentence_len']:.1f}` to `{feats_after['avg_sentence_len']:.1f}` "
        f"({fmt_pct(sentlen_pct)}), making instructions easier for the model to follow.\n"
        f"- **Filler words:** Stopword ratio moved from "
        f"`{feats_before['stopword_ratio']:.3f}` to `{feats_after['stopword_ratio']:.3f}` "
        f"({fmt_pct(stop_pct)}), so the prompt is more direct and less wordy.\n"
        f"- **Energy footprint (toy model):** Predicted energy changed from "
        f"`{energy_before:.4f}` to `{energy_after:.4f}` kWh "
        f"({fmt_pct(energy_pct)}), meaning the improved prompt is designed to be more "
        f"computationally efficient.\n"
        f"- **Meaning preserved:** Semantic similarity is `{similarity:.3f}`, showing that the "
        f"optimized prompt still expresses almost the same intent while being cleaner.\n"
    )
