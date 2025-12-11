# app.py
import streamlit as st

from styles import apply_custom_css
from models import PromptData
from prompt_analyzer import PromptAnalyzer
from prompt_improver import PromptImprover
from energy_estimator import EnergyEstimator
from ui_components import render_input_panel, render_results_panel


def main():
    # ---------- Page config ----------
    st.set_page_config(
        page_title="GreenMind: Prompt Transformation",
        page_icon="🌱",
        layout="wide",
    )

    apply_custom_css()

    st.title("GreenMind: Prompt Transformation")
    st.write("Analyze your prompt and generate a **more energy-aware** version.")

    # ---------- Layout ----------
    col_left, col_right = st.columns([2, 1])

    analyzer = PromptAnalyzer()
    improver = PromptImprover()
    energy_estimator = EnergyEstimator()

    # ---------- Left: Inputs ----------
    with col_left:
        prompt, params, actions = render_input_panel()

    # ---------- Right: Results ----------
    with col_right:
       
        render_results_panel(
            prompt=prompt,
            params=params,
            actions=actions,
            analyzer=analyzer,
            improver=improver,
            energy_estimator=energy_estimator,
        )


if __name__ == "__main__":
    main()
