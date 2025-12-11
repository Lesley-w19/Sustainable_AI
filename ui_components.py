# ui_components.py
import streamlit as st

from models import PromptData
from prompt_analyzer import PromptAnalyzer, run_check_prompt
from prompt_improver import PromptImprover, run_improve_prompt
from energy_estimator import EnergyEstimator

from insights_utils import build_improvement_insights
from logging_utils import log_energy_event
from anomaly_detector import detect_energy_anomaly

from visualization import (
    feature_comparison_bar,
    energy_distribution_hist,
    token_breakdown_bar,
    anomaly_score_bar,
)


# ==========================
# Input panel
# ==========================
def render_input_panel() -> tuple[PromptData, dict, dict]:
    """
    Renders the left side:
    - text inputs
    - parameter controls
    - action buttons

    Returns:
        prompt (PromptData)
        params dict with ('layers', 'training_hours', 'flops_hr')
        actions dict with ('check_clicked', 'improve_clicked')
    """
    st.markdown('<div class="green-section">', unsafe_allow_html=True)
    st.subheader("Prompt input")

    role = st.text_area("Role", height=80, key="role")
    context = st.text_area("Context", height=140, key="context")
    expectations = st.text_area("Expectations", height=80, key="expectations")

    prompt = PromptData(role=role, context=context, expectations=expectations)

    st.markdown("### Set parameters")
    p1, p2, p3 = st.columns(3)

    with p1:
        layers = st.number_input("# Layers", min_value=1, value=4, step=1)
    with p2:
        training_hours = st.number_input(
            "Training time (h)", min_value=0.0, value=2.0, step=0.5
        )
    with p3:
        flops_hr = st.number_input(
            "FLOPs/hr",
            min_value=0.0,
            value=1e20,
            step=1e18,
            format="%.2e",
        )

    params = {
        "layers": layers,
        "training_hours": training_hours,
        "flops_hr": flops_hr,
    }

    st.markdown("#### Actions")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="blue-btn">', unsafe_allow_html=True)
        check_clicked = st.button("Check Prompt", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="orange-btn">', unsafe_allow_html=True)
        improve_clicked = st.button("Make It Better", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    actions = {
        "check_clicked": check_clicked,
        "improve_clicked": improve_clicked,
    }


    st.markdown("</div>", unsafe_allow_html=True)

    return prompt, params, actions


# ==========================
# Results panel
# ==========================
def render_results_panel(
    prompt: PromptData,
    params: dict,
    actions: dict,
    analyzer: PromptAnalyzer,
    improver: PromptImprover,
    energy_estimator: EnergyEstimator,
):
    """
    Renders the right panel based on which action was clicked.
    Delegates almost all heavy lifting to the ML-style helpers.

    - Logs each run to CSV/JSONL
    - Shows improvement insights
    """

    # reconstruct the combined string for display and analysis
    combined_prompt = prompt.combined()

    st.markdown('<div class="green-section">', unsafe_allow_html=True)
    st.subheader("Results")

    results_box = st.empty()

    check_btn = actions.get("check_clicked", False)
    improve_btn = actions.get("improve_clicked", False)

    layers = params["layers"]
    training_hours = params["training_hours"]
    flops_hr = params["flops_hr"]

    # ---------- Check Prompt ----------
    if check_btn:
        if not combined_prompt.strip():
            st.warning("Please enter at least one of Role, Context, or Expectations.")
        else:
            with results_box.container():
                st.markdown(
                    '<p class="loading-text">Analyzing prompt…</p>',
                    unsafe_allow_html=True,
                )

                result = run_check_prompt(
                    analyzer=analyzer,
                    combined_prompt=combined_prompt,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )

                features = result["features"]
                energy = result["energy"]

                # --- Core UI ---
                st.markdown("### `Check Prompt Result`")
                st.markdown("**Original prompt:**")
                st.write(combined_prompt)

                st.metric("Estimated energy (ML-style)", f"{energy:.4f} kWh")
                st.markdown("**Features:**")
                st.write(f"- Tokens: `{features['tokens']:.0f}`")
                st.write(f"- Type–token ratio: `{features['type_token_ratio']:.3f}`")
                st.write(
                    f"- Avg. sentence length: `{features['avg_sentence_len']:.1f}` tokens"
                )
                st.write(
                    f"- Stopword ratio: `{features['stopword_ratio']:.3f}`"
                )
                st.write(
                    f"- Sections detected: `{features['sections']:.0f}`"
                )

                # --- Token breakdown per section ---
                st.markdown("#### Token usage per section")
                fig_tokens = token_breakdown_bar(prompt)
                st.pyplot(fig_tokens)

                # --- Energy distribution vs history ---
                st.markdown("#### Energy usage in historical context")
                fig_hist = energy_distribution_hist(energy_kwh=energy)
                if fig_hist is not None:
                    st.pyplot(fig_hist)
                else:
                    st.info("Not enough historical data to show energy distribution yet.")


                # --- Logging for transparency / reporting ---
                log_energy_event(
                    action="check",
                    variant="original",
                    prompt_text=combined_prompt,
                    features=features,
                    energy_kwh=energy,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )
                
                 # --- Anomaly detection: flag unusually high-energy prompts ---
                anomaly = detect_energy_anomaly(
                    features=features,
                    energy_kwh=energy,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )

                if anomaly["is_anomaly"]:
                    st.error(
                        f"⚠ High-energy prompt detected (score={anomaly['score']:.3f}). "
                        f"{anomaly['reason']}"
                    )
                else:
                    st.info(
                        f"This prompt is within the normal energy range. "
                        f"(Anomaly score={anomaly['score']:.3f})"
                    )

    # ---------- Make It Better ----------
    if improve_btn:
        if not combined_prompt.strip():
            st.warning("Please enter at least one of Role, Context, or Expectations.")
        else:
            with results_box.container():
                st.markdown(
                    '<p class="loading-text">Improving prompt…</p>',
                    unsafe_allow_html=True,
                )

                result = run_improve_prompt(
                    analyzer=analyzer,
                    improver=improver,
                    energy_estimator=energy_estimator,
                    prompt=prompt,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )

                improved = result["improved"]
                feats_before = result["features_before"]
                feats_after = result["features_after"]
                energy_before = result["predicted_kwh_before"]
                energy_after = result["predicted_kwh_after"]
                sim = result["similarity"]

                improved_text = (
                    f"Role: {improved.role}\n\n"
                    f"Context: {improved.context}\n\n"
                    f"Expectations: {improved.expectations}"
                )

                # --- Core UI ---
                st.markdown("### Make It Better Result")
                st.markdown("**Original prompt:**")
                st.write(combined_prompt)

                st.markdown("**Improved prompt:**")
                st.write(improved_text)

                st.write(f"**Semantic Similarity:** `{sim:.3f}`")

                st.metric(
                    "Predicted energy (before → after)",
                    f"{energy_before:.4f} → {energy_after:.4f} kWh",
                    delta=f"{energy_after - energy_before:.4f} kWh",
                )

                st.markdown("**Features (before → after):**")
                st.write(
                    f"- Tokens: `{feats_before['tokens']:.0f}` "
                    f"→ `{feats_after['tokens']:.0f}`"
                )
                st.write(
                    f"- Type–token ratio: `{feats_before['type_token_ratio']:.3f}` "
                    f"→ `{feats_after['type_token_ratio']:.3f}`"
                )
                st.write(
                    f"- Avg. sentence length: `{feats_before['avg_sentence_len']:.1f}` "
                    f"→ `{feats_after['avg_sentence_len']:.1f}`"
                )
                st.write(
                    f"- Stopword ratio: `{feats_before['stopword_ratio']:.3f}` "
                    f"→ `{feats_after['stopword_ratio']:.3f}`"
                )
                st.write(
                    f"- Sections: `{feats_before['sections']:.0f}` "
                    f"→ `{feats_after['sections']:.0f}`"
                )
                
                # VISUALIZATIONS
                # --- Token breakdown per section ---
                st.markdown("#### Token usage per section")
                fig_tokens = token_breakdown_bar(prompt)
                st.pyplot(fig_tokens)

                # --- Energy distribution vs history ---
                st.markdown("#### Energy usage in historical context")
                fig_hist = energy_distribution_hist(energy_kwh=energy_after)
                if fig_hist is not None:
                    st.pyplot(fig_hist)
                else:
                    st.info("Not enough historical data to show energy distribution yet.")

                # --- Visual comparison of features before vs after ---
                st.markdown("#### Visual comparison: features before vs after")
                fig_feats = feature_comparison_bar(
                    feats_before=feats_before,
                    feats_after=feats_after,
                )
                st.pyplot(fig_feats)


                # Human-readable insight summary
                insights = build_improvement_insights(
                    feats_before=feats_before,
                    feats_after=feats_after,
                    energy_before=energy_before,
                    energy_after=energy_after,
                    similarity=sim,
                )
                st.markdown(insights)

                # --- Logging both original & improved for reporting ---
                log_energy_event(
                    action="improve",
                    variant="original",
                    prompt_text=combined_prompt,
                    features=feats_before,
                    energy_kwh=energy_before,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )
                log_energy_event(
                    action="improve",
                    variant="improved",
                    prompt_text=improved_text,
                    features=feats_after,
                    energy_kwh=energy_after,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )
                
                # Anomaly detection on the IMPROVED prompt
                improved_anomaly = detect_energy_anomaly(
                    features=feats_after,
                    energy_kwh=energy_after,
                    layers=layers,
                    training_hours=training_hours,
                    flops_hr=flops_hr,
                )

                if improved_anomaly["is_anomaly"]:
                    st.warning(
                        "⚠ Even after optimization, this prompt is still high-energy. "
                        + improved_anomaly["reason"]
                    )
                else:
                    st.success(
                        "The improved prompt is now within the normal energy range. "
                        f"(Anomaly score={improved_anomaly['score']:.3f})"
                    )
                
                

    st.markdown("</div>", unsafe_allow_html=True)
