# styles.py
import streamlit as st

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #eef6f0 0%, #ffffff 100%);
    font-family: system-ui, sans-serif;
}

/* Header */
.main > div:first-child h1 {
    text-align: center;
}

/* Panels */
.green-section {
    background: #f7fafc;
    padding: 16px;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

/* Buttons */
.blue-btn button {
    background-color: #2f855a !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    width: 100% !important;
    height: 45px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.blue-btn button:hover {
    background-color: #276749 !important;
    transform: scale(1.03) !important;
}

.orange-btn button {
    background-color: #FF8C00 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    width: 100% !important;
    height: 45px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.orange-btn button:hover {
    background-color: #FF9F33 !important;
    transform: scale(1.03) !important;
}

/* Loading text */
.loading-text {
    font-size: 1.1em;
    margin-bottom: 10px;
    color: #2f855a;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { opacity: 0.3; }
  50% { opacity: 1; }
  100% { opacity: 0.3; }
}
</style>
"""

def apply_custom_css():
    """Injects the custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
