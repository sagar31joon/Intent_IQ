import sys
import os

# Add project root so absolute imports work (core.*, intent_system.*, etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import plotly.express as px
import pandas as pd
import streamlit as st
from core import config
from intent_system.intent_recognizer import IntentRecognizer

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntentIQ Demo",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "recognizer" not in st.session_state:
    st.session_state.recognizer = None
if "loaded_model_info" not in st.session_state:
    st.session_state.loaded_model_info = None

# ---------------------------------------------------------
# TITLE + DESCRIPTION
# ---------------------------------------------------------
st.title("🤖 Intent IQ")
st.markdown("""
### Intelligent Intent Recognition (Text-Only Demo)

This demo lets you test IntentIQ using:
- Sentence Transformer embeddings  
- ML classifier families (LR / SVC)  
- Versioned models (v1, v2, …)  
- Dynamic skill routing  

⚠ **Voice mode disabled online**  
(The Vosk STT model is ~2GB and cannot run on free hosting.)
""")

st.divider()

# ---------------------------------------------------------
# INPUT MODE SELECTOR
# ---------------------------------------------------------
st.subheader("Select Input Mode")
input_mode = st.radio(
    "",
    ["Text Input", "Voice Input (Disabled)"],
    index=0
)

if input_mode == "Voice Input (Disabled)":
    st.error("Voice input is disabled for the online demo.")
    st.stop()


# ---------------------------------------------------------
# MODEL FAMILY SELECTOR
# ---------------------------------------------------------
st.subheader("Select Model Family")

model_families = ["LR", "SVC", "NeuralNet (coming soon)"]

model_choice = st.radio(
    "Choose a model type:",
    model_families,
    index=0
)

if "NeuralNet" in model_choice:
    st.warning("NeuralNet is not implemented yet.")
    st.stop()

real_model_type = model_choice  # LR or SVC


# ---------------------------------------------------------
# MODEL VERSION SELECTOR
# ---------------------------------------------------------
st.subheader("Select Model Version")

model_dir = config.MODEL_TYPES[real_model_type]
versions = [
    f.split("_v")[1].split(".")[0]
    for f in os.listdir(model_dir)
    if f.startswith("classifier_v")
]

if not versions:
    st.error(f"No trained models found in: {model_dir}")
    st.stop()

version_choice = st.radio(
    "Choose model version:",
    versions,
    index=0
)

# ---------------------------------------------------------
# LOAD MODEL BUTTON
# ---------------------------------------------------------
if st.button("Load Model"):
    with st.spinner("Loading selected model..."):
        try:
            recognizer = IntentRecognizer(
                model_type=real_model_type,
                version=version_choice
            )
            st.session_state.recognizer = recognizer
            st.session_state.loaded_model_info = f"{real_model_type} v{version_choice}"
            st.success(f"Model {real_model_type} v{version_choice} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

if st.session_state.loaded_model_info:
    st.info(f"✅ Loaded Model: **{st.session_state.loaded_model_info}**")

st.divider()

# ---------------------------------------------------------
# TEXT INPUT + PREDICTION
# ---------------------------------------------------------
st.subheader("Enter Text")

# Show warning if model is not loaded
if st.session_state.recognizer is None:
    st.warning("⚠ Please load a model before running predictions.")
    st.stop()

user_text = st.text_input("Type a command or phrase:")

if st.button("Run Intent Recognition"):
    if not user_text.strip():
        st.warning("Enter some text first.")
        st.stop()

    recognizer = st.session_state.recognizer

    with st.spinner("Running inference..."):
        intent, probs = recognizer.predict_intent(user_text)

    st.success(f"### Predicted Intent: **{intent}**")

    # -------------------------
    # REAL-TIME DATA VISUALISATION
    # -------------------------
    if probs is not None:
        st.write("### Intent Probabilities")

        labels = recognizer.label_encoder.classes_
        probs_float = [float(p) * 100 for p in probs]  # real numeric values
        probs_str = [f"{p:.2f}%" for p in probs_float]  # pretty-formatted

        df = pd.DataFrame({
            "Intent": labels,
            "Probability": probs_float,      # numeric (no %)
            "Probability (%)": probs_str     # formatted string (for table only)
        })
        # -------------------------
        # TABLE
        # -------------------------
        st.table(df[["Intent", "Probability (%)"]])
        
        # -------------------------
        # PIE CHART
        # -------------------------
        st.write("### Probability Distribution (BAR Chart)")
        fig_bar = px.bar(
            df,
            x="Intent",
            y="Probability",
            title="Intent Probability Distribution",
            labels={"Probability": "Probability (%)"},
            text="Probability (%)"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

        # -------------------------
        # BAR CHART
        # -------------------------
        st.write("### Probability Distribution (PIE Chart)")
        fig_pie = px.pie(
            df,
            names="Intent",
            values="Probability",
            title="Intent Probability Breakdown"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

st.caption("Backend skill execution is disabled in this demo UI.")
