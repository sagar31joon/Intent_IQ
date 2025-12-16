import sys
import os

# Add project root so absolute imports work (core.*, intent_system.*, etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import streamlit as st
import json

from core import config
from intent_system.intent_recognizer import IntentRecognizer


st.set_page_config(
    page_title="IntentIQ Demo",
    page_icon="🤖",
    layout="centered"
)

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

model_families = list(config.MODEL_TYPES.keys())
model_families_display = ["LR", "SVC", "NeuralNet (coming soon)"]

# Disable NeuralNet for now
disabled_index = model_families.index("NeuralNet")

model_choice = st.selectbox(
    "Choose a model type:",
    model_families_display,
    index=0
)

if "NeuralNet" in model_choice:
    st.warning("NeuralNet is not implemented yet.")
    st.stop()

real_model_type = model_choice  # "LR" or "SVC"

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

version_choice = st.selectbox(
    "Choose model version:",
    versions,
    index=0
)

# ---------------------------------------------------------
# INITIALIZE SELECTED MODEL
# ---------------------------------------------------------
st.divider()
st.subheader("Enter Text")

user_text = st.text_input("Type a command or phrase:")

if st.button("Run Intent Recognition"):
    if not user_text.strip():
        st.warning("Enter some text first.")
        st.stop()

    with st.spinner("Loading model and running inference..."):
        recognizer = IntentRecognizer(
            model_type=real_model_type,
            version=version_choice
        )

        intent, probs = recognizer.predict_intent(user_text)

    st.success(f"### Predicted Intent: **{intent}**")

    if probs is not None:
        st.write("### Intent Probabilities")
        labels = recognizer.label_encoder.classes_
        prob_dict = {labels[i]: f"{float(p) * 100:.2f}%" for i, p in enumerate(probs)}
        st.json(prob_dict)

st.caption("Backend skill execution is disabled in this demo UI.")
