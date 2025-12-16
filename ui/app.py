import streamlit as st

# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="IntentIQ Demo",
    page_icon="🤖",
    layout="centered"
)

# --------------------------------------------------------------------
# TITLE + DESCRIPTION
# --------------------------------------------------------------------
st.title("🤖 Intent IQ")
st.markdown("""
### Your Personal Voice + Text Intent Recognition System  

IntentIQ is an intelligent command understanding engine built with:
- **Sentence Transformer Embeddings**
- **Versioned ML Classifiers (LR / SVC / NeuralNet-ready)**
- **Dynamic Skill Routing System**
- **Optional Vosk-based Speech-to-Text (Offline)**

This online demo shows the project UI only.  
The full engine runs offline due to heavy ML model requirements.
""")

st.divider()

# --------------------------------------------------------------------
# BUTTONS FOR INTERACTION
# --------------------------------------------------------------------
st.subheader("Try The Interface")

col1, col2 = st.columns(2)

with col1:
    text_btn = st.button("📝 Text Interface", type="primary")

with col2:
    voice_btn = st.button("🎤 Voice Interface (Disabled)")

# --------------------------------------------------------------------
# TEXT MODE (works)
# --------------------------------------------------------------------
if text_btn:
    st.write("### Text Mode Selected")
    user_input = st.text_input("Type something:")

    if user_input:
        st.success(f"📥 You typed: `{user_input}`")
        st.info("⚠️ The backend intent engine is not deployed online yet.")
        st.caption("This demo UI will be integrated with the real engine later.")

# --------------------------------------------------------------------
# VOICE MODE (not functional online)
# --------------------------------------------------------------------
if voice_btn:
    st.error("🎤 Voice mode disabled")
    st.caption("""
**Why?**  
The Vosk STT model used by IntentIQ is ~2GB and cannot run in a browser environment.
Online deployment requires GPU server hosting which is not used for this portfolio site.
""")

