# main.py
# command for running streamlit ui : venv/bin/python3 -m streamlit run ui/app.py
from core.engine import IntentIQEngine

if __name__ == "__main__":
    engine = IntentIQEngine()
    engine.run()
