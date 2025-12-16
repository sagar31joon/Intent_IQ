# utils/ensure_transformer.py

import os
from sentence_transformers import SentenceTransformer
from core import config

def get_transformer_model():
    model_dir = config.TRANSFORMER_PATH

    # If folder already contains model → load it
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"[TRANSFORMER] Using cached model at {model_dir}")
        return SentenceTransformer(model_dir)

    # Folder exists but empty → first-time download
    print("[TRANSFORMER] Downloading model for first-time setup...")
    os.makedirs(model_dir, exist_ok=True)

    # IMPORTANT: download using model *name*, not folder path
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    # Save to disk
    model.save(model_dir)

    print(f"[TRANSFORMER] Saved model to {model_dir}")
    return model
