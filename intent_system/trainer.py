# intent_system/trainer.py

import os
import json
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from core import config
from core.logger import log
from intent_system.preprocess import preprocess_text

from intent_system.model_handlers import (
    LogisticRegressionHandler,
    SVCHandler,
)


# Register available models
MODEL_REGISTRY = {
    "LR": LogisticRegressionHandler,
    "SVC": SVCHandler,
    # "NeuralNet": NeuralNetHandler   # later
}


# ================================================================
# Dataset loader
# ================================================================
def load_dataset(dataset_name):
    path = os.path.join(config.DATASET_DIR, dataset_name)
    print(f"[TRAINER] Loading dataset: {dataset_name}")
    return pd.read_csv(path)


def preprocess_dataset(df):
    print("[TRAINER] Preprocessing dataset...")

    cleaned_texts, valid_labels = [], []

    for text, label in zip(df["text"], df["intent"]):
        _, cleaned = preprocess_text(text)
        if cleaned.strip():
            cleaned_texts.append(cleaned)
            valid_labels.append(label)

    return cleaned_texts, valid_labels


# ================================================================
# Embeddings
# ================================================================
def create_embeddings(texts):
    print(f"[TRAINER] Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("[TRAINER] Creating sentence embeddings...")
    embeddings = embedder.encode(texts)

    return embeddings


# ================================================================
# Versioning
# ================================================================
def get_next_version(model_type, prefix):
    model_dir = config.MODEL_TYPES[model_type]
    os.makedirs(model_dir, exist_ok=True)

    versions = []

    for f in os.listdir(model_dir):
        if f.startswith(prefix) and f.endswith(".pkl"):
            ver = int(f.split("_v")[1].split(".")[0])
            versions.append(ver)

    return max(versions) + 1 if versions else 1


# ================================================================
# Save artifacts
# ================================================================
def save_artifacts(model_type, version, classifier, label_encoder,
                   dataset_name, cleaned_texts, labels):

    model_dir = config.MODEL_TYPES[model_type]

    classifier_path = os.path.join(model_dir, f"classifier_v{version}.pkl")
    encoder_path = os.path.join(model_dir, f"label_encoder_v{version}.pkl")
    metadata_path = os.path.join(model_dir, f"metadata_v{version}.json")

    joblib.dump(classifier, classifier_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "model_type": model_type,
        "version": version,
        "samples": len(cleaned_texts),
        "unique_labels": len(set(labels)),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "dataset_used": dataset_name
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    log.info("\n[TRAINER] Saved:")
    log.info(f" → {classifier_path}")
    log.info(f" → {encoder_path}")
    log.info(f" → {metadata_path}")


# ================================================================
# Main trainer loop
# ================================================================
def main():

    # -------------------------
    # Select model
    # -------------------------
    model_types = list(config.MODEL_TYPES.keys())
    print("\nAvailable Model Types:")
    for i, m in enumerate(model_types, 1):
        print(f"{i}. {m}")

    model_choice = int(input("Select model to train: "))
    model_type = model_types[model_choice - 1]

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"[TRAINER] Model '{model_type}' not implemented.")

    handler_class = MODEL_REGISTRY[model_type]
    model_handler = handler_class()

    # -------------------------
    # Select dataset
    # -------------------------
    datasets = [f for f in os.listdir(config.DATASET_DIR) if f.endswith(".csv")]
    print("\nAvailable Datasets:")
    for i, d in enumerate(datasets, 1):
        print(f"{i}. {d}")

    dataset_choice = int(input("Select dataset: "))
    dataset_name = datasets[dataset_choice - 1]

    df = load_dataset(dataset_name)
    cleaned_texts, labels = preprocess_dataset(df)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    embeddings = create_embeddings(cleaned_texts)

    # -------------------------
    # Train model
    # -------------------------
    classifier = model_handler.train(embeddings, encoded_labels)

    # -------------------------
    # Save?
    # -------------------------
    choice = input("\n[TRAINER] Save model? (Y/N): ").strip().lower()
    if choice in ("y", "yes"):
        version = get_next_version(model_type, "classifier")
        save_artifacts(model_type, version, classifier, label_encoder,
                       dataset_name, cleaned_texts, labels)
    else:
        log.info("[TRAINER] Model NOT saved.")

    log.info("\n[TRAINER] Training complete!")


if __name__ == "__main__":
    main()
