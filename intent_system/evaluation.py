# intent_system/evaluation.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from core import config
from intent_system.preprocess import preprocess_text
from core.logger import log


# ---------------------------
# USER SELECTION HELPERS
# ---------------------------
def choose_model_family():
    print("\nAvailable Model Families:")
    model_fams = list(config.MODEL_TYPES.keys())

    for i, fam in enumerate(model_fams, 1):
        print(f"{i}. {fam}")

    choice = int(input("Choose model type: "))
    return model_fams[choice - 1]


def choose_model_version(model_type):
    model_dir = config.MODEL_TYPES[model_type]
    versions = []

    for f in os.listdir(model_dir):
        if f.startswith("classifier_v") and f.endswith(".pkl"):
            versions.append(f.split("_v")[1].split(".")[0])

    if not versions:
        raise FileNotFoundError(f"No classifiers found in {model_dir}")

    print("\nAvailable Versions:")
    for i, v in enumerate(versions, 1):
        print(f"{i}. v{v}")

    choice = int(input("Choose version: "))
    return versions[choice - 1]


def choose_dataset():
    print("\nAvailable Datasets:")
    datasets = [f for f in os.listdir(config.DATASET_DIR) if f.endswith(".csv")]

    for i, d in enumerate(datasets, 1):
        print(f"{i}. {d}")

    choice = int(input("Choose dataset: "))
    return datasets[choice - 1]


# ---------------------------
# ARTIFACT LOADING
# ---------------------------
def load_classifier(model_type, version):
    model_dir = config.MODEL_TYPES[model_type]
    classifier_path = os.path.join(model_dir, f"classifier_v{version}.pkl")
    classifier = joblib.load(classifier_path)
    return classifier


# ---------------------------
# MAIN EVALUATION LOGIC
# ---------------------------
def main():
    model_type = choose_model_family()
    version = choose_model_version(model_type)
    dataset_name = choose_dataset()

    print(f"\n[Evaluation] Using model: {model_type}_v{version}")
    print(f"[Evaluation] Using dataset: {dataset_name}")

    classifier = load_classifier(model_type, version)

    df = pd.read_csv(os.path.join(config.DATASET_DIR, dataset_name))

    texts, labels = [], []

    for t, l in zip(df["text"], df["intent"]):
        _, cleaned = preprocess_text(t)
        if cleaned.strip():
            texts.append(cleaned)
            labels.append(l)

    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(texts)

    # IMPORTANT: temporary label encoder
    temp_encoder = LabelEncoder()
    encoded_labels = temp_encoder.fit_transform(labels)

    preds = classifier.predict(embeddings)

    print("\n========= Evaluation Report =========")
    print("Accuracy:", accuracy_score(encoded_labels, preds))
    print("\nClassification Report:")
    print(classification_report(encoded_labels, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(encoded_labels, preds))
    print("=====================================")


if __name__ == "__main__":
    main()
