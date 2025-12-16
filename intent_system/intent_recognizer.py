# intent_system/intent_recognizer.py

import os
import json
import joblib
from sentence_transformers import SentenceTransformer
from utils.ensure_transformer import get_transformer_model

from core import config
from core.logger import log


class IntentRecognizer:
    """
    Loads a specific family of models (LR, SVC, NeuralNet) 
    and a specific version (v1, v2, ...) from:
    
    models/intent_models/<MODEL_TYPE>/classifier_vX.pkl
    models/intent_models/<MODEL_TYPE>/label_encoder_vX.pkl
    models/intent_models/<MODEL_TYPE>/metadata.json
    """

    def __init__(self, model_type=None, version=None, interactive=False):
        """
        model_type : "LR" | "SVC" | "NeuralNet"
        version    : "1", "2", "3", ...
        interactive: If True → ask the user which model to load
        """

        if interactive:
            model_type = self._ask_model_type()
            version = self._ask_model_version(model_type)

        if model_type not in config.MODEL_TYPES:
            raise ValueError(f"[Recognizer] Unknown model type: {model_type}")

        self.model_type = model_type
        self.version = version
        self.model_dir = config.MODEL_TYPES[model_type]

        self.embedding_model = None
        self.classifier = None
        self.label_encoder = None
        self.metadata = {}

        self._load_models()

    # =====================================================
    # USER INPUT HELPERS (optional)
    # =====================================================

    def _ask_model_type(self):
        print("\nAvailable Model Types:")
        model_types = list(config.MODEL_TYPES.keys())
        for i, m in enumerate(model_types, 1):
            print(f"{i}. {m}")

        idx = int(input("Select model type: "))
        selected = model_types[idx - 1]
        print(f"[Recognizer] Selected model family: {selected}")
        return selected

    def _ask_model_version(self, model_type):
        versions = self._available_versions(model_type)

        print("\nAvailable Versions:")
        for i, v in enumerate(versions, 1):
            print(f"{i}. v{v}")

        idx = int(input("Select version: "))
        selected = versions[idx - 1]
        print(f"[Recognizer] Selected version: v{selected}")
        return selected

    # =====================================================
    # VERSION DISCOVERY
    # =====================================================

    def _available_versions(self, model_type):
        model_dir = config.MODEL_TYPES[model_type]
        versions = []

        for f in os.listdir(model_dir):
            if f.startswith("classifier_v") and f.endswith(".pkl"):
                ver = f.split("_v")[1].split(".")[0]
                versions.append(ver)

        if not versions:
            raise FileNotFoundError(f"[Recognizer] No classifier files found in {model_dir}")

        return sorted(versions, key=lambda x: int(x))

    # =====================================================
    # PATH RESOLUTION
    # =====================================================

    def _resolve_path(self, prefix, version):
        """
        Finds a classifier/encoder path either by:
        - exact version provided, OR
        - automatically loading latest version.
        """
        if version:
            # Load specific version
            filename = f"{prefix}_v{version}.pkl"
            full_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"[Recognizer] {filename} not found.")
            return full_path

        # Auto-load latest version
        versions = self._available_versions(self.model_type)
        latest = versions[-1]
        filename = f"{prefix}_v{latest}.pkl"
        return os.path.join(self.model_dir, filename)

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def _load_models(self):
        log.info(f"[Recognizer] Loading {self.model_type} model...")

        # Load embedding model
        self.embedding_model = get_transformer_model()
        log.info("[Recognizer] Embedding model loaded.")

        # Load classifier
        classifier_path = self._resolve_path("classifier", self.version)
        self.classifier = joblib.load(classifier_path)
        log.info(f"[Recognizer] Loaded classifier: {classifier_path}")

        # Load label encoder
        encoder_path = self._resolve_path("label_encoder", self.version)
        self.label_encoder = joblib.load(encoder_path)
        log.info(f"[Recognizer] Loaded label encoder: {encoder_path}")

        # Load optional metadata
        # Load versioned metadata
        meta_filename = f"metadata_v{self.version}.json"
        metadata_path = os.path.join(self.model_dir, meta_filename)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            log.info(f"[Recognizer] Loaded {meta_filename}")
        else:
            log.warn(f"[Recognizer] No metadata file found for version v{self.version}.")


    # =====================================================
    # INFERENCE
    # =====================================================

    def predict_intent(self, text):
        embedding = self.embedding_model.encode([text])
        pred_class = self.classifier.predict(embedding)[0]
        label = self.label_encoder.inverse_transform([pred_class])[0]

        # Probability support
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(embedding)[0]
        else:
            probs = None

        return label, probs
