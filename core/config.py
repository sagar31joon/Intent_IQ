# core/config.py
"""
Central configuration file for IntentIQ.
All key paths, model directories and global settings are defined here.
This ensures the whole project stays modular and maintainable.
"""

import os

# =========================
# ROOT PATH
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# DATASET DIRECTORY
# =========================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

# =========================
# MODEL ROOT DIRECTORIES
# =========================
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
INTENT_MODEL_DIR = os.path.join(MODEL_ROOT, "intent_models")

# Structured model families
MODEL_TYPES = {
    "LR": os.path.join(INTENT_MODEL_DIR, "LR"),
    "SVC": os.path.join(INTENT_MODEL_DIR, "SVC"),
    "NeuralNet": os.path.join(INTENT_MODEL_DIR, "NeuralNet"),
}

# =========================
# VOICE MODELS
# =========================
VOICE_MODEL_DIR = os.path.join(MODEL_ROOT, "voice_models")
VOSK_MODEL_PATH = os.path.join(VOICE_MODEL_DIR, "vosk/gigaspeech")

# =========================
# EMBEDDINGS
# =========================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =========================
# AUDIO CONFIG
# =========================
SAMPLE_RATE = 16000
BLOCK_SIZE = 2000

# =========================
# LOGGING CONFIG
# =========================
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_LEVEL = "INFO"  # INFO | DEBUG | WARNING | ERROR
