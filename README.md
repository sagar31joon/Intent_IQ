# IntentIQ — A Modular Intent Recognition System

IntentIQ is a fully modular, extensible intent recognition engine built for **voice** and **text** inputs.  
It combines classical ML classifiers, sentence-transformer embeddings, dynamic skill routing, 
and optional offline speech-to-text using Vosk.

This repository includes:
- A CLI engine for real-time intent recognition  
- A Streamlit web UI for demo and deployment  
- Trainable, versioned ML models  
- Auto-discovery and auto-generation of skill modules  
- Transformer model caching for offline inference  

---

# 🚀 Features Overview

### 🔹 1. Text & Voice Input Support
- **Text mode:** Fully functional in CLI + Streamlit UI  
- **Voice mode:** Powered by Vosk (offline STT), available only in CLI  
- Online demos disable voice mode due to 2GB Vosk model size

### 🔹 2. Multi-Model Architecture
Supports multiple ML families with versioning:
- **LR** (Logistic Regression)  
- **SVC** (Support Vector Classifier)  
- **NeuralNet** (Reserved for future expansion)

### 🔹 3. Versioned Models
Every trained model is saved in:
```
models/intent_models/<MODEL_TYPE>/classifier_vX.pkl
models/intent_models/<MODEL_TYPE>/label_encoder_vX.pkl
models/intent_models/<MODEL_TYPE>/metadata_vX.json
```

The engine supports:
- Loading any model type  
- Loading any version  
- Always backward compatible  

### 🔹 4. Dynamic Skill Routing
Each predicted intent maps to a Python skill file:
```
skills/<intent>.py
```
If a skill does **not** exist, the router **auto-creates** a placeholder module:
```
def run(text):
    print('Placeholder skill executed for intent: <intent>')
```

### 🔹 5. Offline Transformer Caching
The embedding model `all-MiniLM-L6-v2` is downloaded **once**, saved inside:
```
models/transformer_model/
```
Subsequent runs load it instantly without re-downloading.

### 🔹 6. Streamlit Web UI
The UI provides:
- Model selection (family(model_type) + version)
- Load Model button
- Text prediction
- Probability visualization
- Fully client-friendly layout for deployment on Render

---

# 📁 Project Structure

```
IntentIQ_Lappy/
│
├── core/
│   ├── config.py        # Paths, constants, model directories
│   ├── engine.py        # Main CLI engine
│   ├── router.py        # Dynamic skill routing
│   └── logger.py        # Logging utilities
│
├── intent_system/
│   ├── preprocess.py            # Text cleanup, wake word, fillers, lemmatization
│   ├── intent_recognizer.py     # Loads embeddings + classifier
│   ├── trainer.py               # Dataset preprocessing, embedding, training
│   ├── evaluation.py            # Full evaluation pipeline
│   ├── model_handlers.py        # LR, SVC handlers
│   └── ...
│
├── io_layer/
│   ├── stt_vosk.py      # Offline speech recognition
│   └── audio_utils.py
│   
│
├── models/
│   ├── intent_models/
│   │   ├── LR/
│   │   ├── SVC/
│   │   └── NeuralNet/   # Coming in Future
│   ├── transformer_model/  # Cached transformer
│   └── voice_models/
│
├── ui/
│   └── app.py           # Streamlit user interface
│
├── skills/
│   ├── greeting.py
│   ├── get_time.py
│   ├── weather_query.py
│   ├── open_app.py
│   ├── general_conversation.py
│   └── exit.py
│
├── utils/
│   ├── ensure_transformer.py
│   └── file_utils.py
│
├── logs/
│   └── intent_iq.log
│
├── dataset/
│   └── intents.csv
│
├── main.py              # CLI entrypoint
└── README.md
```

---

# 🧠 System Architecture (Deep Explanation)

## 1. Input Layer
### ✔ Text Input  
Direct string input (CLI or UI)

### ✔ Voice Input  
Pipeline:
```
Microphone → RawAudio → VoskSTT → Recognized text → Preprocess → IntentRecognizer
```

---

## 2. Preprocessing Layer
Functions performed:
- Lowercasing
- Punctuation cleanup
- Wake-word detection (“lappy”)
- Filler-word removal (uh, umm, please…)
- Lemmatization (spaCy optional)

Returns:
```
(has_wake_word, cleaned_text)
```

---

## 3. Embeddings Layer (Transformer)
Embedding model:  
`all-MiniLM-L6-v2`

Workflow:
```
Raw text → preprocess → transformer.encode() → 384-dim embedding vector
```

Caching ensures fast inference offline.

---

## 4. Intent Recognition Layer
- Loads classifier + label encoder for chosen model + version
- Produces:
```
intent_label, probability_distribution
```

---

## 5. Router Layer
Maps recognized intent to:
```
skills/<intent>.py
```

If missing → auto-created placeholder.

Executes:
```
run(text)
```

---

## 6. Skills Layer
Simple Python scripts performing actions:
- greet
- tell time
- fetch weather
- exit system
- general conversation
- open apps

Fully extensible.

---

## 7. Streamlit UI (Demo Layer)
3 sections:

### ✔ 1. Project Information  
Explains system functionality.

### ✔ 2. Model Selection  
User picks:
- model family (LR/SVC)
- version (v1/vX)
- clicks **Load Model**

### ✔ 3. Prediction Panel  
User enters text → system outputs:
- intent
- probability table
- real-time data visualition as the model predicts the intents      # Coming in Future.

---

# 📊 Training Pipeline

### 1. Load dataset
`dataset/intents.csv`

### 2. Preprocess text
Uses same cleaning pipeline as inference.

### 3. Encode using transformer
Creates an embedding for each sample.

### 4. Train using chosen ML handler
- LR → logistic regression
- SVC → radial-basis SVM

### 5. Save artifacts
Classifier, label encoder, metadata.

---

# 🧪 Evaluation Pipeline
`evaluation.py` provides:
- Accuracy
- Classification report
- Confusion matrix

---

# 💻 Running the CLI Engine

### Command:
```
python3 main.py
```

### Workflow:
1. Select input mode (voice/text)
2. Select model family
3. Select version
4. System enters real-time inference loop

---

# 🌐 Running Streamlit UI

### Command:
```
streamlit run ui/app.py
```

Available online demo features:
- Text-only predictions  
- Model selection  
- Intent + probability display + Real-Time Data visualition (coming in future)

Voice mode is disabled for deployment.

---

# 🚀 Deployment Guide (Render)

### 1. Create `requirements.txt`
Include:
```
streamlit
sentence-transformers
scikit-learn
pandas
joblib
numpy
```

### 2. Deploy Streamlit app  
Render will:
- Start the app using `streamlit run ui/app.py`
- Provide a public URL  
- You can link this URL in your portfolio

---

# 🏁 Future Enhancements
- NeuralNet classifier support
- Real skill implementations
- Live probability charts in UI
- Task automation integrations
- Full web-based STT through WebRTC (future)

---

# 🙌 Credits
Built by **Sagar Joon**, with guidance and system refinement using AI-assisted architecture.  
IntentIQ is designed as a modular showcase of ML engineering, inference systems, and UI integration.

