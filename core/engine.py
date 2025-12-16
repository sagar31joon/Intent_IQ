# core/engine.py

from core.logger import log
from core import config

from io_layer.stt_vosk import VoskSTT
from intent_system.intent_recognizer import IntentRecognizer
from core.router import IntentRouter


class IntentIQEngine:
    def __init__(self):
        self.input_mode = None    # "voice" or "text"
        self.stt = None
        self.recognizer = None
        self.router = IntentRouter()
    
    def select_input_mode(self):
        print("\nSelect Input Mode:")
        print("1. Voice Input (Vosk STT)")
        print("2. Text Input (Keyboard)")

        choice = int(input("Choose mode: ").strip())

        if choice == 1:
            self.input_mode = "voice"
            log.info("[Engine] Selected Input Mode: Voice")
        else:
            self.input_mode = "text"
            log.info("[Engine] Selected Input Mode: Text")


    # ============================================================
    # MODEL SELECTION (model_type + version)
    # ============================================================
    def select_model_family(self):
        print("\nAvailable Model Families:")
        model_types = list(config.MODEL_TYPES.keys())

        for i, m in enumerate(model_types, 1):
            print(f"{i}. {m}")

        idx = int(input("Select model type: "))
        model_type = model_types[idx - 1]

        log.info(f"[Engine] Selected Model Family: {model_type}")
        return model_type

    def select_model_version(self, model_type):
        from os import listdir

        model_dir = config.MODEL_TYPES[model_type]
        versions = []

        for f in listdir(model_dir):
            if f.startswith("classifier_v") and f.endswith(".pkl"):
                versions.append(f.split("_v")[1].split(".")[0])

        if not versions:
            raise FileNotFoundError(f"[Engine] No trained versions found in {model_dir}")

        print("\nAvailable Versions:")
        for i, v in enumerate(versions, 1):
            print(f"{i}. v{v}")

        idx = int(input("Select version: "))
        version = versions[idx - 1]

        log.info(f"[Engine] Selected Version: v{version}")
        return version

    # ============================================================
    # INITIALIZE COMPONENTS
    # ============================================================
    def initialize(self):
        self.select_input_mode()

        if self.input_mode == "voice":
            log.info("[Engine] Initializing STT...")
            self.stt = VoskSTT(
                model_path=config.VOSK_MODEL_PATH,
            )

        # ---- MODEL SELECTION UI ----
        model_type = self.select_model_family()
        version = self.select_model_version(model_type)

        log.info("[Engine] Loading Intent Recognizer...")
        self.recognizer = IntentRecognizer(
            model_type=model_type,
            version=version
        )

        log.info("[Engine] Ready.\n")
    
    def shutdown(self):
        print("[Engine] Shutting down...")

        # Stop STT safely
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass

        log.info("[Engine] Clean exit.")


    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        self.initialize()

        while True:

            # Select input source
            if self.input_mode == "voice":
                print("\n[Listening...] Say something:")
                text = self.stt.listen()
            else:
                text = input("\n[You] ").strip()

            if not text:
                continue

            print(f"[User] {text}")
            
            # User-triggered shutdown
            if text.lower() in ("exit", "quit", "stop", "shutdown"):
                print("[System] Shutdown command received.")
                self.shutdown()
                break

            # Intent recognition
            intent, probs = self.recognizer.predict_intent(text)
            print(f"[Predicted Intent] {intent}")
            
            # Shutdown if predicted intent is exit
            if intent == "exit":
                print("[System] Exit intent detected. Shutting down...")
                self.shutdown()
                break

            # Probability visualization
            if probs is not None:
                labels = self.recognizer.label_encoder.classes_
                print("\n[Probabilities]")
                print("-" * 40)

                for label, p in zip(labels, probs):
                    pct = float(p) * 100
                    print(f"{label:<20} {pct:>6.2f}%")

                print("-" * 40)

            # Route to skill
            self.router.route(intent, text)
