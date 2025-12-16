import os
import numpy as np
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class VoskSTT:
    def __init__ (self, model_path = "models/voice_models/vosk"):
        #Load Vosk model for real-time command recognition.
        print ("[Vosk] initializing...")
        os.mkdirs(model_path, exist_ok = True)
        
        if not os.listdir(model_path):
            raise RuntimeError("❌ No Vosk model found! Download a model into: " + model_path)
        self.model = Model(model_path)
        self.sample_rate = 16000  # Vosk required mic rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)

        self.audio_q = queue.Queue()
        self.last_audio = None  # Used by Whisper fallback
        
        print ("[VOSK] ready..")
        
    def _audio_callback(self, indata, frames, time, status):
        #Store live audio chunks into queue.
        if status:
            print(f"[Audio Status] {status}")
        
        self.audio_q.put(bytes(indata))
        
    def transcribe(self, duration = 2):
        #Capture & recognize short voice commands.
        #Returns: text, confidence_score (0–1)
        
        print(f"[Vosk] listening for {duration}s...")
        self.audio_q = queue.Queue() #clear old audio
        
        #Start non-blocking stream
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            sd.sleep(duration * 1000)
        
        audio_bytes = b"".join(list(self.audio_q.queue))
        self.last_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.recognizer.AcceptWaveform(audio_bytes):
            result_json = json.loads(self.recognizer.Result())
        else:
            result_json = json.loads(self.recognizer.PartialResult())

        text = result_json.get("text", "").strip()
        confidence = 0.0

        # Confidence available only in full results
        if "result" in result_json and len(result_json["result"]) > 0:
            conf_scores = [w["conf"] for w in result_json["result"] if "conf" in w]
            if conf_scores:
                confidence = sum(conf_scores) / len(conf_scores)

        print(f"[VOSK] → {text} (conf: {confidence:.2f})")
        return text, confidence
        