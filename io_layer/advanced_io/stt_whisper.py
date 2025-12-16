import whisper
import os
import sounddevice as sd
import numpy as np

class WhisperSTT:
    def __init__(self, model_name = "base"):
        #Loads Whisper model into memory.
        #model_name: Which Whisper version to use (base)
        model_path = f"models/voice_models/{model_name}"
        os.makedirs(model_path, exist_ok=True)
        print(f"[STT] Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, download_root=model_path)
        print(f"[STT] {model_name} model loaded successfully")
        
        self.sample_rate = 44100 #my macbook prorequires 44.1khz sample rate
        
    def record_audio(self, duration):
        #Records audio from the microphone for 'duration' seconds.
        #Returns: numpy array of audio samples
        print(f"[STT] Listening for {duration} seconds...")
        audio = sd.rec(int(duration * self.sample_rate),
                        samplerate = self.sample_rate,
                        channels = 1,
                        dtype = 'float32')
        sd.wait() # Wait for recording to finish
        return audio.flatten()
    
    def transcribe(self, audio):
        #Converts audio (numpy array) → text using Whisper
        print("[STT] Transcribing audio...")
        result = self.model.transcribe(audio, fp16 = False)
        text = result.get("text", "").strip()
        return text
    
        