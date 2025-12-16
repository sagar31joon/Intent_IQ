#stt_vosk
import os
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class VoskSTT:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        
        print ("[VOSK] lodaing model...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.recognizer.SetWords(False)
        
        self.audio_queue = queue.Queue()
        self.samplerate = 16000
        self.blocksize = 8000
    
    def _callback(self, indata, frames, time, status):
        #Audio callback pushes mic chunks into queue.
        if status:
            print(f"[VOSK ERROR] {status}", flush=True)
        self.audio_queue.put(bytes(indata))
        
    def listen(self):
        #Captures one full sentence and returns the transcribed text.
        print("\n[Listening...] Speak now.")

        with sd.RawInputStream(samplerate = self.samplerate,
                               blocksize = self.blocksize,
                               dtype = 'int16',
                               channels = 1,
                               callback = self._callback):
            while True:
                data = self.audio_queue.get()
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()

                    if text != "":
                        print(f"[Voice Captured] : {text}")
                        return text
                # else:
                #     partial = json.loads(self.recognizer.PartialResult()).get("partial", "")
                #     print("Partial:", partial)