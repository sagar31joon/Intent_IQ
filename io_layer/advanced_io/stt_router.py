#Decides which model handles the request (vosk or whisper)

from io_layer.stt_vosk import VoskSTT
from io_layer.stt_whisper import WhisperSTT

class STTRouter:
    def __init__(self):
        # Load both engines once, reused forever
        self.vosk = VoskSTT()
        self.whisper = WhisperSTT(model_name = "medium")
        
        # Routing thresholds
        self.short_command_max_len = 3 #words
        self.low_conf_threshold = 0.75 #below this confidence, fallback to whisper
        
    def transcribe(self, duration = 3):
        #Step 1: Real-time transcription with Vosk
        #Step 2: Check confidence score
        #Step 3: Decide if whisper fallback is needed
        
        text, confidence = self.vosk.trancribe(duration)
        
        if len(text.split()) <= self.short_command_max_len and confidence >= self.low_conf_threshold:
            return text #vosk is fine for this one
        else:
            # command is big, gotta use whisper
            audio = self.vosk.last_audio
            whisper_text = self.whisper.transcribe(audio)
            return whisper_text
        