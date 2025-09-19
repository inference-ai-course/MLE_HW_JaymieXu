from pathlib import Path
import whisper
from enum import Enum

class ModelSize(str, Enum):
    SMALL    = "small"
    MEDIUM   = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class Asr:
    def __init__(
        self,
        in_model_size: ModelSize = ModelSize.LARGE_V3
    ):
        self.asr_model = whisper.load_model(
            in_model_size.value,
            device="cuda"
        )
        
        print("ASR model with size {in_model_size.value} loaded With {in_compute_type.value}")


    def __del__(self):
        del self.asr_model
        
        
    def transcribe_audio(asr_model, audio_bytes, lang = "en"):
        # Get the directory where this script lives
        script_dir = Path(__file__).resolve().parent
        temp_file = script_dir / "temp.wav"
        
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        result = asr_model.transcribe(str(temp_file), language=lang)
    
        return result["text"]