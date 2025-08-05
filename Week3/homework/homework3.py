import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import io

import torchaudio

sys.path.append('third_party/Matcha-TTS')

import torch
import uvicorn
import whisper
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig
from dotenv import load_dotenv

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

@asynccontextmanager
async def lifespan(app: FastAPI):
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir.parent.parent / ".env"
    was_loaded = load_dotenv(dotenv_path=dotenv_path)
    print(f"Was .env file loaded? {was_loaded}")

    # This code runs ONCE when the Uvicorn worker starts
    print("--- Lifespan event: Loading models... ---")

    # Hugging Face Login
    hf_token = os.getenv('MY_HUGGING_FACE_TOKEN')
    login(token=hf_token)

    # Load and store the ASR model in the app's state
    app.state.asr_model = whisper.load_model("small")
    print("ASR model loaded.")

    # Load TTS
    app.state.tts_engine = CosyVoice(model_dir='pretrained_models/CosyVoice-300M-SFT', fp16=True)
    print("TTS engine loaded.")

    available_speakers = app.state.tts_engine.list_available_spks()

    print("================================================")
    print("Speakers available for your loaded model:")
    print(available_speakers)
    print("================================================")

    # --- LLM Loading ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading LLM...")
    # Load and store the LLM in the app's state
    app.state.llm = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={
            "quantization_config": quantization_config,
            "device_map": "auto",
        }
    )
    print("LLM loaded.")

    # Verify CUDA usage
    model_device = app.state.llm.model.device
    print("-" * 50)
    print(f"✅ Model is running on device: {model_device}")
    print("-" * 50)

    yield

    # Code below yield runs on shutdown (optional)
    print("--- Lifespan event: Cleanup complete. ---")

app = FastAPI(lifespan=lifespan)

conversation_history = []

def transcribe_audio(asr_model, audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav", language="en")

    return result["text"]

def generate_response(llm, user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    prompt = ""
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"

    outputs = llm(prompt, max_new_tokens=100, return_full_text=False)

    bot_response = outputs[0]["generated_text"]
    conversation_history.append({"role": "assistant", "text": bot_response})

    return bot_response


def synthesize_speech_to_file(tts, text, speaker='英文女', filename="response.wav"):
    print(f"Synthesizing with speaker '{speaker}' and saving with torchaudio...")

    for result in tts.inference_sft(text, speaker, stream=False):
        audio_data = result['tts_speech']
        sample_rate = tts.sample_rate

        # Ensure the tensor is on the CPU before saving.
        audio_data_cpu = audio_data.cpu()

        # torchaudio.save expects a 2D tensor (channels, time).
        # If the output tensor is 1D, we add a channel dimension.
        if audio_data_cpu.dim() == 1:
            audio_data_cpu = audio_data_cpu.unsqueeze(0)

        # Save the audio using torchaudio, as shown in the official examples.
        torchaudio.save(filename, audio_data_cpu, sample_rate)

        print(f"Successfully saved audio to {filename}")
        break

    return filename

@app.post("/chat/")
async def chat_endpoint(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()

    asr_model = request.app.state.asr_model
    llm = request.app.state.llm
    tts = request.app.state.tts_engine

    user_text = transcribe_audio(asr_model, audio_bytes)
    print(f"DEBUG: Transcribed text -> '{user_text}'")

    bot_text = generate_response(llm, user_text)
    print(f"DEBUG: LLM response -> '{bot_text}'")

    file_path = synthesize_speech_to_file(tts, bot_text, filename="response.wav")
    print(f"DEBUG: Generated TTS response and saved to {file_path}")

    return FileResponse(file_path, media_type="audio/wav", filename="response.wav")

if __name__ == "__main__":
    uvicorn.run("homework3:app", host="127.0.0.1", port=8000, reload=True)