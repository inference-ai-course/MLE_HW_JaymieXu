import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
import whisper
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig
from dotenv import load_dotenv

from TTS.api import TTS

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
        model="microsoft/Phi-3-mini-4k-instruct",
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

    print("Loading TTS")
    app.state.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(model_device)
    print("TTS Loaded")

    #print(TTS().list_models())

    yield

    # Code below yield runs on shutdown (optional)
    print("--- Lifespan event: Cleanup complete. ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an helpful AI assistant. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription."
}

conversation_history = []

def transcribe_audio(asr_model, audio_bytes):
    # Get the directory where this script lives
    script_dir = Path(__file__).resolve().parent
    temp_file = script_dir / "temp.wav"
    
    with open(temp_file, "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe(str(temp_file), language="en")

    return result["text"]


def generate_response(llm, user_text):
    # Frame the user's message to give context to the model.
    conversation_history.append({"role": "user", "content": user_text})

    # Get the last 5 turns of the conversation.
    recent_conversation = conversation_history[-10:]

    # Combine the system prompt with the recent conversation for the model.
    prompt_context = [SYSTEM_PROMPT] + recent_conversation

    # Use the tokenizer to apply the model's official chat template.
    prompt = llm.tokenizer.apply_chat_template(
        prompt_context,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate a response.
    outputs = llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Clean the output to get only the new assistant message.
    raw_bot_response = outputs[0]["generated_text"]
    bot_response = raw_bot_response.split("<|assistant|>")[-1].strip()

    # Add the bot's response to the history.
    conversation_history.append({"role": "assistant", "content": bot_response})

    return bot_response


def synthesize_speech(tts_engine, text, filename="response.wav"):
    try:
        # Get the directory where this script lives
        script_dir = Path(__file__).resolve().parent
        file_path = script_dir / filename
        
        tts_engine.tts_to_file(
            text=text,
            file_path=str(file_path),
            speaker="Ana Florence",  # A good, standard female voice
            language="en"
        )

        return filename
    except Exception as e:
        # Print a more detailed error message
        print(f"Error during TTS synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page for the voice assistant UI."""
    html_path = Path(__file__).resolve().parent / "index.html"
    if html_path.is_file():
        return html_path.read_text(encoding="utf-8-sig")
    else:
        return "<h1>Error: index.html not found</h1><p>Please make sure the index.html file is in the same directory as your Python script.</p>", 404


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = Path(__file__).resolve().parent / filename
    if ".." in filename or "/" in filename:
        return {"error": "Invalid filename"}, 400
    if audio_path.is_file():
        # Create the response object first
        response = FileResponse(audio_path, media_type="audio/wav")

        # Add headers to prevent caching on the browser side.
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response
    return {"error": "File not found"}, 404

@app.post("/chat/")
async def chat_endpoint(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()

    asr_model = request.app.state.asr_model
    llm = request.app.state.llm
    tts = request.app.state.tts

    user_text = transcribe_audio(asr_model, audio_bytes)
    print(f"DEBUG: Transcribed text -> '{user_text}'")

    # If transcription is empty, don't bother with the LLM
    if not user_text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected in audio."}
        )

    bot_text = generate_response(llm, user_text)
    print(f"DEBUG: LLM response -> '{bot_text}'")

    output_filename = "response.wav"
    output_audio_path = synthesize_speech(tts, bot_text, filename=output_filename)

    if output_audio_path:
        return JSONResponse(content={
            "user_text": user_text,
            "bot_text": bot_text,
            "audio_url": f"/audio/{output_filename}"  # Provide a URL to the audio
        })
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio response."}
        )

if __name__ == "__main__":
    uvicorn.run("homework6:app", host="127.0.0.1", port=8000, reload=True)