import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
import whisper
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig
from dotenv import load_dotenv

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

    yield

    # Code below yield runs on shutdown (optional)
    print("--- Lifespan event: Cleanup complete. ---")

app = FastAPI(lifespan=lifespan)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an helpful AI assistant. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription."
}

conversation_history = []

def transcribe_audio(asr_model, audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav", language="en")

    return result["text"]


def generate_response(llm, user_text):
    # Frame the user's message to give context to the model.
    conversation_history.append({"role": "user", "content": user_text})

    # Get the last 5 turns of the conversation.
    recent_conversation = conversation_history[-5:]

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

@app.post("/chat/")
async def chat_endpoint(request: Request, file: UploadFile = File(...)):
    audio_bytes = await file.read()

    asr_model = request.app.state.asr_model
    llm = request.app.state.llm

    user_text = transcribe_audio(asr_model, audio_bytes)
    print(f"DEBUG: Transcribed text -> '{user_text}'")

    bot_text = generate_response(llm, user_text)
    print(f"DEBUG: LLM response -> '{bot_text}'")

    return FileResponse("response.wav", media_type="audio/wav", filename="response.wav")

if __name__ == "__main__":
    uvicorn.run("homework3:app", host="127.0.0.1", port=8000, reload=True)