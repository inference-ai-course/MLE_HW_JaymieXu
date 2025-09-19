from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO
from dotenv.main import DotEnv, StrPath
from dotenv import find_dotenv
from fastapi import FastAPI
from transformers import Optional
import os

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from asr import Asr
from llm import LLM
from tts import TTSEngine


def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> bool:
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    dotenv = DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=override,
        encoding=encoding,
    )
    
    return dotenv.set_as_environment_variables()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the Uvicorn worker starts
    print("--- Lifespan started... ---")
    
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir.parent / ".env"
    was_loaded = load_dotenv(dotenv_path=dotenv_path)
    
    print(f"Was .env file loaded? {was_loaded}")
    
    # --- Loading ASR ---
    app.state.asr_model = Asr()
    
    # --- Loading LLM ---
    app.state.llm = LLM()
    
    # --- Loading TTS ---
    app.state.tts = TTSEngine()
    
    yield

    # Code below yield runs on shutdown (optional)
    print("--- Lifespan endded... ---")
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    asr_model : Asr       = request.app.state.asr_model
    llm       : LLM       = request.app.state.llm
    tts       : TTSEngine = request.app.state.tts

    user_text = asr_model.transcribe_audio(audio_bytes)
    print(f"USER QUERY: {user_text}")

    # If transcription is empty, don't bother with the LLM
    if not user_text.strip():
        print("No speech detected in audio")
        return JSONResponse(
            status_code=400,
            content={"error": "No speech detected in audio."}
        )
    
    # Route the LLM output through function calling logic
    final_response = llm.generate_response(user_text)
    print(f"FINAL RESPONSE: {final_response}")
    print("=" * 50)  # Separator between queries

    # TTS
    output_filename = "response.wav"
    output_audio_path = tts.synthesize_speech(final_response, filename=output_filename)

    if output_audio_path:
        return JSONResponse(content={
            "user_text": user_text,
            "bot_text": final_response,  # Show the final response after function calling
            "audio_url": f"/audio/{output_filename}"  # Provide a URL to the audio
        })
        
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio response."}
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
    