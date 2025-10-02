# Project 2 - Cybersecurity Voice Agent

![Demo](demo.mkv)

## Overview

A voice-enabled cybersecurity assistant that combines conversational AI with speech recognition and synthesis. This agent is designed to help security professionals and developers interact with cybersecurity knowledge through natural voice conversations, providing access to security documentation and best practices.

## Project Details

### Core Technologies

- **Language Model**: Built on Qwen2.5 7B with prompt engineering for conversational and tool-use capabilities
- **RAG (Retrieval-Augmented Generation)**: Powered by LlamaIndex and ChromaDB for efficient retrieval of cybersecurity theory knowledge and documentation
- **ASR (Automatic Speech Recognition)**: OpenAI Whisper for accurate speech-to-text conversion
- **TTS (Text-to-Speech)**: Coqui TTS for natural voice synthesis
- **Agent Architecture**: Unified agent with prompt engineering (no routing needed)

### Key Features

- **Voice Interaction**: Natural voice input and output for hands-free operation
- **Security Knowledge Base**: Access comprehensive cybersecurity theory through RAG
- **Conversational AI**: Chat naturally while maintaining access to tools and knowledge
- **Speech Recognition**: High-quality transcription with OpenAI Whisper
- **Voice Synthesis**: Natural-sounding responses with Coqui TTS

## Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for faster inference)
- Audio input/output devices for voice interaction

### Step 1: Install Dependencies and Build RAG Index

Run the installation script to install all Python packages and build the RAG database:

```bash
./llm/install.sh
```

This script will:
1. Install all required Python packages from `requirements.txt`
2. Build the RAG vector database from your knowledge base

### Step 2: Run the Backend Server

Start the backend server with the Qwen2.5 7B model:

```bash
python server_api.py server
```

This will load the model and start the API server on port 8000.

### Step 3: Access the Frontend

The frontend is served directly by the backend server. Once the server is running, access the application at:

```
http://localhost:8000
```

No separate frontend launch command is needed.

## Usage

Once the frontend is running, you can:
- **Voice Mode**: Click the microphone button to speak your questions naturally
- **Text Mode**: Type questions about cybersecurity concepts
- **Knowledge Retrieval**: Query security documentation through the RAG system
- **Tool Usage**: The agent can access tools while maintaining natural conversation
- **Get assistance**: Receive help with security best practices and theory

## Architecture

```
Voice Input → Whisper (ASR) → Qwen2.5 7B Agent → RAG Search Tools
                                    ↓                    ↓
                             Text Response ← Summerizer ← LlamaIndex/ChromaDB
                                    ↓
                            Coqui TTS (Voice Output)
```

The agent uses prompt engineering to seamlessly handle both conversational queries and tool usage without needing separate routing logic.

## Notes

- **GPU Memory**: The 7B model is lightweight and runs efficiently on consumer GPUs
- **Audio Quality**: Better microphone quality improves speech recognition accuracy
- **RAG Database**: Ensure the RAG database is built before running queries that require knowledge retrieval
- **Port**: The application runs on port 8000 by default
