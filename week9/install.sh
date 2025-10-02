#!/bin/bash
sudo apt update
sudo apt install ffmpeg

pip install -r requirements.txt

# Override conqui tts dependency nightmare
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Build RAG database
echo "Building RAG database..."
python rag2/rag_build.py