# Week 5 Homework: Hybrid Retrieval System

This project implements a hybrid search system combining semantic search (FAISS) with keyword search (SQLite FTS5), evaluated using Reciprocal Rank Fusion (RRF) + BM25 ranking.

<img width="1461" height="590" alt="image" src="https://github.com/user-attachments/assets/4879db1b-e648-4aa4-9f75-2e521d40c2d7" />

## System Components

- **Semantic Search**: FAISS vector search with sentence transformers
- **Keyword Search**: SQLite FTS5 full-text search  
- **Hybrid Search**: RRF fusion of semantic + keyword results
- **Evaluation**: Hit Rate@3 metrics across 12 test queries

## Quick Start

### 1. Build the Index
```bash
# Download papers, extract text, create chunks, and build FAISS index
python fetch_arxiv.py
```

### 2. Start the Search API
```bash
# Start FastAPI server on http://127.0.0.1:8000
python search_api.py
```

### 3. Test the Endpoints

**Semantic Search:**
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "neural network architecture", "k": 3}'
```

**Keyword Search:**
```bash
curl -X POST "http://127.0.0.1:8000/keyword_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention", "k": 3}'
```

**Hybrid Search:**
```bash
curl -X POST "http://127.0.0.1:8000/hybrid_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning optimization", "k": 3}'
```

## Evaluation

📊 **[View Complete Evaluation Notebook](rag_eval.ipynb)**

The evaluation notebook compares all three search methods using Hit Rate@3 on 12 ML/AI domain queries.

**Results Summary:**
- Semantic Search: 100% hit rate
- Keyword Search: 50% hit rate  
- Hybrid Search: 100% hit rate

## File Structure

```
├── fetch_arxiv.py          # Data pipeline (fetch → extract → chunk → index)
├── search_api.py           # FastAPI server with 3 search endpoints
├── settings.py             # Configuration
├── sql_util.py             # SQLite database utilities
├── rag_eval.ipynb         # Evaluation notebook with metrics & analysis
├── data/
│   ├── raw_pdfs/          # Downloaded arXiv papers
│   ├── processed/         # SQLite databases (documents.db, chunks.db)
│   └── index/             # FAISS index + metadata
└── README.md              # This file
```

## Key Features

✅ **SQLite + FAISS Hybrid Architecture**  
✅ **FTS5 Full-Text Search**  
✅ **Reciprocal Rank Fusion (RRF) + BM25 ranking**  
✅ **FastAPI REST Endpoints**  
✅ **Comprehensive Evaluation Framework**  

## Requirements

- Python 3.8+
- Dependencies: `pip install -r requirements.txt` (if available)
- Key packages: faiss-cpu, sentence-transformers, fastapi, sqlite3

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/healthz
