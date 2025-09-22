# RAG2 - Enhanced Document Search & Chat System

A modern RAG (Retrieval-Augmented Generation) system that fetches academic papers from ArXiv, processes them with advanced text cleaning, and provides intelligent search capabilities through a chat interface.

## Quick Start

1. **Generate RAG Database** (follow order below)
2. **Run the Server** with `python server_api.py`
3. **Chat with your documents** through the web interface

---

## RAG2 Pipeline - Building Your Database

The `rag2/` folder contains scripts to build your RAG database. **Run these in order:**

### 1. Fetch Papers (`rag2/rag2_fetch.py`)
Downloads research papers from ArXiv based on keywords.

```bash
python rag2/rag2_fetch.py
```

**What it does:**
- Searches ArXiv by specific keywords (not categories)
- Downloads papers from 2020+ only
- Prevents duplicate downloads
- Saves PDFs to `rag2/data/raw_pdfs/`

**Features:**
- ✅ Keyword-based search for better quality
- ✅ Year filtering (2020+)
- ✅ Duplicate prevention across keywords
- ✅ Robust download with retry logic

### 2. Parse PDFs (`rag2/rag2_parser.py`)
Extracts and cleans text from downloaded PDFs.

```bash
python rag2/rag2_parser.py
```

**What it does:**
- Extracts text from PDFs with layout awareness
- Removes headers/footers and academic artifacts
- Anonymizes sensitive information
- Enhanced text cleaning (removes repeated punctuation, citations)
- Filters out low-quality text (too short, figures, etc.)
- Saves clean text to `rag2/data/processed/parsed.json`

**Features:**
- ✅ Smart PDF parsing with column detection
- ✅ Text quality filtering
- ✅ Academic artifact removal (arXiv IDs, citations)
- ✅ Anonymization for privacy

### 3. Build Search Index (`rag2/rag2_build.py`)
Creates searchable embeddings and indexes.

```bash
python rag2/rag2_build.py
```

**What it does:**
- Chunks text into optimal sizes for BGE-large model
- Creates FAISS vector index for semantic search
- Builds FTS5 index for keyword search
- Generates BM25 index for re-ranking
- Saves indexes to `rag2/data/index/`

**Features:**
- ✅ BGE-large optimized chunking (512 tokens)
- ✅ Multiple search indexes (semantic + keyword + BM25)
- ✅ Quality filtering at chunk level
- ✅ BGE-specific prefixes for better embeddings

---

## Configuration

### RAG2 Settings (`rag2/rag2_config.toml`)
Customize the RAG pipeline:

```toml
[model]
embed_model  = "BAAI/bge-large-en-v1.5"
chunk_size   = 512
overlap_frac = 0.20
min_char     = 40
batch        = 8

[search]
min_score      = 0.25
min_score_fts  = 0.1
```

### Keywords in Fetch Script
Edit `rag2/rag2_fetch.py` to customize search terms:

```python
keywords = [
    "machine learning security",
    "zero-day vulnerability",
    "blockchain security",
    # Add your keywords here
]
```

---

## Running the Server

Start the chat server:

```bash
python server_api.py
```

### Server Configuration

**For Local Development:**
```python
is_local = True  # Uses localhost
```

**For Server Deployment:**
```python
is_local = False  # Uses 0.0.0.0 for external access
```

### Server Features
- 🌐 **Web Interface** - Chat with your documents
- 🔍 **Hybrid Search** - Combines semantic + keyword + BM25
- 🤖 **LLM Integration** - Qwen 2.5 models (3B/7B)
- 📝 **Notion Integration** - Save conversations to Notion
- ⚡ **Fast Response** - Optimized search and caching

---

## Project Structure

```
rag2/                           # New enhanced RAG system
├── rag2_fetch.py              # Download papers from ArXiv
├── rag2_parser.py             # Extract & clean text from PDFs
├── rag2_build.py              # Build search indexes
├── rag2_config.toml           # Configuration settings
├── rag2_sql_util.py           # Database utilities
└── data/                      # Generated data
    ├── raw_pdfs/              # Downloaded PDFs
    ├── processed/             # Cleaned text (parsed.json)
    └── index/                 # Search indexes

server_api.py                  # Main server application
search.py                      # Search engine (semantic + hybrid)
llm.py                         # LLM integration (Qwen models)
notion.py                      # Notion integration
summarize.py                   # Text summarization
```

---

## Usage Examples

### Full Pipeline
```bash
# 1. Download papers
python rag2/rag2_fetch.py

# 2. Parse PDFs
python rag2/rag2_parser.py

# 3. Build indexes
python rag2/rag2_build.py

# 4. Start server
python server_api.py
```

### Server Modes
```python
# Local development (localhost only)
is_local = True

# Server deployment (external access)
is_local = False
```

### Chat Commands
- "Search for papers about quantum computing"
- "Save this conversation to Notion"
- "Find research on zero-day vulnerabilities"

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Dependencies: `transformers`, `sentence-transformers`, `faiss`, `pymupdf`, `fastapi`, `uvicorn`

## Notes

- **GPU Recommended**: BGE-large model works much faster with CUDA
- **Disk Space**: Each paper ~1-5MB, plan accordingly
- **First Run**: Building indexes takes time, subsequent searches are fast
- **Year Filter**: Currently set to 2020+, modify in `rag2_fetch.py` if needed

---

## Advanced Usage

### Custom Keywords
Modify the keywords list in `rag2/rag2_fetch.py`:
```python
keywords = ["your", "custom", "research", "topics"]
```

### Adjust Quality Filters
Edit `rag2/rag2_config.toml`:
```toml
min_char = 100      # Longer minimum text
chunk_size = 256    # Smaller chunks
```

### Search Tuning
Modify search parameters in the configuration for better results.